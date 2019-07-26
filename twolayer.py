from torch import optim
from torch import nn
import torch.nn.functional as F
import torch
import sys

class SimpleClassifier(nn.Module): 
    """
    A simple neural network with a single ReLU activation
    between two linear layers.
    
    Softmax is applied to the final layer to get a (log) probability
    vector over the possible labels.
    
    """    
    def __init__(self, input_size, hidden_size, num_labels):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.linear1(input_vec).clamp(min=0)
        nextout = self.linear2(nextout)
        return F.log_softmax(nextout, dim=1)


def classify(net, dmanager, phrase):    
    raw_inputs = [dmanager.vectorize(phrase)]
    inputs = []
    for inp in raw_inputs:
        if inp is not None:
            inputs.append(inp)
    inputs = torch.Tensor(inputs)
    val_outputs = net(inputs)  
    outputs = [output.argmax() for output in val_outputs]
    return outputs


def evaluate(net, dmanager, partition):
    """
    Evaluates a trained neural network classifier on a partition of a
    data manager (e.g. "train", "dev", "test").

    The accuracy (i.e. percentage of correct classifications) is
    returned, along with a list of the misclassifications. Each
    misclassification is a triple (phrase, guessed, actual), where
        - phrase is the misclassified phrase
        - guessed is the classification made by the classifier
        - actual is the correct classification
    
    """    
    def accuracy(outputs, labels, phrases, dmanager):
        correct = 0
        total = 0
        misclassified = []
        for (i, output) in enumerate(outputs):
            total += 1
            if labels[i] == output.argmax():
                correct += 1
            else:
                misclassified.append((phrases[i], 
                                      dmanager.tag(output.argmax().item()),
                                      dmanager.tag(labels[i].item())))
        return correct, total, misclassified
    val_loader = dmanager.batched_loader(partition, 128)
    total_val_loss = 0
    correct = 0
    total = 0
    misclassified = []
    loss = torch.nn.CrossEntropyLoss()    
    for data in val_loader:
        inputs = torch.Tensor(data['instance'])#[dmanager.vectorize(inst) for 
                               #inst in data['instance']])
        labels = torch.LongTensor([dmanager.tag_index(c) for
                                   c in data['tag']])
            
        val_outputs = net(inputs)            
        val_loss_size = loss(val_outputs, labels)
        correct_inc, total_inc, misclassified_inc = accuracy(val_outputs, 
                                                             labels, 
                                                             data['instance'], 
                                                             dmanager)
        correct += correct_inc
        total += total_inc
        misclassified += misclassified_inc
        total_val_loss += val_loss_size.data.item()
    return correct/total, misclassified       

def train_net(net, dmanager, batch_size, n_epochs, learning_rate, verbose=True):
    """
    Trains a neural network classifier on the 'train' partition of the
    provided DataManager.
    
    The return value is the trained neural network.
    
    """    
    def log(text):
        if verbose:
            sys.stdout.write(text)
                
    train_loader = dmanager.batched_loader('train', batch_size)
    loss = torch.nn.CrossEntropyLoss()    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)    
    best_net = net
    best_acc = 0.0
    log("Training classifier.\n")
    for epoch in range(n_epochs):      
        log("  Epoch {} Accuracy = ".format(epoch))
        running_loss = 0.0
        total_train_loss = 0       
        for i, data in enumerate(train_loader, 0):
            print('{}/{}'.format(i, len(train_loader)))
            inputs = data['instance']#[dmanager.vectorize(inst) for inst in data['instance']]
            labels = [dmanager.tag_index(c) for 
                          c in data['tag']]
            inputs = torch.Tensor(inputs)
            labels = torch.LongTensor([dmanager.tag_index(c) for 
                                       c in data['tag']])
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()             
        acc, misclassified = evaluate(net, dmanager, 'dev')
        if acc > best_acc:
            best_net = net
            best_acc = acc
        log("{:.2f}".format(acc))
    return best_net

