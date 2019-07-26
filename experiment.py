from data import DataManager
from twolayer import SimpleClassifier, train_net, evaluate
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import torch
from treebank import read_question_bank
from data import TaggedSpanDataset

def example():
    qbank = read_question_bank()
    dataset = TaggedSpanDataset(qbank[0][:5])
    run_binary_classification(dataset, 'negative','positive')
    

def run_binary_classification(datasource, tag1, tag2, verbose = True):    
    """
    Trains a binary classifier to distinguish between TaggedPhraseDataSource
    phrases tagged with tag1 and phrases tagged with tag2.
    
    This returns the accuracy of the binary classifier on the test
    partition.
    
    """    
    dmanager = DataManager(datasource, [tag1, tag2])
    classifier = SimpleClassifier(1536,100,2)
    net = train_net(classifier, dmanager,
                    batch_size=32, n_epochs=30, learning_rate=0.001,
                    verbose=True)
    acc, misclassified = evaluate(net, dmanager, 'test')
    if verbose:        
        for tag in sorted(dmanager.tags):
            print('{} phrases are tagged with "{}".'.format(
                    dmanager.num_phrases[tag], tag))
        print('\nERRORS:')
        for (phrase, guessed, actual) in sorted(misclassified):
            print('"{}" classified as "{}"\n  actually: "{}".'.format(
                    phrase, guessed, actual))
        print("\nOverall test accuracy = {:.2f}".format(acc))
    return net, dmanager
