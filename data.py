import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
from treebank import all_spans
from tokenizer import bert_tokens_and_spans
from pytorch_transformers import BertModel, BertConfig, BertTokenizer

class TaggedSpan:
    
    def __init__(self, tokens, span):
        self.tokens = tokens
        self.span = span


config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel(config)    

def vectorize_instance(tokens, span):
    (start, stop) = span
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)  # Batch size 1    
    outputs = bert(input_ids)[0].squeeze(0)
    #return outputs[stop-1] - outputs[start]
    return torch.cat([outputs[start], outputs[stop-1]])

class TaggedSpanDataset(Dataset):
  
    def __init__(self, treebank):
        self.treebank = treebank
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel(self.config)    
        result = []
        for parse in self.treebank:
            tokens, spans = bert_tokens_and_spans(parse)
            for span in spans:
                result.append({'tag': 'positive', 'instance': TaggedSpan(tokens, span)})
            nontriv_spans = [(i,j) for (i,j) in all_spans(len(tokens)) if j-i >= 2]
            for span in set(nontriv_spans) - set(spans):
                result.append({'tag': 'negative', 'instance': TaggedSpan(tokens, span)})
        random.shuffle(result)
        self.result = result

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        inst = self.result[idx]['instance']
        tag = self.result[idx]['tag']
        vec = vectorize_instance(inst.tokens, inst.span)
        return {'tag': tag, 'instance': vec}
        


class DataManager:
    """
    Wrapper for a TaggedPhraseSource that indexes the tags, and splits
    the data into train, dev, and test partitions.
    
    The DataManager also provides a mechanism for converting the phrases
    into vector embeddings.
    
    """    
    def __init__(self, train_dataset, dev_dataset, test_dataset, tags):
        
        self.dataset = {'train': train_dataset, 
                        'dev': dev_dataset, 
                        'test': test_dataset}

        self.tags = tags
        self.train = DataManager.init_sampler(self.dataset['train'])
        self.dev = DataManager.init_sampler(self.dataset['dev'])
        self.test = DataManager.init_sampler(self.dataset['test'])
        self._tag_indices = {tags[i]: i for i in range(len(tags))}
        
    def tag(self, tag_index):
        """
        Returns the tag associated with the given index.
        
        """
        return self.tags[tag_index]
        
    def tag_index(self, tag):
        """
        Returns the index associated with the given tag.
        
        """
        return self._tag_indices[tag]

    def get_sampler(self, partition):
        """
        Returns a Torch sampler for the specified partition id.
        
        Recognized partition ids: 'train', 'dev', 'test'.
        
        """
        if partition == 'train':
            return self.train
        elif partition == 'dev':
            return self.dev
        elif partition == 'test':
            return self.test
        else:
            raise Exception('Unrecognized partition: {}'.format(partition))

    def batched_loader(self, partition, batch_size):        
        """
        Returns a Torch DataLoader for the specified partition id. You must
        specify the size of each batch to load.
        
        Recognized partition ids: 'train', 'dev', 'test'.
        
        """        
        return DataLoader(self.dataset[partition], batch_size=batch_size,
                          sampler=self.get_sampler(partition))

    
    @staticmethod
    def init_sampler(dataset):        
        train_ids = list(range(len(dataset)))
        train_sampler = SubsetRandomSampler(train_ids)
        return train_sampler


