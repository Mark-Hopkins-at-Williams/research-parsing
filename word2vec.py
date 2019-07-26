from gensim.models.keyedvectors import KeyedVectors
from os.path import join

load_w2v = KeyedVectors.load_word2vec_format

class WordVectorModel:
    """An abstract class for word vector models."""
    
    def __init__(self):
        self.base_model = None    
    
    def format_word(self, word):
        """
        Takes the provided word string, and converts it into a string
        format that is compatible with this word vector model.
        
        """
        raise NotImplementedError()

    def robust_format(self, word):
        """
        Tries a sequence of fallback strategies for formatting the
        given word.
        
        First, it checks whether self.format_word(word) can be
        found in the base model. If so, this format is returned.
        
        Second, it checks whether the lowercased version of
        self.format_word(word) can be found in the base model. If so,
        this format is returned.
        
        Finally, it checks it lowercases and removes whitespace from word,
        then calls self.format_word. If this format can be found in the base
        model, then it is returned.
        
        If nothing works, then it simply returns self.format_word(word).
        
        """
        formatted_word = self.format_word(word)
        if formatted_word in self.base_model:
            return formatted_word
        elif formatted_word.lower() in self.base_model:
            return formatted_word.lower()
        else:
            alt_format = ''.join(word.lower().split())
            if alt_format in self.base_model:
                return alt_format
            else:
                return formatted_word
        
    def get_word_vector(self, word):
        """
        Returns the word vector correponding to the given word or phrase.
        If no such vector can be found, then this returns None.
        
        """
        formatted_word = self.robust_format(word)
        if formatted_word in self.base_model:
            return self.base_model[formatted_word]
        else:
            return None

        
    def similarity(self, word1, word2):
        """
        Computes the cosine similarity of the vectors corresponding
        to the provided words.
        
        """        
        return self.base_model.similarity(self.robust_format(word1), 
                                          self.robust_format(word2))


class FreebaseModel(WordVectorModel):
    """
    Skip-gram vectors trained on 100B words from various news articles, 
    using the deprecated /en/ naming from Freebase.
    
    The file "freebase-vectors-skipgram1000-en.bin" is downloadable from:
    
    https://docs.google.com/file/d/0B7XkCwpI5KDYeFdmcVltWkhtbmM/edit?usp=sharing
    
    """
    def __init__(self, vectors_dir):
        WordVectorModel.__init__(self)
        self.base_model = load_w2v(join(vectors_dir,
                                   "freebase-vectors-skipgram1000-en.bin",),
                              binary=True)
 
    def format_word(self, word):
        return '/en/' + '_'.join(word.strip().split()).lower()


class GoogleNewsModel(WordVectorModel):
    """
    Pre-trained skip-gram vectors trained on part of Google News dataset
    (about 100 billion words). The model contains 300-dimensional
    vectors for 3 million words and phrases.
    
    The file "GoogleNews-vectors-negative300.bin" is downloadable from:
        
    https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    
    """
    def __init__(self, vectors_dir):
        WordVectorModel.__init__(self)
        self.base_model = load_w2v(join(vectors_dir,
                                   "GoogleNews-vectors-negative300.bin"),
                              binary=True)
 
    def format_word(self, word):
        return '_'.join(word.strip().split())

    
class GloveModel6B(WordVectorModel):
    """
    Pre-trained Glove vectors trained on 6 billion tokens from 
    Wikipedia 2014 + Gigaword 5 (400K vocab, uncased, 300d vectors).
    
    The file "glove.6B.300d.bin" is downloadable from:
        
    https://nlp.stanford.edu/projects/glove/
    
    """    
    def __init__(self, vectors_dir):
        WordVectorModel.__init__(self)
        self.base_model = load_w2v(join(vectors_dir, "glove.6B.300d.bin"))
 
    def format_word(self, word):
        return '_'.join(word.lower().strip().split())

    
class GloveModel42B(WordVectorModel):
    """
    Pre-trained Glove vectors trained on 42 billion tokens from 
    Common Crawl (1.9M vocab, uncased, 300d vectors).
    
    The file "glove.42B.300d.w2v" is downloadable from:
        
    https://nlp.stanford.edu/projects/glove/
    
    """ 
    def __init__(self, vectors_dir):
        WordVectorModel.__init__(self)
        self.base_model = load_w2v(join(vectors_dir, "/glove.42B.300d.w2v"))
 
    def format_word(self, word):
        return '_'.join(word.lower().strip().split())


class GloveModel840B(WordVectorModel):
    """
    Pre-trained Glove vectors trained on 42 billion tokens from 
    Common Crawl (2.2M vocab, cased, 300d vectors).
    
    The file "glove.840B.300d.w2v" is downloadable from:
        
    https://nlp.stanford.edu/projects/glove/
    
    """ 
    def __init__(self, vectors_dir):
        WordVectorModel.__init__(self)
        self.base_model = load_w2v(join(vectors_dir, "/glove.840B.300d.w2v"))
 
    def format_word(self, word):
        return '_'.join(word.strip().split())


# Change this to wherever you store your word vector data files.
VECTORS_DIRECTORY = '/Users/hopkinsm/Projects/thirdparty/word2vec/trunk/'
    
def load_model(name):
    """
    Simple dispatch function for loading in a set of word vectors from
    a string id.
    
    e.g. GOOGLENEWS_MODEL = load_model('googlenews')
    
    """
    print('loading word2vec vectors')
    if name == 'googlenews':
        model = GoogleNewsModel(VECTORS_DIRECTORY)
    elif name == 'glove' or name == 'glove42b':
        model = GloveModel42B(VECTORS_DIRECTORY)     
    elif name == 'glove6b':
        model = GloveModel6B(VECTORS_DIRECTORY)     
    elif name == 'glove840b':
        model = GloveModel840B(VECTORS_DIRECTORY)     
    elif name == 'freebase':
        model = FreebaseModel(VECTORS_DIRECTORY)
    else:
        return ValueError("Unrecognized model name: {}".format(name))
    print('...done!')
    return model    

GOOGLENEWS_MODEL = load_model('googlenews')
