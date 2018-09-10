import os
import codecs
import logging as log
from collections import OrderedDict
from torch.utils.data import Dataset
import json

from collections import Counter
import spacy

class ReutersDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_location, split, remove_stopwords, min_freq, lowercase):
        assert split in {"training", "test"}, "Invalid split"
        self.split = split
        self.root_location = root_location
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.keys = []
        self.cat = OrderedDict()        
        
        self.counterAll = Counter()
        self.least_frequent = []
        
        self.nlp = spacy.load("en")
        
        self.stop_words = []
        with codecs.open(os.path.join(root_location, "stopwords"), "r", "utf-8") as reader:
            for line in reader:
                line = line.strip()
                self.stop_words.append(line)
                
        with codecs.open(os.path.join(root_location, "cats.txt"), "r", "utf-8") as reader:
            for line in reader:
                line = line.strip().split()
                _id = line[0]
                categories = line[1:]
                if _id.startswith(self.split):
                    self.cat[_id] = categories
                    self.keys.append(_id)
                    
        #updates counterAll  
        print('Counting words')
        for k in self.keys:
            self.count_words(k)
        
        
        with open('vocab_counter.json', 'w') as file:
            json.dump(self.counterAll, file)
            
        for c in self.counterAll:
            if self.counterAll[c] < self.min_freq:
                self.least_frequent.append(c)
                
        log.info("Loaded {} dataset: {} entries".format(self.split, len(self.cat)))

    def get_tokenized_text_from_id(self, id_):
        
        with codecs.open(os.path.join(self.root_location, id_), "r", "utf-8") as reader:
            doc = reader.read()
            doc = self.nlp(doc, disable=['parser', 'tagger', 'ner'])
            
        return doc
    
    def count_words(self, id_):
        
        doc = self.get_tokenized_text_from_id(id_)
        
        if self.lowercase:
            doc = [token.text.lower() for token in doc if token.text.lower()]   
            
        else:
            doc = [token.text for token in doc]

        text_counter = Counter(doc)
        self.counterAll += text_counter  
        
        
    def __len__(self):
        return len(self.cat)
    
    def __getitem__(self, idx):
        id_ = self.keys[idx]
        label = self.cat[id_]
        
        doc = self.get_tokenized_text_from_id( id_)
        
        if self.lowercase:
            if self.remove_stopwords:
                #convert to lowercase, remove stopwords and spaces
                doc = [token.text.lower() for token in doc if token.text.lower() not in self.stop_words and token.text.strip() != '']   
            else:
                #just convert to lowercase and remove spaces
                doc = [token.text.lower() for token in doc if token.text.strip() != '']

        else:
            if self.remove_stopwords:
                #remove stopwords and remove spaces
                doc = [token.text for token in doc if token.text not in self.stop_words and token.text.strip() != '']  
            else:
                #just remove spaces
                doc = [token.text.lower() for token in doc if token.text.strip() != '']

        doc.insert(0, '<SOT>')
        doc.append('<EOT>')
         
        for tl in range(len(doc)):
            if doc[tl] in self.least_frequent:
                doc[tl] = '<UNK>'
    
        return id_, label, doc