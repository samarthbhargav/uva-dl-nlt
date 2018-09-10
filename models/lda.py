from gensim.corpora.dictionary import Dictionary

class LdaModel:
    def __init__(self, num_topics):
        self.num_topics = num_topics
    
    def build(self, data):
        for _id, categories, text in data:
            print(text)
    
    def fit(self, data):
        pass
    
    def predict(self, text):
        pass
    