class NaiveTokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        
        self.encoding_table = {c: i for i, c in enumerate(self.vocab)}
        self.decoding_table = {i: c for i, c in enumerate(self.vocab)}
        
    def encode(self, x):
        return [self.encoding_table[c] for c in x]
    
    def decode(self, s):
        return ''.join([self.decoding_table[i] for i in s])
        