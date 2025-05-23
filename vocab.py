import nltk
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

        # Додаємо синоніми для сумісності з CocoDataset
        self.word2idx = self.stoi
        self.idx2word = self.itos

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        # Оновлюємо word2idx та idx2word після побудови
        self.word2idx = self.stoi
        self.idx2word = self.itos

    def numericalize(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())
        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in tokenized_text
        ]
