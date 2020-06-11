import random
import numpy as np


class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self.batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.readlines()

        self.data = []
        self.labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index = int(token.split(':')[0])
                value = float(token.split(':')[1])
                vector[index] = value
            self.data.append(vector)
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.num_epoch = 0
        self.batch_id = 0

    def next_batch(self):
        start = self.batch_id * self.batch_size
        end = start + self.batch_size
        self.batch_id += 1

        if end + self.batch_size > len(self.data):
            end = len(self.data)
            self.num_epoch += 1
            self.batch_id = 0
            indices = list(range(len(self.data)))
            random.seed(2018)
            random.shuffle(indices)
            self.data, self.labels = self.data[indices], self.labels[indices]

        return self.data[start:end], self.labels[start:end]