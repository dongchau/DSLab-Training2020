import random
import numpy as np


class DataReader:
    """
    Read data in batch
    """
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self.data = []
        self.labels = []
        self.tokens = []
        self.sentence_lengths = []
        self.current_part = 0
        # read data from file
        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id, sentence_length = int(features[0]), int(features[1]), int(features[2])
            vector = np.array([int(feature) for feature in features[3].split()])
            # print(vector)
            self.data.append(vector)
            self.tokens.append(vector[:sentence_length])
            self.labels.append(label)
            self.sentence_lengths.append(sentence_length)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.sentence_lengths = np.array(self.sentence_lengths)
        self.tokens = np.array(self.tokens)
        self.num_epoch = 0
        self.batch_id = 0

    def next_batch(self, random_seed):
        start = self.batch_id * self.batch_size
        end = start + self.batch_size
        self.batch_id += 1

        # create a batch
        if end + self.batch_size > len(self.data):
            end = len(self.data)
            start = end - self.batch_size
            self.num_epoch += 1
            self.batch_id = 0
            indices = list(range(len(self.data)))
            random.seed(random_seed)
            random.shuffle(indices)
            self.data, self.labels, self.tokens, self.sentence_lengths \
                = self.data[indices], self.labels[indices], self.tokens[indices], self.sentence_lengths[indices]

        return self.data[start:end], self.labels[start:end], self.sentence_lengths[start:end], self.tokens[start:end]
