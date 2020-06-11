import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


MAX_DOC_LENGTH = 500
NUM_CLASSES = 20


class RNN:
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 lstm_size,
                 batch_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.data = tf.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, ])
        self.sentence_lengths = tf.placeholder(tf.int32, shape=[batch_size, ])
        self.final_tokens = tf.placeholder(tf.int32, shape=[batch_size, ])

    def embedding_layer(self, indices):
        pretrained_vectors = [np.zeros(self.embedding_size)]
        np.random.seed(42)
        for _ in range(self.vocab_size + 1):
            pretrained_vectors.append(np.random.normal(loc=0., scale=1., size=self.embedding_size))

        pretrained_vectors = np.array(pretrained_vectors)

        embedding_matrix = tf.get_variable(
            name='embedding',
            shape=(self.vocab_size + 2, self.embedding_size),
            initializer=tf.constant_initializer(pretrained_vectors)
        )
        return tf.nn.embedding_layer(embedding_matrix, indices)

    def LSTM_layer(self, embeddings):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        zero_state = tf.zeros(shape=(self.batch_size, self.lstm_size))
        initial_state = tf.contrib.rnn.LSTMStateTuple(zero_state, zero_state)

        lstm_inputs = tf.unstack(tf.transpose(embeddings, perm=[1, 0, 2]))
        lstm_outputs, lstm_state = tf.nn.static_rnn(
            cell=lstm_cell,
            inputs=lstm_inputs,
            initial_state=initial_state,
            sequence_length=self.sentence_lengths
        )  # a length-500 list of [num_docs, lstm_size]

        lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, perm=[1, 0, 2]))
        lstm_outputs = tf.concat(
            lstm_outputs,
            axis=0
        )  # [num_docs * MAX_SENT_LENGTH, lstm_size]

        # self.mask: [num_docs * MAX_SENT_LENGTH, ]
        mask = tf.sequence_mask(
            lengths=self.sentence_lengths,
            maxlen=MAX_DOC_LENGTH,
            dtype=tf.float32
        )  # [num_docs, MAX_SENT_LENGTH]
        mask = tf.concat(tf.unstack(mask, axis=0), axis=0)
        mask = tf.expand_dims(mask, -1)
        lstm_outputs = mask * lstm_outputs
        lstm_outputs_split = tf.split(lstm_outputs, num_or_size_splits=self.batch_size)
        lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis=1)  # [num_docs, lstm_size]
        lstm_outputs_average = lstm_outputs_sum / tf.expand_dims(
            tf.cast(self.sentence_lengths, tf.float32),  # expand_dims only work with tensor of float type
            -1
        )  # [num_docs, lstm_size]
        return lstm_outputs_average

    def build_graph(self):
        embeddings = self.embedding_layer(self.data)
        lstm_outputs = self.LSTM_layer(embeddings)

        weights = tf.get_variable(
            name='final_layer_weights',
            shape=(self.lstm_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=42)
        )
        biases = tf.get_variable(
            name='final_layer_bias',
            shape=(self.lstm_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=42)
        )
        logits = tf.matmul(lstm_outputs, weights) + biases

        labels_one_hot = tf.one_hot(
            indices=self.labels,
            depth=NUM_CLASSES,
            dtype=tf.float32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits
        )
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axits=1)
        predicted_labels = tf.squeeze(predicted_labels)
        return predicted_labels, loss

    @staticmethod
    def trainer(loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op


def train_and_evaluate_RNN():
    with open('../datasets/w2v/vocab_w2v.txt') as f:
        vocab_size = len(f.readlines())

    tf.set_random_seed(42)
    rnn = RNN(
        vocab_size=vocab_size,
        embedding_size=300,
        lstm_size=50,
        batch_size=32
    )
    predicted_labels, loss = rnn.build_graph()
    train = rnn.trainer(loss=loss, learning_rate=0.01)

    with tf.Session() as sess:
        train_data_reader = Da
