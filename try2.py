# Imports

import tensorflow as tf
import numpy as np
import collections

from tensorflow.contrib import rnn

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIR = '/home/simi/Desktop/nlp_proj1_data/data/'
TRAIN_FILE = DATA_DIR + 'sentences.train'
TEST_FILE = DATA_DIR + 'sentences.eval'

BOS = '<bos>'
UNK = '<unk>'
EOS = '<eos>'
PAD = '<pad>'

SENTENCE_LENGTH = 30


def _read_words(filename):
    data = []
    with tf.gfile.GFile(filename, 'r') as f:
        for line in f.readlines():
            # Beginning of line
            line_arr = [BOS]
            # split line in white spaces
            line_arr.extend(line.split())
            # pad line
            line_arr.extend([PAD] * (SENTENCE_LENGTH - len(line_arr) - 1))
            # cut line if to long
            line_arr = line_arr[:SENTENCE_LENGTH - 1]
            # append eos
            line_arr.append(EOS)
            # add to all lines
            data.append(line_arr)
    return data
    # return ("<bos> " + f.read().replace('\n', '<eos> <bos>')).split()


flatten = lambda l: [item for sublist in l for item in sublist]


def _build_vocab(filename, vocab_length):
    data = _read_words(filename)
    data_ = flatten(data)

    counter = collections.Counter(data_)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = list(words[:vocab_length])

    words[-1] = UNK
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [[word_to_id['<unk>'] if word not in word_to_id else word_to_id[word] for word in sentence]
            for sentence in data]


def main(args):
    word_to_id = _build_vocab(TRAIN_FILE, 20000)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    test_data = _file_to_word_ids(TEST_FILE, word_to_id)
    train_data = _file_to_word_ids(TRAIN_FILE, word_to_id)

    data = train_data

    num_words = len(word_to_id.keys())

    BATCH_SIZE = 64
    SIZE_EMBEDDING = 100
    LSTM_CELLS = 512

    # model params
    MAX_GRAD_NORM = 5
    EPOCHS = 10

    num_batches = len(data) // BATCH_SIZE
    num_timesteps = 1

    W_embedding = tf.Variable(tf.random_normal([num_words, SIZE_EMBEDDING]))
    W_softmax = tf.get_variable('W_softmax', shape=(LSTM_CELLS, num_words),
                                initializer=tf.contrib.layers.xavier_initializer())

    b_softmax = tf.get_variable('b_softmax', shape=(num_words),
                                initializer=tf.contrib.layers.xavier_initializer())

    optimizer = tf.train.AdamOptimizer()

    def RNN(x, W_embedding, W_softmax, b_softmax, optimizer):
        cell = rnn.BasicLSTMCell(LSTM_CELLS, forget_bias=1.0)

        state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        outputs = []

        inputs = tf.one_hot(x, num_words)
        inputs = tf.tensordot(inputs, W_embedding, axes=((2), (0)))

        for i in range(SENTENCE_LENGTH):
            output, state = cell(inputs[:, i], state)
            outputs.append(output)
        loss = tf.zeros((BATCH_SIZE))
        for i in range(SENTENCE_LENGTH - 1):
            output_ = outputs[i]
            correct_pred = x[:, i + 1]
            output_ = tf.nn.xw_plus_b(output_, W_softmax, b_softmax)
            output_soft = tf.nn.softmax(output_)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_pred, logits=output_)

            cost = tf.reduce_sum(loss)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), MAX_GRAD_NORM)

            optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())

        return cost

    x = tf.placeholder(dtype=np.int64, shape=(BATCH_SIZE, SENTENCE_LENGTH), name='x')  # data_tensor[:BATCH_SIZE]

    cost = RNN(x, W_embedding, W_softmax, b_softmax, optimizer)
    inits = tf.global_variables_initializer()
    with tf.Session() as sess:
        print('Initialize')
        sess.run([inits])
        for epoch in range(EPOCHS):
            print('Epoch: {}'.format(epoch))
            for i in range(num_batches):
                print(sess.run([cost], feed_dict={
                    x: data[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
                }))


if __name__ == '__main__':
    tf.app.run()
