import time

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
import tensorflow_probability as tfp

from model import Encoder, Decoder
from dataloader import load_cnn_dailymail_experiment, \
                       BOS_SYM, EOS_SYM

rng = np.random.default_rng()

def experiment1():
    batch_size = 8
    vocab_size = 5000
    ds_train, _, _, vocab = load_cnn_dailymail_experiment(batch_size=batch_size,
                                                          max_vocab=vocab_size,
                                                          max_sequence=400)
    encoder = Encoder(batch_size=batch_size, vocab_len=vocab_size)
    decoder = Decoder(batch_size=batch_size, vocab_len=vocab_size)
    epochs = 1

    optimizer = tf.keras.optimizers.Adam()

    def predict(article):
        h_i, hidden = encoder(article, None)
        summary = []
        token = "bos"
        input = tf.expand_dims(tf.convert_to_tensor([vocab.index(token)]), axis=0)
        while token != "eos" and len(summary) < 151:
            P_vocab, hidden = decoder(input, hidden[-2:], h_i)
            predict_index = np.argmax(rng.multinomial(n=1, pvals=P_vocab[0][0]))
            token = vocab[predict_index]
            input = tf.expand_dims(tf.convert_to_tensor([predict_index]), axis=0)
            summary.append(token)
        return summary

    def NLL_loss(target, P_vocab):
        loss = 0
        for index, word in enumerate(target):
            loss += -tf.math.log(P_vocab[index][int(word)])
        return (1/len(target))*loss

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_num, batch_data in enumerate(ds_train):
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            with tf.GradientTape() as tape:
                h_i, hidden = encoder(X, None)
                P_vocab, hidden = decoder(y, hidden[-2:], h_i)
                loss = sum([NLL_loss(y[index], P_vocab[index])
                               for index in range(len(y))])
                vars = encoder.trainable_variables + decoder.trainable_variables
                gradients = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(gradients, vars))
            epoch_loss += loss
            print("batch_loss: ", loss)
        predict(X)
        print("loss: ", epoch_loss/batch_num)




if __name__ == "__main__":
    experiment1()