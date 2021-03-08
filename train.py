import time
import logging

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from tqdm import tqdm

from model import Encoder, Decoder
from dataloader import load_cnn_dailymail_deep, \
                       BOS_SYM, EOS_SYM

BATCH_SIZE = 16
VOCAB_SIZE = 50000
EPOCHS = 30
HIDDEN_DIMS = 256
EMBEDDING_DIMS = 128
ARTICLE_TOKENS = 400
TRAIN_TOKENS = 100
TEST_MAX_TOKENS = 120
NALLAPATI_BASELINE = True
SEE_BASELINE = False
RUM_BOTH = False
RUM_DECODER = False
RUM_DECODER = False
BEAM_DECODING = False
CHECKPOINT_DIR = '../drive/MyDrive/NLP_Results/NALLAPATI_BASELINE_CHECKPOINTS'

rng = np.random.default_rng()
logging.basicConfig(filename='../drive/MyDrive/NLP_Results/NALLAPATI_BASELINE' +
                             '_' + str(EPOCHS) + '_EPOCHS_' + str(BATCH_SIZE) +
                             '_BATCH_SIZE_' + str(VOCAB_SIZE) + '_VOCAB_SIZE.log'
                             , level=logging.DEBUG)



def experiment1():
    ds_train, ds_val, ds_test, vocab = load_cnn_dailymail_deep(batch_size=BATCH_SIZE,
                                                               max_vocab=VOCAB_SIZE,
                                                               max_sequence=ARTICLE_TOKENS)
    encoder = Encoder(batch_size=BATCH_SIZE, vocab_len=VOCAB_SIZE)
    decoder = Decoder(batch_size=BATCH_SIZE, vocab_len=VOCAB_SIZE)

    # TODO: Get Adam optimizer working?
    optimizer = tf.keras.optimizers.Adam()

    # Set up checkpoints.
    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                     decoder=decoder)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    logging.info("Checkpoint status:", status)

    # Predict summary for one article.
    def predict(article):
        h_i, hidden = encoder(article, None)
        summary = []
        token = BOS_SYM
        input = tf.expand_dims(tf.convert_to_tensor([vocab.index(token)]), axis=0)
        while token != EOS_SYM and len(summary) < TEST_MAX_TOKENS:
            P_vocab, hidden, attn = decoder(input, hidden[-2:], h_i)

            # Numpy's multinomial gets upset if the probability values
            # summed exceed 1, but this happens sometimes due to type
            # conversion/numerical issues.
            # https://stackoverflow.com/questions/23257587/
            # how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
            # Above was helpful for identifying problem and solving!
            P_vocab = np.asarray(P_vocab[0][0]).astype('float64')
            P_vocab = P_vocab / np.sum(P_vocab)
            predict_index = np.argmax(rng.multinomial(n=1, pvals=P_vocab))
            token = vocab[predict_index]
            input = tf.expand_dims(tf.convert_to_tensor([predict_index]), axis=0)
            summary.append(token)
        return summary

    # Process a single batch.
    @tf.function
    def process_batch(X, y):
        loss = 0
        # Pass articles {w_i,...,w_i+batch_size}
        # into encoder (single-layer bidirectional
        # RNN), to produce encoder hidden state
        # sequences {h_i,...,h_i+batch_size}.
        #
        # h_i output shape is:
        #   (batch_size, max_article_seq_len, #_RNN_unitsx2*).
        #       *x2 because of bidirectionality!
        #
        # Passing None for hidden state, is initialized
        # using random normal distribution inside of
        # forward pass for each sequence.
        h_i, hidden = encoder(X, None)

        # Initialize decoder with encoder hidden
        # state. For now, grabbing hidden/carry
        # states output from backwards RNN.
        decoder_hidden = hidden[-2:]

        # For teacher-forcing procedure, initialize
        # by passing in start token (BOS_SYM) for
        # each target in batch.
        decoder_input = tf.expand_dims([vocab.index(BOS_SYM)]* BATCH_SIZE, 1)

        # At each timestep, decoder (single-layer,
        # unidirectional RNN), gets the target word.
        # Limiting target sequence to 100 tokens.
        for timestep in range(1, TRAIN_TOKENS+1):
            P_vocab, decoder_hidden, attn = decoder(decoder_input, decoder_hidden, h_i)
            loss += NLL_loss(y[:, timestep], P_vocab)

            # Grab next word in target sequence for
            # next pass through decoder.
            decoder_input = tf.expand_dims(y[:, timestep], 1)
        return loss

    # Process a single batch, and update gradients.
    @tf.function
    def train_batch(X, y):
        with tf.GradientTape() as tape:
            loss = process_batch(X, y)
        
        # Get loss over the whole sequence.
        batch_loss = loss / int(y.shape[1])
        vars = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(batch_loss, vars)
        optimizer.apply_gradients(zip(gradients, vars))
        return batch_loss

    # Negative log likelihood loss.
    def NLL_loss(target, P_vocab):
        loss = 0
        P_vocab = tf.squeeze(P_vocab)
        # Yuck. Try to figure out how to use tensor operations
        # for this process, or else this will be likely be slooooow.
        for index in range(target.shape[0]):
            # When tokens at end of sequence are empty/0,
            # do not accumulate loss.
            if [target[index]] != 0:
                loss += -tf.math.log(P_vocab[index][target[index]])
        return loss

    # Training Loop!
    for epoch in range(EPOCHS):
        epoch_loss = 0
        logging.info("Number of training batches:", len(ds_train), "\n")
        for batch_num, batch_data in tqdm(enumerate(ds_train)):
            # X shape: (batch_size, max_article_seq_len)
            # y shape: (batch_size, max_summary_len)
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            batch_loss = train_batch(X,y)
            epoch_loss += loss
            logging.info("Batch_loss: ", batch_loss)
        logging.info("Average epoch loss: ", epoch_loss/batch_num)

        # Save model checkpoints every epoch.
        checkpoint.save(file_prefix='.cpt')

        # Validate.
        logging.info("\n\nValidating!")
        average_val_loss = 0
        val_epoch_loss = 0
        logging.info("Number of val batches:",len(ds_val),"\n")
        for batch_num, batch_data in tqdm(enumerate(ds_val)):
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            loss = process_batch(X,y)

            # Get loss over the whole sequence.
            val_batch_loss = loss / int(y.shape[1])
            val_epoch_loss += val_batch_loss

            # Try to predict.
            if batch_num % 25 == 0:
                article = " ".join([vocab[word] for word in X[0]])
                summary = " ".join([vocab[word] for word in y[0]])
                logging.info("Article: \n\t", article)
                logging.info("Actual summary: \n\t", summary)
                logging.info("Generated summary: \n\t", " ".join(predict(tf.expand_dims(X[0],axis=0))))

        logging.info("Val loss: ", val_epoch_loss/batch_num)


if __name__ == "__main__":
    experiment1()