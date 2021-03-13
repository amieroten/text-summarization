import time
import os

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tqdm import tqdm

from model import Encoder, DecoderBaseline, \
                  DecoderPointerGenerator
from dataloader import load_cnn_dailymail_deep, \
                       load_cnn_dailymail_experiment, \
                       BOS_SYM, EOS_SYM

rng = np.random.default_rng()

def greedy_predict(article, encoder, decoder, vocab):
    '''
    Perform inference using encoder/decoder to generate
    a summary of the article passed in. This uses a
    very simple greedy decoding method, where the .

    :param article: The article to generate a summary for.
                    Assumed to be in vectorized form already.
    :param encoder: Trained encoder model.
    :param decoder: Trained decoder model.
    :param vocab: Vocabulary object.
    :return: A list of English words corresponding to the
             generated summary.
    '''
    h_i, hidden = encoder(article, None)
    summary = []
    token = BOS_SYM
    input = tf.expand_dims(tf.convert_to_tensor([vocab.index(token)]), axis=0)
    while token != EOS_SYM and len(summary) < 121:
        P_vocab, hidden, attn = decoder(input, hidden[-2:], h_i)

        # Numpy's multinomial gets upset if the probability values
        # summed exceed 1, but this happens sometimes due to type
        # conversion/numerical issues.
        # https://stackoverflow.com/questions/23257587/
        # how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
        # Above was helpful for identifying problem and solving!
        P_vocab = np.asarray(P_vocab[0][0]).astype('float64')
        P_vocab = P_vocab / np.sum(P_vocab)

        # This is what we did in HW3. However, this seems more appropriate
        # for a strictly generative model, and it seems as though the
        # true greedy decoding method operates on the final distribution
        # itself and grabs the word with the highest probability from
        # the softmax itself.
        #predict_index = np.argmax(rng.multinomial(n=1, pvals=P_vocab))

        # ...as in below!
        predict_index = np.argmax(P_vocab)
        token = vocab[predict_index]
        input = tf.expand_dims(tf.convert_to_tensor([predict_index]), axis=0)
        summary.append(token)
    return summary

def NLL_loss(target, P_vocab, P_gen=None,
             attention=None, X_words=None,
             y_words=None, timestep=0,
             vocab=None):
    '''
    Compute negative log likelihood loss for a (possibly batch)
    of target words, for a single time point. Only accumulate
    loss when target is an actual word! Since these summaries
    are different shapes, and are padded by '', we do not want
    to accumulate loss in the padding.

    :param target: Target word(s) vectorized.
    :param P_vocab: Probability distribution(s) predicted by model.
    :return: Log loss over time step for current batch.
    '''
    loss = 0
    P_vocab = tf.squeeze(P_vocab)
    # Yuck. Try to figure out how to use tensor operations
    # for this process, or else this will be slooooow.
    for index in range(target.shape[0]):
        # When tokens at end of sequence are empty/0,
        # do not accumulate loss.
        if [target[index]] != 0:
            loss += -tf.math.log(P_vocab[index][target[index]])
    return loss

def greedy_predict_point_gen(article, encoder, decoder, vocab,
                             article_full):
    '''
    Perform inference using encoder/decoder to generate
    a summary of the article passed in. This uses a
    very simple greedy decoding method, where the .

    :param article: The article to generate a summary for.
                    Assumed to be in vectorized form already.
    :param encoder: Trained encoder model.
    :param decoder: Trained decoder model.
    :param vocab: Vocabulary object.
    :return: A list of English words corresponding to the
             generated summary.
    '''

    vocab_size = len(vocab)
    unknown_index = vocab.index('[UNK]')

    # Find words which are present in the article, but not present
    # in the vocabulary (aka "extended vocabulary").
    article_words = [word.decode("UTF-8") for word in article_full]
    article_only_words = list(set(article_words).difference(set(vocab)))
    full_vocab = vocab + article_only_words

    h_i, hidden = encoder(article, None)
    summary = []
    token = BOS_SYM

    input = tf.expand_dims(tf.convert_to_tensor([vocab.index(token)]), axis=0)
    while token != EOS_SYM and len(summary) < 121:
        P_vocab, hidden, attn, p_gen = decoder(input, hidden, h_i)
        P_vocab = np.array(tf.squeeze(P_vocab))
        attn = tf.squeeze(attn)
        p_gen = tf.squeeze(p_gen)

        # Pad P_vocab with zeros by the length of
        # the extra words in extended vocabulary.
        P_vocab = np.pad(P_vocab, (0,len(article_only_words)))

        # Get second term.
        attn_sum = np.zeros((len(full_vocab)))
        for i in range(len(article_words)):
            # Find vocabulary index for current word,
            # and add attention value.
            attn_sum[full_vocab.index(article_words[i])] += attn[i]

        P_extended_vocab_probs = np.array(p_gen * P_vocab + (1 - p_gen) * attn_sum).astype('float64')

        # Normalize.
        P_extended_vocab_probs = P_extended_vocab_probs / np.sum(P_extended_vocab_probs)

        # A couple of options for selecting the word.
        predict_index = np.argmax(rng.multinomial(n=1, pvals=P_extended_vocab_probs))
        #predict_index = np.argmax(P_vocab)

        token = full_vocab[predict_index]

        # If word selected is OOV, change to unknown.
        if predict_index >= vocab_size:
            predict_index = unknown_index

        input = tf.expand_dims(tf.convert_to_tensor([predict_index]), axis=0)
        summary.append(token)

    if summary[-1] == EOS_SYM:
        return summary[:-1]
    return summary

def NLL_loss_point_gen(target, P_vocab, P_gen=None,
             attention=None, X_words=None,
             y_words=None, timestep=0,
             vocab=None):
    '''
    Compute negative log likelihood loss for a (possibly batch)
    of target words, for a single time point. Only accumulate
    loss when target is an actual word! Since these summaries
    are different shapes, and are padded by '', we do not want
    to accumulate loss in the padding.

    :param target: Target word(s) vectorized.
    :param P_vocab: Probability distribution(s) predicted by model.
    :return: Log loss over time step for current batch.
    '''
    loss = 0
    P_vocab = tf.squeeze(P_vocab)
    # Yuck. Try to figure out how to use tensor operations
    # for this process, or else this will be slooooow.
    for index in range(target.shape[0]):
        # When tokens at end of sequence are empty/0,
        # do not accumulate loss.
        if target[index] != 0:
            # This time, target word comes directly from the
            # source document, not the vectorized targets.
            # thus, word could be OOV.
            target_word = y_words[index][timestep + 1].decode('UTF-8')

            # If word is not in vocabulary, P_vocab_w is 0.
            if vocab[target[index]] == '[UNK]':
                P_vocab_w = 0
                # Select a pad value that we know won't
                # match the target for use in computing
                # sum over attention.
                pad_value = '.[UNK].'
            else:
                P_vocab_w = P_vocab[index][target[index]]
                pad_value = '[UNK]'

            # Sum over probabilities in attention distribution
            # where the corresponding word in article is the
            # target word. If target word is not in article,
            # this will be zero.
            X_words_arr = np.pad(np.array([word.decode("UTF-8") for word in X_words[index]]),
                                 (0,400-len(X_words[index])), constant_values=(0,pad_value))
            attn_sum = sum(attention[index][X_words_arr == target_word])
            P_w = P_gen[index]*P_vocab_w + (1-P_gen[index])*attn_sum + 0.00001 # To not take log of zero.
                                                                              # It is bizarre to have to
                                                                              # do this in my opinion, but
                                                                              # it really does seem that
                                                                              # if the target word is neither
                                                                              # in the vocabulary or the
                                                                              # source article, then the
                                                                              # P_w will be zero.
            loss += -tf.math.log(P_w)
    return loss

def vector_to_string(vector, vocab):
    '''
    Helper function to quickly examine the text contained
    in a vectorized word sequence.

    :param vector: Vector to "translate".
    :param vocab: Trained vocabulary.
    :return: "Translated" vector.
    '''
    string = []
    for index in vector:
        string.append(vocab[index])
    return " ".join(string)

def prep_word_X_y(X,y, max_article_seq=400, max_summary_seq=120):
    X_words = tf.strings.split(X).numpy()
    X_words_reduced = []
    for article in X_words:
        X_words_reduced.append(article[:max_article_seq])

    y_words = tf.strings.split(y).numpy()
    y_words_reduced = []
    for summary in y_words:
        y_words_reduced.append(summary[:max_summary_seq])

    return X_words_reduced, y_words_reduced


def validation_inference(encoder, decoder, vocab, dataset, 
                         batch_size, epoch, avg_epoch_loss, 
                         save_dir, point_gen=True):
    print('\n Performing inference on validation set!')
    index = 1
    filepath = os.path.join(save_dir, "validation_output_" + str(epoch) + ".txt")
    with open(filepath, "w") as f:
        f.write("Average loss for this epoch (train set):" + str(avg_epoch_loss))
        for batch_data in dataset:
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            X_words, y_words = prep_word_X_y(batch_data[2], batch_data[3])

            for i in range(batch_size):
                article = " ".join([vocab[word] for word in X[i]])
                summary = " ".join([vocab[word] for word in y[0]])
                article_full = X_words[i]
                summary_full = y_words[i]
                if point_gen:
                    prediction = " ".join(greedy_predict_point_gen(tf.expand_dims(X[i], axis=0),
                                                                   encoder, decoder, vocab,
                                                                   article_full))
                else:
                    prediction = " ".join(greedy_predict(tf.expand_dims(X[0],axis=0),
                                                         encoder, decoder, vocab))

                f.write("\nExample #" + str(index) + ":")
                f.write("\nIn vocab article:\n")
                f.write(article)
                print(article)
                f.write("\nIn vocab summary:\n")
                f.write(summary)
                print(summary)
                f.write("\nComplete article:\n")
                f.write(" ".join([word.decode('UTF-8') for word in article_full]))
                print(" ".join([word.decode('UTF-8') for word in article_full]))
                f.write("\nComplete summary:\n")
                f.write(" ".join([word.decode('UTF-8') for word in summary_full]))
                print(" ".join([word.decode('UTF-8') for word in summary_full]))
                f.write("\nPredicted summary:\n")
                f.write(prediction)
                print(prediction)
                index += 1

def experiment_pointer_gen():
    '''
    Basic code to train text summarization model.
    '''
    batch_size = 16
    vocab_size = 50000
    ds_train, ds_val, ds_test, vocab = load_cnn_dailymail_deep(batch_size=batch_size,
                                                          max_vocab=vocab_size,
                                                          max_sequence=400)
    encoder = Encoder(batch_size=batch_size, vocab_len=vocab_size)
    decoder = DecoderPointerGenerator(batch_size=batch_size, vocab_len=vocab_size)
    epochs = 30

    # As in See et al.
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.15, initial_accumulator_value=0.1)

    # Set up checkpointing. Assisted by https://www.tensorflow.org/guide/checkpoint
    checkpoint_directory = "/content/drive/MyDrive/NLP_Results/train_checkpoints_pointer_generator/"
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    cpt_manager = tf.train.CheckpointManager(checkpoint,
                                             checkpoint_directory,
                                             max_to_keep=None)
    checkpoint.restore(cpt_manager.latest_checkpoint)

    if cpt_manager.latest_checkpoint:
        print("Restored from", cpt_manager.latest_checkpoint)
    else:
        print("Starting training from ground zero!")

    print("Beginning training!")

    for epoch in range(int(checkpoint.step), epochs):
        print("Epoch", epoch+1)
        epoch_loss = 0
        for batch_num, batch_data in tqdm(enumerate(ds_train)):
            loss = 0
            # X shape: (batch_size, max_article_seq_len)
            # y shape: (batch_size, max_summary_len)
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            X_words, y_words = prep_word_X_y(batch_data[2], batch_data[3])

            # Training, so must capture gradients.
            with tf.GradientTape() as tape:
                h_i, encoder_hidden = encoder(X, None)

                decoder_input = y[:,0:100]
                target = y[:,1:101]

                # Initialize decoder with encoder hidden state.
                decoder_hidden = encoder_hidden
                for timestep in range(0, 100):
                    input_i = tf.expand_dims(decoder_input[:, timestep], 1)
                    P_vocab, decoder_hidden, attn, P_gen = decoder(input_i, decoder_hidden, h_i)
                    loss += NLL_loss_point_gen(target[:, timestep], P_vocab, tf.squeeze(P_gen),
                                     attn, X_words, y_words, timestep, vocab)

                # Get loss over the whole sequence.
                batch_loss = loss / int(y.shape[1])
                vars = encoder.trainable_variables + decoder.trainable_variables
                gradients = tape.gradient(batch_loss, vars)
                optimizer.apply_gradients(zip(gradients, vars))

            epoch_loss += batch_loss
            print("batch_loss: ", batch_loss)

            # Monitor intermittent predictions.
            if batch_num % 50 == 0:
                article = " ".join([vocab[word] for word in X[0]])
                summary = " ".join([vocab[word] for word in y[0]])
                article_full = X_words[0]
                print("Article: \n\t", article)
                print("Summary: \n\t", summary)
                print("Generated summary: \n\t", " ".join(greedy_predict_point_gen(tf.expand_dims(X[0],axis=0),
                                                                                   encoder, decoder, vocab,
                                                                                   article_full)))
        average_batch_loss_per_epoch = epoch_loss/(batch_num+1)
        print("average batch loss for epoch: ", average_batch_loss_per_epoch)

        # Save checkpoint every epoch.
        checkpoint.step.assign_add(1)
        cpt_manager.save()

        # Perform inference on val set and output for inspection.
        validation_inference(encoder, decoder, vocab, ds_val,
                             batch_size, epoch+1, average_batch_loss_per_epoch,
                             point_gen=True)


def experiment_baseline():
    '''
    Basic code to train text summarization model.
    '''
    batch_size = 16
    vocab_size = 50000
    ds_train, ds_val, ds_test, vocab = load_cnn_dailymail_deep(batch_size=batch_size,
                                                          max_vocab=vocab_size,
                                                          max_sequence=400)
    encoder = Encoder(batch_size=batch_size, vocab_len=vocab_size)
    decoder = DecoderBaseline(batch_size=batch_size, vocab_len=vocab_size)
    epochs = 1

    # As in See et al.
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.15, initial_accumulator_value=0.1)

    # Set up checkpointing.
    checkpoint_directory = "drive/MyDrive/train_checkpoints_baseline"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    print("Beginning training!")
    total_train_batches = len(ds_train)

    for epoch in range(epochs):
        print("Epoch", epoch+1)
        epoch_loss = 0
        for batch_num, batch_data in tqdm(enumerate(ds_train)):
            loss = 0
            # X shape: (batch_size, max_article_seq_len)
            # y shape: (batch_size, max_summary_len)
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            # Training, so must capture gradients.
            with tf.GradientTape() as tape:
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
                h_i, encoder_hidden = encoder(X, None)

                # At each timestep, decoder (single-layer,
                # unidirectional RNN), gets the target word.
                # Limiting target sequence to 100 tokens.
                # Input is always trying to predict next
                # token.
                decoder_input = y[:,0:100]
                target = y[:,1:101]

                # Initialize decoder with encoder hidden state.
                decoder_hidden = encoder_hidden
                for timestep in range(0, 100):
                    input_i = tf.expand_dims(decoder_input[:, timestep], 1)
                    P_vocab, decoder_hidden, attn = decoder(input_i, decoder_hidden, h_i)
                    loss += NLL_loss(target[:, timestep], P_vocab)


                # Get loss over the whole sequence.
                batch_loss = loss / int(y.shape[1])
                vars = encoder.trainable_variables + decoder.trainable_variables
                gradients = tape.gradient(batch_loss, vars)
                optimizer.apply_gradients(zip(gradients, vars))

            epoch_loss += loss
            print("batch_loss: ", batch_loss)

            # Try to predict.
            if batch_num % 50 == 0:
                article = " ".join([vocab[word] for word in X[0]])
                summary = " ".join([vocab[word] for word in y[0]])
                print("Article: \n\t", article)
                print("Generated summary: \n\t", " ".join(greedy_predict(tf.expand_dims(X[0],axis=0),
                                                                         encoder, decoder, vocab)))

        print("average batch loss over epoch: ", epoch_loss/total_train_batches)

        # Save checkpoint every epoch.
        checkpoint.save(file_prefix=checkpoint_prefix)

        # Perform inference on val set and output for inspection.
        validation_inference(encoder, decoder, vocab, ds_val, batch_size,
                             epoch+1, epoch_loss/total_train_batches, point_gen=True)



if __name__ == "__main__":
    #experiment_baseline()
    experiment_pointer_gen()