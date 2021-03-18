'''
Module to perform inference using trained model,
e.g. create automatic summaries from articles!
'''

import tensorflow as tf
from glob import glob
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import pickle as pkl

from model import Encoder, DecoderPointerGenerator
from dataloader import load_cnn_dailymail_deep, BOS_SYM, EOS_SYM
from train import greedy_inference, NLL_loss_point_gen, prep_word_X_y, \
                  greedy_predict_point_gen

# Thanks to https://gist.github.com/thriveth/8560036
# for colorblind friendly color cycle!
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

BATCH_SIZE = 16
VOCAB_SIZE = 50000
rng = np.random.default_rng()

def get_val_train_loss(ds_val, vocab, num_epochs=27):
    all_val_loss = []
    for epoch in range(1, num_epochs+1):
        # Load model checkpoint.
        encoder, decoder = load_specific_model(epoch)
        epoch_loss = 0
        for batch_num, batch_data in tqdm(enumerate(ds_val)):
            loss = 0
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            X_words, y_words = prep_word_X_y(batch_data[2], batch_data[3])

            h_i, encoder_hidden = encoder(X, None)

            decoder_input = y[:, 0:100]
            target = y[:, 1:101]

            # Initialize decoder with encoder hidden state.
            decoder_hidden = encoder_hidden
            for timestep in range(0, 100):
                input_i = tf.expand_dims(decoder_input[:, timestep], 1)
                P_vocab, decoder_hidden, attn, P_gen = decoder(input_i, decoder_hidden, h_i)
                loss += NLL_loss_point_gen(target[:, timestep], P_vocab, tf.squeeze(P_gen),
                                           attn, X_words, y_words, timestep, vocab)

            # Get loss over the whole sequence.
            batch_loss = loss / int(y.shape[1])
            epoch_loss += batch_loss
        print("Average batch loss:", epoch_loss / (batch_num + 1))
        all_val_loss.append(epoch_loss / (batch_num + 1))

    with open("results/val_loss_by_epoch_1_27.pkl", 'wb') as f:
        pkl.dump(all_val_loss, f)

def decode(decoder, hidden, h_i, article_only_words, full_vocab, article_words, input):
    P_vocab, hidden, attn, p_gen = decoder(input, hidden, h_i)
    P_vocab = np.array(tf.squeeze(P_vocab))
    attn = tf.squeeze(attn)
    p_gen = tf.squeeze(p_gen)

    # Pad P_vocab with zeros by the length of
    # the extra words in extended vocabulary.
    P_vocab = np.pad(P_vocab, (0, len(article_only_words)))

    # Get second term.
    attn_sum = np.zeros((len(full_vocab)))
    for i in range(len(article_words)):
        # Find vocabulary index for current word,
        # and add attention value.
        attn_sum[full_vocab.index(article_words[i])] += attn[i]

    P_extended_vocab_probs = np.array(p_gen * P_vocab + (1 - p_gen) * attn_sum).astype('float64')

    # Normalize.
    P_extended_vocab_probs = P_extended_vocab_probs / np.sum(P_extended_vocab_probs)

    return P_extended_vocab_probs, hidden

# Referenced Graves 2012,
# https://guillaumegenthial.github.io/sequence-to-sequence.html,
# and https://towardsdatascience.com/an-intuitive-explanation-of-beam-search-9b1d744e7a0f
def beam_search(article, encoder, decoder, vocab, article_full, K=2):

    vocab_size = len(vocab)
    unknown_index = vocab.index('[UNK]')
    final_seqs = []

    # Find words which are present in the article, but not present
    # in the vocabulary (aka "extended vocabulary").
    article_words = [word.decode("UTF-8") for word in article_full]
    article_only_words = list(set(article_words).difference(set(vocab)))
    full_vocab = vocab + article_only_words

    h_i, hidden = encoder(article, None)

    # Initialize!
    rem_in_beam = K
    H = [BOS_SYM]
    probs = {}
    probs[(BOS_SYM)] = {}
    probs[(BOS_SYM)]["probability"] = np.log(1)
    probs[(BOS_SYM)]["hidden state"] = hidden
    probs[(BOS_SYM)]["final token index"] = vocab.index(BOS_SYM)

    # For each timestep t.
    t = 1
    while t < 121 and rem_in_beam > 0:

        # For new possibilities!
        H_possibilities = set()

        # For each of the best sequences, find the K best extensions.
        for seq in H:
            hidden = probs[seq]["hidden state"]
            probability = probs[seq]["probability"]
            input = tf.expand_dims(tf.convert_to_tensor([probs[seq]["final token index"]]), axis=0)
            P_extended_vocab_probs, hidden = decode(decoder, hidden, h_i,
                                            article_only_words, full_vocab,
                                            article_words, input)
            if "top":
                top_k = np.argpartition(P_extended_vocab_probs, -K)[-K:]
                top_k_probs = P_extended_vocab_probs[top_k]
                top_k_tokens = np.array(full_vocab)[top_k]
            else:
                top_k = []
                for k in range(K):
                    new_index = np.argmax(rng.multinomial(n=1, pvals=P_extended_vocab_probs))
                    while new_index in top_k:
                        new_index = np.argmax(rng.multinomial(n=1, pvals=P_extended_vocab_probs))
                    top_k.append(new_index)
                top_k_probs = P_extended_vocab_probs[top_k]
                top_k_tokens = np.array(full_vocab)[top_k]

            for k in range(K):
                new_seq = " ".join([seq, top_k_tokens[k]])
                probs[new_seq] = {}
                probs[new_seq]["probability"] = np.log(top_k_probs[k]) + probability
                probs[new_seq]["hidden state"] = hidden
                probs[new_seq]["final token index"] = top_k[k] if top_k[k] < vocab_size \
                                                                                else unknown_index
                H_possibilities.add(new_seq)
            H_poss_probs = [(possibility, probs[possibility]["probability"]) for possibility in H_possibilities]

        # Nab k best new sequences!
        H = [seq for seq, prob in sorted(H_poss_probs,key=lambda x: x[1])[-rem_in_beam:]]

        # Check if we've hit EOS.
        for seq in H:
            if probs[seq]["final token index"] == vocab.index(EOS_SYM):
                final_seqs.append((seq, probs[seq]["probability"]))
                H.remove(seq)
                rem_in_beam -= 1
        t += 1

    for seq in H:
        final_seqs.append((seq, probs[seq]["probability"]))

    best = sorted(final_seqs,key=lambda x: x[1])[-1][0]
    best = best.replace(BOS_SYM, "").replace(EOS_SYM, "")
    return best


def inference(encoder, decoder, vocab, dataset,
              epoch, avg_epoch_loss, save_dir,
              beam=True, data_type='validation'):
    print(f'\n Performing inference on {data_type} set!')
    index = 1
    filepath = os.path.join(save_dir, f"{data_type}_output_" + str(epoch) + ".txt")
    with open(filepath, "w") as f:
        if data_type == 'validation':
            f.write("Average loss for this epoch (train set):" + str(avg_epoch_loss))
        for batch_data in dataset:
            X = tf.squeeze(batch_data[0], axis=1)
            y = tf.squeeze(batch_data[1], axis=1)
            X_words, y_words = prep_word_X_y(batch_data[2], batch_data[3])

            for i in range(len(y)):
                article = " ".join([vocab[word] for word in X[i]])
                summary = " ".join([vocab[word] for word in y[i]])
                article_full = X_words[i]
                summary_full = y_words[i]
                if beam:
                    prediction = beam_search(tf.expand_dims(X[i], axis=0),
                                                      encoder, decoder, vocab,
                                                      article_full, K=3)
                else:
                    prediction = " ".join(greedy_predict_point_gen(tf.expand_dims(X[i], axis=0),
                                                                   encoder, decoder, vocab,
                                                                   article_full))

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



def load_best_model():
    encoder = Encoder(batch_size=BATCH_SIZE, vocab_len=VOCAB_SIZE)
    decoder = DecoderPointerGenerator(batch_size=BATCH_SIZE, vocab_len=VOCAB_SIZE)

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.15, initial_accumulator_value=0.1)

    # Restore model from checkpoint.
    checkpoint_directory = "train_checkpoints_pointer_generator/"
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    cpt_manager = tf.train.CheckpointManager(checkpoint,
                                             checkpoint_directory,
                                             max_to_keep=None)
    checkpoint.restore(cpt_manager.latest_checkpoint)
    last_epoch = cpt_manager.latest_checkpoint
    print("Model restored from", last_epoch)
    return encoder, decoder, int(checkpoint.step)


def load_specific_model(epoch):
    encoder = Encoder(batch_size=BATCH_SIZE, vocab_len=VOCAB_SIZE)
    decoder = DecoderPointerGenerator(batch_size=BATCH_SIZE, vocab_len=VOCAB_SIZE)

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.15, initial_accumulator_value=0.1)

    # Restore model from checkpoint.
    checkpoint_directory = "train_checkpoints_pointer_generator/"
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    cpt_manager = tf.train.CheckpointManager(checkpoint,
                                             checkpoint_directory,
                                             max_to_keep=None)
    checkpoint.restore(os.path.join(checkpoint_directory, f"ckpt-{epoch}"))
    print("Model restored from", epoch)
    return encoder, decoder


def load_test():
    _, ds_val, ds_test, vocab = load_cnn_dailymail_deep(batch_size=BATCH_SIZE,
                                                   max_vocab=VOCAB_SIZE,
                                                   max_sequence=400)
    return ds_val, ds_test, vocab


def get_val_results(val_file):
    val_results = {}
    with open(val_file, "r") as f:
        line = f.readline()
        val_results["epoch train loss"] = line.split("[")[-1].split("]")[0]
        val_results["samples"] = {}
        while line != "":
            if line == "In vocab article:":
                current_article = f.readline().replace(BOS_SYM, "").replace(EOS_SYM,"")
                val_results["samples"][current_article] = {}
            elif line == "In vocab summary:":
                val_results["samples"][current_article]["in_vocab_summary"] = f.readline().strip().replace(BOS_SYM, "").replace(EOS_SYM,"")
            elif line == "Complete article:":
                val_results["samples"][current_article]["complete_article"] = f.readline().strip().replace(BOS_SYM, "").replace(EOS_SYM,"")
            elif line == "Complete summary:":
                val_results["samples"][current_article]["complete_summary"] = f.readline().strip().replace(BOS_SYM, "").replace(EOS_SYM,"")
            elif line == "Predicted summary:":
                val_results["samples"][current_article]["predicted_summary"] = f.readline().strip().replace(BOS_SYM, "").replace(EOS_SYM,"")
            line = f.readline().strip()
    return val_results


def val_rouge(all_data):
    rouge_all = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])

    for epoch, epoch_data in all_data.items():
        all_scores_epoch = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        for article, results in epoch_data['samples'].items():
            gold = results["complete_summary"]
            model = results["predicted_summary"]
            scores = scorer.score(gold, model)
            for type, score in scores.items():
                all_scores_epoch[type].append(score)

        all_averages = {}
        for type in all_scores_epoch.keys():
            all_averages[type] = {}
            all_averages[type]["f_mean"] = np.mean([score.fmeasure for score in all_scores_epoch[type]])
            all_averages[type]["f_sd"] = np.std([score.fmeasure for score in all_scores_epoch[type]])
            all_averages[type]["recall_mean"] = np.mean([score.recall for score in all_scores_epoch[type]])
            all_averages[type]["recall_sd"] = np.std([score.recall for score in all_scores_epoch[type]])
            all_averages[type]["precision_mean"] = np.mean([score.precision for score in all_scores_epoch[type]])
            all_averages[type]["precision_sd"] = np.std([score.precision for score in all_scores_epoch[type]])
        #print(all_averages)
        rouge_all[epoch] = all_averages
    return rouge_all


def val_rouge_evaluation():
    all_val_results = {}
    for val_file in glob("train_checkpoints_pointer_generator/validation_output_*.txt"):
        epoch = int(Path(val_file).stem.split("_")[-1])
        all_val_results[epoch] = get_val_results(val_file)

    train_loss = []
    for key, data in all_val_results.items():
        try:
            loss = float(data["epoch train loss"])
        except:
            loss = np.nan
        train_loss.append((key, loss))

    train_loss = sorted(train_loss, key=lambda x: x[0])
    train_epochs = [epoch for epoch, loss in train_loss]
    train_loss = [loss for epoch, loss in train_loss]

    rouge_scores = val_rouge(all_val_results)
    val_rouge1_f1_mean = []
    val_rouge2_f1_mean = []
    val_rougeLsum_f1_mean = []

    for epoch in train_epochs:
        val_rouge1_f1_mean.append(rouge_scores[epoch]["rouge1"]["f_mean"])
        val_rouge2_f1_mean.append(rouge_scores[epoch]["rouge2"]["f_mean"])
        val_rougeLsum_f1_mean.append(rouge_scores[epoch]["rougeLsum"]["f_mean"])

    if not "plot rouge by epochs":
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True)
        axes[0].bar(train_epochs, val_rouge1_f1_mean, label="Rouge1", color=CB_color_cycle[5])
        axes[0].set_title("ROUGE1")
        axes[1].bar(train_epochs, val_rouge2_f1_mean, label="Rouge2", color=CB_color_cycle[0])
        axes[1].set_title("ROUGE2")
        axes[2].bar(train_epochs, val_rougeLsum_f1_mean, label="RougeL", color=CB_color_cycle[2])
        axes[2].set_title("ROUGELsum")
        axes[0].set_xlabel("Epochs")
        axes[1].set_xlabel("Epochs")
        axes[2].set_xlabel("Epochs")
        axes[0].set_ylabel("ROUGE Score")
        plt.suptitle("ROUGE Scores for Validation Data", size=24)
        plt.show()

    if "print evolution":
        # General evolution.
        sample = list(all_val_results[1]['samples'].keys())[1]
        print("Target: ", all_val_results[epoch]['samples'][sample]['in_vocab_summary'])
        for epoch in train_epochs:
            print(epoch, ":", all_val_results[epoch]['samples'][sample]["predicted_summary"])

        # Pointer example.
        sample = list(all_val_results[1]['samples'].keys())[5]
        print("Target: ", all_val_results[epoch]['samples'][sample]['in_vocab_summary'])
        for epoch in train_epochs:
            print(epoch, ":", all_val_results[epoch]['samples'][sample]["predicted_summary"])



if __name__ == "__main__":
    # do stuff
    encoder, decoder, last_epoch = load_best_model()
    val_data, test_data, vocab = load_test()
    test_subset = test_data.take(4)
    inference(encoder, decoder, vocab, test_subset, last_epoch, 0,
            'output/pointer_generator/qual', beam=False, data_type="test")

    for beam in [True, False]:
        if beam == True:
            directory = 'output/pointer_generator_beam'
        else:
            directory = 'output/pointer_generator'
        inference(encoder, decoder, vocab, test_data, last_epoch, 0,
                  directory, beam=beam, data_type="test")

    for epoch in range(1, last_epoch+1):
        encoder, decoder = load_specific_model(epoch)
        for beam in [True, False]:
            if beam == True:
                directory = 'output/pointer_generator_beam'
            else:
                directory = 'output/pointer_generator'
            inference(encoder, decoder, vocab, val_data, epoch, 0,
                      directory, beam=beam, data_type="val")

    #val_loss = get_val_train_loss(val_data, vocab, num_epochs=last_epoch)
    # val_rouge_evaluation()
    # greedy_inference(encoder, decoder, vocab, test_data, last_epoch,
    #                  0, 'output/pointer_generator', point_gen=True, data_type="test")
