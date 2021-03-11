'''
Module to import and prep data for use
in text summarization experiments.
'''
import string
import re

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# As in HW3, but S = summary, not sentence!
BOS_SYM = '<BOS>'
EOS_SYM = '<EOS>'

# Preprocess datasets.
def remove_newline(article, highlights):
    return tf.strings.regex_replace(article, "\n", ""), \
           tf.strings.regex_replace(highlights, "\n", "")

def duplicate_originals(article, highlights):
    original_article = tf.identity(article)
    original_highlights = tf.identity(highlights)
    return article, highlights, original_article, original_highlights

def standardize_map(article, highlights, original_article, original_highlights):
    # Keep '!', '.', ',','?', '&', '$'.
    punc_to_remove = '"#%\'()*+/:;<=>@[\\]^_`-{|}~'
    regex_to_remove = f'[{re.escape(punc_to_remove)}]'

    original_article = tf.strings.lower(original_article)
    original_article = tf.strings.regex_replace(original_article, regex_to_remove, " ")
    original_article = tf.strings.regex_replace(original_article, "([?!.,&])", r" \1 ")
    original_article = tf.strings.join([BOS_SYM, original_article, EOS_SYM], separator=" ")

    original_highlights = tf.strings.lower(original_highlights)
    original_highlights = tf.strings.regex_replace(original_highlights, regex_to_remove, " ")
    original_highlights = tf.strings.regex_replace(original_highlights, "([?!.,&])", r" \1 ")
    original_highlights = tf.strings.join([BOS_SYM, original_highlights, EOS_SYM], separator=" ")

    return article, highlights, original_article, original_highlights

def standardize(text):
    punc_to_remove = '"#%\'()*+-/:;<=>@[\\]^_`-{|}~'
    regex_to_remove = f'[{re.escape(punc_to_remove)}]'

    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, regex_to_remove, " ")
    text = tf.strings.regex_replace(text, "([?!.,&])", r" \1 ")
    text = tf.strings.join([BOS_SYM, text, EOS_SYM], separator=" ")
    return text

def inspect_processed_examples(dataset, vocab):
    for item in dataset.take(5).as_numpy_iterator():
        article = " ".join([vocab[int(word)] for word in item[0][0]])
        summary = " ".join([vocab[int(word)] for word in item[1][0]])
        print(article)
        print(summary)

# Load and preprocess CNN/DailyMail corpus using tensorflow
# datasets pipeline. Informed by:
# https://www.tensorflow.org/datasets/keras_example
def load_cnn_dailymail_lead(batch_size=128, max_vocab=5000, max_sequence=400):

    (_, ds_test, ds_val), ds_info = tfds.load("cnn_dailymail",
                                                     split=['train', 'test', 'validation'],
                                                     shuffle_files=True,
                                                     as_supervised=True,
                                                     with_info=True)

    ds_val = ds_val.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_val

# Load and preprocess CNN/DailyMail corpus using tensorflow
# datasets pipeline. Informed by:
# https://www.tensorflow.org/datasets/keras_example
# TODO: Preprocess targets?
def load_cnn_dailymail_deep(batch_size=1, max_vocab=5000, max_sequence=400):

    (ds_train, ds_test, ds_val), ds_info = tfds.load("cnn_dailymail",
                                                     split=['train', 'test', 'validation'],
                                                     shuffle_files=True,
                                                     as_supervised=True,
                                                     with_info=True)

    int_vectorize = TextVectorization(
        max_tokens=max_vocab,
        output_mode='int',
        output_sequence_length=max_sequence,
        standardize=standardize)

    ds_train = ds_train.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # "Train" vectorization layer on training articles.
    int_vectorize.adapt(ds_train.map(lambda article, highlights: article))

    def int_vectorize_map(article, highlights, original_article, original_highlights):
        article = tf.expand_dims(article, -1)
        highlights = tf.expand_dims(highlights, -1)
        return int_vectorize(article), int_vectorize(highlights), original_article, original_highlights

    ds_train = ds_train.map(duplicate_originals, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(standardize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size=batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(duplicate_originals, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(standardize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size=batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(duplicate_originals, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(standardize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.shuffle(ds_info.splits['validation'].num_examples)
    ds_val = ds_val.batch(batch_size=batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, int_vectorize.get_vocabulary()

def load_cnn_dailymail_experiment(batch_size=1, max_vocab=5000, max_sequence=400):

    (_, _, ds_val), ds_info = tfds.load("cnn_dailymail",
                                                     split=['train', 'test', 'validation'],
                                                     shuffle_files=True,
                                                     as_supervised=True,
                                                     with_info=True)

    int_vectorize = TextVectorization(
        max_tokens=max_vocab,
        output_mode='int',
        output_sequence_length=max_sequence,
        standardize=standardize)

    ds_val = ds_val.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_val = ds_val.map(add_EOS_BOS, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # "Train" vectorization layer on validate articles. *for debugging purposes only!!*
    int_vectorize.adapt(ds_val.map(lambda article, highlights: article))

    def int_vectorize_map(article, highlights, original_article, original_highlights):
        article = tf.expand_dims(article, -1)
        highlights = tf.expand_dims(highlights, -1)
        return int_vectorize(article), int_vectorize(highlights), original_article, original_highlights

    ds_val = ds_val.map(duplicate_originals, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(standardize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.shuffle(ds_info.splits['validation'].num_examples)
    ds_val = ds_val.batch(batch_size=batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_val, None, None, int_vectorize.get_vocabulary()

if __name__ == "__main__":
    ds_train, ds_test, ds_val, vocab = load_cnn_dailymail_experiment()
    print("pause")