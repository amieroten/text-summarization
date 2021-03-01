'''
Module to import and prep data for use
in text summarization experiments.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from nltk.tokenize import sent_tokenize


# As in HW3, but S = summary, not sentence!
BOS_SYM = '[BOS]'
EOS_SYM = '[EOS]'

tokenizer = text.WhitespaceTokenizer()

# Preprocess datasets.
#TODO: Think about preprocessing.
def remove_newline(article, highlights):
    return tf.strings.regex_replace(article, "\n", ""), \
           tf.strings.regex_replace(highlights, "\n", "")

def add_EOS_BOS(article, highlights):
    article = tf.strings.join([BOS_SYM, article, EOS_SYM], separator=" ")
    highlights = tf.strings.join([BOS_SYM, highlights, EOS_SYM], separator=" ")
    return article, highlights

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
        output_sequence_length=max_sequence)

    ds_train = ds_train.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(add_EOS_BOS, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # "Train" vectorization layer on training articles.
    int_vectorize.adapt(ds_train.map(lambda article, highlights: article))

    def int_vectorize_map(article, highlights):
        article = tf.expand_dims(article, -1)
        highlights = tf.expand_dims(highlights, -1)
        return int_vectorize(article), int_vectorize(highlights)

    ds_train = ds_train.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size=batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(add_EOS_BOS, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size=batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(add_EOS_BOS, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
        output_sequence_length=max_sequence)


    ds_val = ds_val.map(remove_newline, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(add_EOS_BOS, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # "Train" vectorization layer on validate articles. *for debugging purposes only!!*
    int_vectorize.adapt(ds_val.map(lambda article, highlights: article))

    def int_vectorize_map(article, highlights):
        article = tf.expand_dims(article, -1)
        highlights = tf.expand_dims(highlights, -1)
        return int_vectorize(article), int_vectorize(highlights)

    ds_val = ds_val.map(int_vectorize_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.shuffle(ds_info.splits['validation'].num_examples)
    ds_val = ds_val.batch(batch_size=batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_val, None, None, int_vectorize.get_vocabulary()

if __name__ == "__main__":
    ds_train, ds_test, ds_val, vocab = load_cnn_dailymail_experiment()
    print("pause")