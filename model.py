'''
Model class for final project. Pointer-generator base
architecture from See et al 2017, with RUM RNN unit
based on Dangovski et al 2019. Sequence to sequence
based on Nallapati et al 2016, as described in See
et al (constructed as an "intro to attention" for
myself).
'''

import tensorflow as tf
from tensorflow.keras import datasets, layers, \
                             models, optimizers, \
                             losses, metrics


# Encoder based on baseline from See et al 2017, which is
# based on Nallapati et al. 2016. Model construction informed
# by tutorial at https://www.tensorflow.org/tutorials/text/nmt_with_attention
class Encoder(tf.keras.Model):
    def __init__(self, vocab_len=50000, embed_dims=128, hidden_dims=256, rnn_type="LSTM", batch_size=128):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.embedding = layers.Embedding(input_dim=vocab_len, output_dim=embed_dims, input_length=400)

        if rnn_type == "LSTM":
            self.recurrent_layer = layers.Bidirectional(layers.LSTM(units=hidden_dims, activation='tanh',
                                                                    return_sequences=True, return_state=True,
                                                                    recurrent_initializer='glorot_uniform'))
        if rnn_type == "GRU":
            self.recurrent_layer = layers.Bidirectional(layers.GRU(units=hidden_dims, activation='tanh',
                                                                   return_sequences=True, return_state=True,
                                                                   recurrent_initializer='glorot_uniform'))
        if rnn_type == "RUM":
            raise NotImplementedError

    def call(self, w_i, hidden):
        if hidden is None:
            batch_size = w_i.shape[0]
            hidden = self.initialize_hidden_state(batch_size)

        # w_i.shape = (batch_size, word_int, sequence_len)
        # Sequence represented by vector length sequence_len,
        # where each word has been converted into an integer.
        # This vector is then fed into the embedding layer,
        # where it is converted into a Tensor with shape:
        # (batch_size, embed_dims, sequence_len)
        w_i = self.embedding(w_i)

        # Now, x.shape = (batch_size, sequence_len, embed_dims),
        # which is passed into the recurrent layer. The layer
        # outputs two items:
        #   - h_i -> sequence of hidden state outputs
        #   - hidden -> list of four items:
        #           hidden[0] -> final memory state for forward LSTM
        #           hidden[1] -> final carry state for forward LSTM
        #           hidden[2] -> final memory state for backward LSTM
        #           hidden[3] -> final carry state for backward LSTM
        h_i, *hidden = self.recurrent_layer(w_i, initial_state=hidden)
        return h_i, hidden

    def initialize_hidden_state(self, batch_size):
        return [tf.random.normal((batch_size, self.hidden_dims)) for i in range(4)]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_h = tf.keras.layers.Dense(units, use_bias=False)
        self.W_s = tf.keras.layers.Dense(units, use_bias=False)
        self.b_attn = tf.Variable(initial_value=1.0, trainable=True)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_t, h_i):
        # h_i: hidden state from encoder, shape:
        #       (batch_size, max_article_seq_len, #_RNN_unitsx2)
        # s_t: hidden state from decoder, shape:
        #       (batch_size, 1, # RNN units)

        # Additive style attention.
        # e_i^t = v.T * tanh(W_h*h_i + W_s*s_t+b_attn)
        # ...so e_t: (batch_size, max_article_seq_len, 1)
        e_t = self.V(tf.nn.tanh(self.W_h(h_i) + self.W_s(s_t) + self.b_attn))

        # Take softmax to get a proper distribution.
        # tf.sum(a_t, axis=1) will be a vector of
        # approximately ones, length of batch_size.
        a_t = tf.nn.softmax(e_t, axis=1)

        # Use attention distribution to weight
        # the encoder hidden states. Shape:
        #   (batch_size, max_article_seq_len, #_RNN_unitsx2)
        context = a_t * h_i

        # Finally, sum over each timepoint to produce
        # a context vector (batch_size, RNN hidden units*2)!
        # Woah! TODO: Think about the shape here.
        context = tf.reduce_sum(context, axis=1)
        return context, a_t

class Decoder(tf.keras.Model):
    def __init__(self, vocab_len=50000, embed_dims=128, hidden_dims=256, rnn_type="LSTM", batch_size=128):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.embedding = layers.Embedding(input_dim=vocab_len, output_dim=embed_dims, input_length=400)

        if rnn_type == "LSTM":
            self.recurrent_layer = layers.LSTM(units=hidden_dims, activation='tanh',
                                               return_sequences=True, return_state=True,
                                               recurrent_initializer='glorot_uniform')
        if rnn_type == "GRU":
            self.recurrent_layer = layers.GRU(units=hidden_dims, activation='tanh',
                                              return_sequences=True, return_state=True,
                                              recurrent_initializer='glorot_uniform')
        if rnn_type == "RUM":
            raise NotImplementedError

        self.attention = BahdanauAttention(self.hidden_dims)
        self.distribution_1 = layers.Dense(units=vocab_len, activation="linear", use_bias=True)
        self.distribution_2 = layers.Dense(units=vocab_len, activation="softmax", use_bias=True)

    def call(self, inputs, hidden, h_i):
        if hidden is None:
            hidden = self.build_initial_state()
        # Get embedding for single word at timepoint t.
        # Well, single word for each item in batch.
        #   (batch_size, 1, embedding_size)
        s_t = self.embedding(inputs)
        # Now, get decoder state.
        #   (batch_size, 1, # RNN hidden units)
        s_t, *hidden = self.recurrent_layer(s_t, hidden)
        # At this point, pass the encoder state sequence h_i
        # and the decoder state s_t into attention mechanism
        # to calculate the attention distribution (a^t), as
        # well as the context vector (h_t*). The calculations
        # which are occuring inside of the attention layer
        # are:
        #   e_i^t = v.T * tanh(W_h*h_i + W_s*s_t+b_attn)
        #   a^t = softmax(e^t)
        h_star, attention_weights = self.attention(s_t, h_i)

        # Reshaping context vector to concatenate with
        # encoder output.
        h_star = tf.expand_dims(tf.tile(h_star, tf.constant([s_t.shape[1],1])), axis=1)
        s_t_h_star = tf.concat([s_t, h_star], axis=-1)

        # Finally, pass through two dense linear layers with the
        # shape of the vocabulary, and a final softmax layer to
        # get the probability distribution over all words!
        # P_vocab=softmax(V′(V[st,h∗t]+b)+b′)
        P_vocab = self.distribution_2(self.distribution_1(s_t_h_star))
        return P_vocab, hidden, attention_weights

    def build_initial_state(self):
        return [tf.random.normal((self.batch_size, self.hidden_dims)) for i in range(2)]

