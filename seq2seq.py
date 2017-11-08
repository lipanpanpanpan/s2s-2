import tensorflow as tf
from tensorflow.contrib import  rnn
from tensorflow.contrib  import legacy_seq2seq as seq2seq
from tensorflow.python.platform import gfile
import random
import numpy as np
import copy
from datautil import *
import scipy.special as sps
import sys

np.random.seed(1234)


def get_inputs(config):
    keep_prob = tf.placeholder("float")  # dropout value
    x_in = tf.placeholder(tf.int32, [None, config.x_seq_len])  # input sequence
    lx_in = tf.placeholder(tf.int32, [None])  # length for input sequence, for mask padding data
    target_in = tf.placeholder(tf.int32, [None, config.y_seq_len])  # output sequence
    target_weight = tf.placeholder(tf.float32, [None, config.y_seq_len])  # mask for output sequence
    return x_in, lx_in, target_in, target_weight, keep_prob


class Seq2Seq(object):
    def __init__(self, emb_size, hid_size, batch_size, vocab_size, x_seq_len=20, y_seq_len=20, grad_clip=10., lr=1.0):
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_len = x_seq_len
        self.y_seq_len = y_seq_len
        self.grad_clip = grad_clip
        self.lr = lr

    def build_model(self, x_in, lx_in, target_in, target_weight, keep_prob, embed_W=None, infer=False):
        '''
        x_in
        lx_in
        target_in
        target_weight
        keep_prob: for dropout
        '''
        # elment component of total work
        self.x_in = x_in
        self.lx_in = lx_in
        self.target_in = target_in

        cell_fn = rnn.GRUCell
        encode_cell = cell_fn(self.hid_size)  # for encode
        decode_cell = cell_fn(self.hid_size)  # for decode
        if embed_W is None:
            embedding = tf.get_variable('word_embedding', [self.vocab_size, self.emb_size])
        else:
            assert embed_W.shape == (self.vocab_size, self.emb_size)
            embedding = tf.Variable(initial_value=embed_W, trainable=True, name='word_embedding')
        self.embedding = embedding
        # encoding
        enc_in = tf.nn.embedding_lookup(embedding, x_in)
        enc_in = tf.split(enc_in, self.seq_len,1)
        enc_in = [tf.squeeze(input_, [1]) for input_ in enc_in]

        en_outputs, en_state = rnn.static_rnn(encode_cell, enc_in, sequence_length=lx_in, dtype='float32', scope='encoder')
        en_outputs = tf.reshape(tf.concat( en_outputs,1), [-1, self.hid_size])

        self.en_state = en_state
        self.en_output = en_outputs
        self.encode_fn = encode_cell

        # decoding
        if infer == False:  # for training
            de_in = tf.nn.embedding_lookup(embedding, tf.concat([tf.zeros([self.batch_size, 1],dtype='int32'),
                                                                    target_in[:, :self.y_seq_len - 1]],1))
            de_in = tf.split( de_in,self.y_seq_len,1)
            de_in = [tf.squeeze(input_, [1]) for input_ in de_in]
        else:
            de_in = tf.nn.embedding_lookup(embedding, target_in)
            de_in = tf.split( de_in,self.y_seq_len,1)
            de_in = [tf.squeeze(input_, [1]) for input_ in de_in]
        # drop out
        # en_state = tf.nn.dropout(en_state, keep_prob)
        de_output, de_state = seq2seq.rnn_decoder(de_in, en_state, decode_cell, scope='decoder')
        de_output = tf.reshape(tf.concat(de_output,1), [-1, self.hid_size])
        self.decode_fn = decode_cell
        self.de_state = de_state
        self.de_output = de_output
        self.de_in = de_in

        # cost function
        softmax_w = tf.get_variable('softmax_w', [self.hid_size, self.vocab_size])
        softmax_b = tf.get_variable('softmax_b', [self.vocab_size])

        logits = tf.matmul(de_output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(target_in, [-1])],
            [tf.reshape(target_weight, [-1])])
        cost = tf.reduce_sum(loss) / self.batch_size  # from skip-thought, cost/self.batch_size
        self.cost = cost
        self.probs = tf.nn.softmax(logits)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50, keep_checkpoint_every_n_hours=1.0)
        # opt
        if infer == False:
            tvars = tf.trainable_variables()
            grads = tf.gradients(cost, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    #
    def encode_step(self, session_in, x_in_v, lx_in_v):
        '''
        *_t : tensor outputs = session.run(output_feed, input_feed)
        '''
        outputs = session_in.run([self.en_state], {self.x_in: x_in_v, self.lx_in: lx_in_v})
        return outputs[0]  # ,

    def encode_step_all(self, session_in, x_in_v, lx_in_v):
        '''
        *_t : tensor outputs = session.run(output_feed, input_feed)
        '''
        outputs = session_in.run([self.en_output, self.en_state], {self.x_in: x_in_v, self.lx_in: lx_in_v})
        return outputs[0], outputs[1]  # output & state

    def decode_step(self, session_in, t_in, state):
        '''
           each time on step
        '''
        outputs = session_in.run([self.probs, self.de_state], {self.target_in: t_in, self.en_state: state})
        probs_w = outputs[0]
        state = outputs[1]
        return probs_w, state

    def get_probs(self, session_in, x_in_v, lx_in_v, target_in_v):
        '''
        Obtain the prob distribution of target_in_v
        '''
        outputs = session_in.run([self.probs], {self.x_in: x_in_v, self.lx_in: lx_in_v, self.target_in: target_in_v})
        return outputs[0]


def creat_model(session_in, config, x_in, lx_in, target_in, target_weight, keep_prob, infer=False):
    if os.path.exists(config.embed_W_path):
        print('Loading embedding matrix ....')
        embed_W = np.loadtxt(config.embed_W_path, dtype='float32')
        embed_W = embed_W[:config.vocab_size, :]
        print(embed_W.shape)
    else:
        embed_W = None

    model = Seq2Seq(config.emb_size,
                        config.hid_size,
                        config.batch_size,
                        config.vocab_size, x_seq_len=config.x_seq_len,
                        y_seq_len=config.y_seq_len,
                        grad_clip=config.grap_clip, lr=config.lr)
    model.build_model(x_in, lx_in, target_in, target_weight, keep_prob, embed_W=embed_W, infer=infer)
    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        session_in.run(tf.global_variables_initializer())
        model.saver.restore(session_in, ckpt.model_checkpoint_path)
    else:
        print("Created model with parameters.")
        session_in.run(tf.global_variables_initializer())
    return model
def generation(session_in,model,x_in,lx_in,max_len = 20):
    next_w = np.zeros((1,1)).astype('int32')
    word_index_list = []

    output,next_state = model.encode_step_all(session_in,x_in,lx_in)
    for idx in range(max_len):
        next_p,next_state = model.decode_step(session_in,next_w,next_state)
        word_index = np.argmax(next_p,1)
        word_index_list.append(word_index[0])
        next_w = np.array([word_index])
    return word_index_list


def generate_beam_with_sample(session_in, model, x_in, lx_in, beam_k=10, maxlen=30, argmax=True,
                              use_unk=False):
    """
    Generate samples, using either beam search or stochastic sampling
    adding a sample process for top K words, ensure all prob words have access in.
    """  # f_init, f_next, ctx
    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype('float32')
    hyp_states = []

    output, next_state = model.encode_step_all(session_in, x_in, lx_in)

    next_w = np.zeros((1, 1)).astype('int32')

    for ii in range(maxlen):
        next_p, next_state = model.decode_step(session_in, next_w, next_state)

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()

        # ranks_flat = cand_flat.argsort()[:(beam_k-dead_k)]
        ranks_flat = cand_flat.argsort()[:5 * (beam_k - dead_k)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]
        # sample by zipf distribution
        next_p_flat = next_p.flatten()
        p_flat = next_p_flat[ranks_flat]
        # word_zifp = word_indices**(-1.4) / sps.zetac(1.4)
        word_uniform_p = np.random.random(5 * (beam_k - dead_k))
        select_prob = p_flat - word_uniform_p
        select_prob = select_prob.flatten()
        select_idx = select_prob.argsort()[-(beam_k - dead_k):]

        # update
        trans_indices = trans_indices[select_idx]
        word_indices = word_indices[select_idx]
        costs = costs[select_idx]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(beam_k - dead_k).astype('float32')
        new_hyp_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == EOS_ID:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= beam_k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_w = np.reshape(next_w, (next_w.shape[0], 1))  # prin(next_w.shape)
        next_state = np.array(hyp_states)

    # dump every remaining one
    if live_k > 0:
        for idx in range(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    return sample, sample_score
