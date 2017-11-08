# encoding:utf-8
class TrainConfig(object):
    """Training config."""
    emb_size = 100  #
    vocab_size = 200000
    hid_size = 100
    batch_size = 128
    x_seq_len = 20  # max sequence length
    y_seq_len = 20
    grap_clip = 10.0
    lr = 0.0002
    keep_prob = 0.5
    max_epoch = 100
    save_freq = 200
    valid_freq = 500
    data_path = './data/'
    embed_W_path = ''
    model_dir = './models/'


class TestConfig(object):
    emb_size = 100
    vocab_size = 200000
    hid_size = 100
    batch_size = 1
    x_seq_len = 20
    y_seq_len = 1
    grap_clip = 10.0
    lr = 0.0005
    keep_prob = 1.0
    max_epoch = 1
    save_freq = 1
    valid_freq = 1
    data_path = './data/'
    model_dir = './models/'
    embed_W_path = '' #'data/w2v.vec.txt'