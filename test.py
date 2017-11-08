import os
import sys
from config import *
from datautil import *
import tensorflow as tf
import seq2seq

def test():
    cf = TestConfig()
    vocab_path = cf.data_path + "dataset.vocab"
    test_ids_path,vocab,rev_vocab,vocab_size = prepare_test_data(cf.data_path,vocab_path)
    print("vocab_size:%d"%vocab_size)
    cf.vocab_size = vocab_size
    test_iter = Itertool(test_ids_path,batch_size=cf.batch_size,\
                         max_len=cf.x_seq_len,eos_flag=True,go_flag=False)

    total_step_num = 0
    fout = open(cf.data_path+"result.data",'w',encoding='utf-8')
    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True,\
                                            log_device_placement = False)) as sess:
        x_in,lx_in,target_in,target_weight,keep_prob = seq2seq.get_inputs(cf)
        model = seq2seq.creat_model(sess,cf,x_in,lx_in,target_in,target_weight,keep_prob,infer=True)

        print("Testing ... ...")
        for x ,len_x,y,y_weight in test_iter:
            word_index_list = seq2seq.generation(sess,model,x,len_x)
            for idx in word_index_list:
                if idx == EOS_ID:
                    fout.write("\n")
                    break
                fout.write(rev_vocab[idx])

    fout.close()




if __name__ == '__main__':
    test()














