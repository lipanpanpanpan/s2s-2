# train
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from datautil import *

def data_type():
    return tf.float32

from config import *
import seq2seq


def main():
    cf = TrainConfig()
    train_ids_path, dev_ids_path, vocab_path, vocab_size = prepare_data(cf.data_path,
                                                                        os.path.join(cf.data_path, 'dataset.vocab'))
    cf.vocab_size = vocab_size
    print(vocab_size)
    # make iteration of train and valid
    train_iter = Itertool(train_ids_path, batch_size=cf.batch_size,
                          max_len=cf.x_seq_len, eos_flag=True, go_flag=False)
    valid_iter = Itertool(dev_ids_path, batch_size=cf.batch_size,
                          max_len=cf.x_seq_len, eos_flag=True, go_flag=False)

    total_step_num = 0
    checkpoint_path = os.path.join(cf.model_dir, "model.ckpt")

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        x_in, lx_in, target_in, target_weight, keep_prob = seq2seq.get_inputs(cf)
        with tf.device('/gpu:2'):
            model = seq2seq.creat_model(sess, cf, x_in, lx_in, target_in, target_weight, keep_prob, infer=False)
        # train
        print('Training ... ...')
        step_time = 0.
        early_stop = 0
        early_test = []
        early_cost = 10000.0
        for e in range(cf.max_epoch):
            if early_stop:
                break
            outs = []
            total_weight = 0
            for x, len_x, y, y_weight in train_iter:
                #print(total_step_num)
                if len(len_x) < cf.batch_size:
                    break
                start_time = time.time()
                feed = dict(zip([x_in, lx_in, target_in, target_weight, keep_prob], [x, len_x, y, y_weight, 0.5]))
                out, _ = sess.run([model.cost, model.train_op], feed)
                outs.append(out)
                total_weight += np.sum(y_weight)
                total_step_num += 1
                step_time += (time.time() - start_time) / cf.save_freq
                if total_step_num % cf.save_freq == 0:
                    model.saver.save(sess, checkpoint_path, global_step=total_step_num)
                    print('Epoch:%d, BatchNum:%d, Step Time: %.06f, Cost: %0.6f' % (
                    e, total_step_num, step_time, np.exp(sum(outs) * cf.batch_size / total_weight)))
                    total_weight = 0
                    outs = []
                    step_time = 0.
                if total_step_num % cf.valid_freq == 0:
                    vcost = []
                    v_weight = 0
                    for x, len_x, y, y_weight in valid_iter:
                        if len(len_x) < cf.batch_size:
                            break
                        feed = dict(
                            zip([x_in, lx_in, target_in, target_weight, keep_prob], [x, len_x, y, y_weight, 1.0]))
                        out = sess.run(model.cost, feed)
                        vcost.append(out)
                        v_weight += np.sum(y_weight)
                    cost = np.exp((sum(vcost) * cf.batch_size / v_weight))

                    print('Validation Cost: %0.6f' % np.exp((sum(vcost) * cf.batch_size / v_weight)))
                    if cost < early_cost:
                        early_cost = cost
                        early_test = []
                    else:
                        early_test.append(cost)
                        if len(early_test)>20:
                            early_stop = 1

                sys.stdout.flush()
                # end of for
                # end of session


if __name__ == "__main__":
    main()


'''
Epoch:0, BatchNum:200, Step Time: 17.780765, Cost: 142.869204
Epoch:0, BatchNum:400, Step Time: 18.038926, Cost: 145.497687
Validation Cost: 136.066051
Epoch:0, BatchNum:600, Step Time: 17.508303, Cost: 133.588766
Epoch:0, BatchNum:800, Step Time: 23.577721, Cost: 130.759473
Epoch:0, BatchNum:1000, Step Time: 17.653312, Cost: 121.251534
'''
