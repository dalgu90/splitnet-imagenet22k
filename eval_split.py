#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
import cPickle as pickle
from IPython import embed

import imagenet_input as data_input
import resnet_split as resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('test_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('test_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 50000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('ngroups1', 1, """Grouping number on logits""")
tf.app.flags.DEFINE_integer('ngroups2', 1, """Grouping number on conv5_x""")
tf.app.flags.DEFINE_integer('ngroups3', 1, """Grouping number on conv4_x""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('gamma1', 0.001, """split loss regularization paramter""")
tf.app.flags.DEFINE_float('gamma2', 0.001, """split loss regularization paramter""")
tf.app.flags.DEFINE_float('gamma3', 0.001, """split loss regularization paramter""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('checkpoint', './split_split-1-1-3/model.ckpt-149999', """Path to the model checkpoint file""")
tf.app.flags.DEFINE_string('basemodel', './group_split-1-1-3/model.ckpt-60000', """Path to the model basemodel file""")
tf.app.flags.DEFINE_string('output_file', './split_split-1-1-3/eval.pkl', """Path to the result pkl file""")
tf.app.flags.DEFINE_integer('test_iter', 250, """Number of test batches during the evaluation""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tImageNet test root: %s' % FLAGS.test_image_root)
    print('\tImageNet test list: %s' % FLAGS.test_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tNumber of Groups: %d-%d-%d' % (FLAGS.ngroups3, FLAGS.ngroups2, FLAGS.ngroups1))
    print('\tCheckpoint file: %s' % FLAGS.checkpoint)
    print('\tBasemodel file: %s' % FLAGS.basemodel)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Evaluation Configuration]')
    print('\tOutput file path: %s' % FLAGS.output_file)
    print('\tTest iterations: %d' % FLAGS.test_iter)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        print('Load ImageNet dataset')
        with tf.device('/cpu:0'):
            print('\tLoading test data from %s' % FLAGS.test_dataset)
            with tf.variable_scope('test_image'):
                test_images, test_labels = data_input.ten_crop_inputs(FLAGS.test_image_root, FLAGS.test_dataset, FLAGS.batch_size, num_threads=1)

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size*10, data_input.IMAGE_HEIGHT, data_input.IMAGE_WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size*10])

        # Get splitted params
        if not FLAGS.basemodel:
            print('No basemodel found to load split params')
            sys.exit(-1)
        else:
            print('Load split params from %s' % FLAGS.basemodel)
            print('\tlogits...')
            reader = tf.train.NewCheckpointReader(FLAGS.basemodel)
            base_split_p1 = reader.get_tensor('group/split_p1/q')
            base_split_q1 = reader.get_tensor('group/split_q1/q')
            base_logits_w = reader.get_tensor('logits/fc/weights')
            base_logits_b = reader.get_tensor('logits/fc/biases')

            split_params = {}

            split_p1_amax = np.argmax(base_split_p1, axis=0)
            split_q1_amax = np.argmax(base_split_q1, axis=0)
            split_p1_idxs = [np.where(split_p1_amax == i)[0] for i in range(FLAGS.ngroups1)]
            split_q1_idxs = [np.where(split_q1_amax == i)[0] for i in range(FLAGS.ngroups1)]
            logits_params = {'weights':[], 'biases':[], 'input_perms':[], 'output_perms':[]}
            for i in range(FLAGS.ngroups1):
                logits_params['weights'].append(base_logits_w[split_p1_idxs[i], :][:, split_q1_idxs[i]])
                logits_params['biases'].append(base_logits_b[split_q1_idxs[i]])
            logits_params['input_perms'] = split_p1_idxs
            logits_params['output_perms'] = split_q1_idxs
            split_params['logits'] = logits_params

            def get_perms(q_name, ngroups):
                split_q = reader.get_tensor(q_name)
                q_amax = np.argmax(split_q, axis=0)
                return [np.where(q_amax == i)[0] for i in range(ngroups)]

            if FLAGS.ngroups2 > 1:
                print('\tconv5_x...')
                base_conv5_1_shortcut_k = reader.get_tensor('conv5_1/shortcut/kernel')
                base_conv5_1_conv1_k = reader.get_tensor('conv5_1/conv_1/kernel')
                base_conv5_1_conv2_k = reader.get_tensor('conv5_1/conv_2/kernel')
                base_conv5_2_conv1_k = reader.get_tensor('conv5_2/conv_1/kernel')
                base_conv5_2_conv2_k = reader.get_tensor('conv5_2/conv_2/kernel')
                split_p2_idxs = get_perms('group/split_p2/q', FLAGS.ngroups2)
                split_q2_idxs = _merge_split_idxs(split_p1_idxs, _get_even_merge_idxs(FLAGS.ngroups1, FLAGS.ngroups2))
                split_r21_idxs = get_perms('group/split_r21/q', FLAGS.ngroups2)
                split_r22_idxs = get_perms('group/split_r22/q', FLAGS.ngroups2)

                conv5_1_params = {'shortcut':[], 'conv1':[], 'conv2':[], 'p_perms':[], 'q_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups2):
                    conv5_1_params['shortcut'].append(base_conv5_1_shortcut_k[:,:,split_p2_idxs[i],:][:,:,:,split_q2_idxs[i]])
                    conv5_1_params['conv1'].append(base_conv5_1_conv1_k[:,:,split_p2_idxs[i],:][:,:,:,split_r21_idxs[i]])
                    conv5_1_params['conv2'].append(base_conv5_1_conv2_k[:,:,split_r21_idxs[i],:][:,:,:,split_q2_idxs[i]])
                conv5_1_params['p_perms'] = split_p2_idxs
                conv5_1_params['q_perms'] = split_q2_idxs
                conv5_1_params['r_perms'] = split_r21_idxs
                split_params['conv5_1'] = conv5_1_params

                conv5_2_params = {'conv1':[], 'conv2':[], 'p_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups2):
                    conv5_2_params['conv1'].append(base_conv5_2_conv1_k[:,:,split_q2_idxs[i],:][:,:,:,split_r22_idxs[i]])
                    conv5_2_params['conv2'].append(base_conv5_2_conv2_k[:,:,split_r22_idxs[i],:][:,:,:,split_q2_idxs[i]])
                conv5_2_params['p_perms'] = split_q2_idxs
                conv5_2_params['r_perms'] = split_r22_idxs
                split_params['conv5_2'] = conv5_2_params

            if FLAGS.ngroups3 > 1:
                print('\tconv4_x...')
                base_conv4_1_shortcut_k = reader.get_tensor('conv4_1/shortcut/kernel')
                base_conv4_1_conv1_k = reader.get_tensor('conv4_1/conv_1/kernel')
                base_conv4_1_conv2_k = reader.get_tensor('conv4_1/conv_2/kernel')
                base_conv4_2_conv1_k = reader.get_tensor('conv4_2/conv_1/kernel')
                base_conv4_2_conv2_k = reader.get_tensor('conv4_2/conv_2/kernel')
                split_p3_idxs = get_perms('group/split_p3/q', FLAGS.ngroups3)
                split_q3_idxs = _merge_split_idxs(split_p2_idxs, _get_even_merge_idxs(FLAGS.ngroups2, FLAGS.ngroups3))
                split_r31_idxs = get_perms('group/split_r31/q', FLAGS.ngroups3)
                split_r32_idxs = get_perms('group/split_r32/q', FLAGS.ngroups3)

                conv4_1_params = {'shortcut':[], 'conv1':[], 'conv2':[], 'p_perms':[], 'q_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups3):
                    conv4_1_params['shortcut'].append(base_conv4_1_shortcut_k[:,:,split_p3_idxs[i],:][:,:,:,split_q3_idxs[i]])
                    conv4_1_params['conv1'].append(base_conv4_1_conv1_k[:,:,split_p3_idxs[i],:][:,:,:,split_r31_idxs[i]])
                    conv4_1_params['conv2'].append(base_conv4_1_conv2_k[:,:,split_r31_idxs[i],:][:,:,:,split_q3_idxs[i]])
                conv4_1_params['p_perms'] = split_p3_idxs
                conv4_1_params['q_perms'] = split_q3_idxs
                conv4_1_params['r_perms'] = split_r31_idxs
                split_params['conv4_1'] = conv4_1_params

                conv4_2_params = {'conv1':[], 'conv2':[], 'p_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups3):
                    conv4_2_params['conv1'].append(base_conv4_2_conv1_k[:,:,split_q3_idxs[i],:][:,:,:,split_r32_idxs[i]])
                    conv4_2_params['conv2'].append(base_conv4_2_conv2_k[:,:,split_r32_idxs[i],:][:,:,:,split_q3_idxs[i]])
                conv4_2_params['p_perms'] = split_q3_idxs
                conv4_2_params['r_perms'] = split_r32_idxs
                split_params['conv4_2'] = conv4_2_params

        # Build model
        hp = resnet.HParams(batch_size=FLAGS.batch_size*10,
                            num_gpus=1,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            ngroups1=FLAGS.ngroups1,
                            ngroups2=FLAGS.ngroups2,
                            ngroups3=FLAGS.ngroups3,
                            split_params=split_params,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
        network = resnet.ResNet(hp, [images], [labels], global_step)
        network.build_model()
        print('\tNumber of Weights: %d' % network._weights)
        print('\tFLOPs: %d' % network._flops)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        '''debugging attempt
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        def _get_data(datum, tensor):
            return tensor == train_images
        sess.add_tensor_filter("get_data", _get_data)
        '''

        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)
            print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            print('No checkpoint file')
            sys.exit(-1)

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        # Test!(10-crop testing. Average softmax scores)
        def softmax(logits):
            x = np.array(logits)
            x = x - np.max(x, axis=1, keepdims=True)
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        test_acc = 0.0
        test_time = 0.0
        confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int32)
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images, test_labels])
            start_time = time.time()
            loss_value, acc_value, pred_value, logits_value = sess.run([network.loss, network.acc, network.preds, network.logits],
                        feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
            duration = time.time() - start_time

            probs_val = softmax(logits_value)
            probs_avg = []
            for j in range(FLAGS.batch_size):
                probs_avg.append(np.average(probs_val[j*10:(j+1)*10], axis=0))
            probs_avg = np.array(probs_avg)
            preds_avg = np.argmax(probs_avg, axis=1)
            labels_avg = np.array([test_labels_val[j*10] for j in range(FLAGS.batch_size)])
            acc_avg = np.average(preds_avg == labels_avg)

            test_acc += acc_avg
            test_time += duration
            for l, p in zip(labels_avg, preds_avg):
                confusion_matrix[l, p] += 1

            if i % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: iter %d, acc=%.4f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), i, acc_avg, examples_per_sec, sec_per_batch))
        test_acc /= FLAGS.test_iter

        # Print and save results
        sec_per_image = test_time/FLAGS.test_iter/FLAGS.batch_size/10
        print ('Done! Acc: %.6f, Test time: %.3f sec, %.7f sec/example' % (test_acc, test_time, sec_per_image))
        print ('Saving result... ')
        result = {'accuracy': test_acc, 'confusion_matrix': confusion_matrix,
                  'test_time': test_time, 'sec_per_image': sec_per_image}
        with open(FLAGS.output_file, 'wb') as fd:
            pickle.dump(result, fd)
        print ('done!')


def _merge_split_idxs(split_idxs, merge_idxs, name='merge'):
    ngroups = len(split_idxs)
    max_idx = np.max(merge_idxs)
    ret = []
    for i in range(max_idx + 1):
        temp = []
        for j in range(ngroups):
            if merge_idxs[j] == i:
                 temp.append(split_idxs[j])
        ret.append(np.concatenate(temp))

    return ret

def _get_even_merge_idxs(N, split):
    assert N >= split
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]



def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
