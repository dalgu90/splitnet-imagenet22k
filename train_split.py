#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
from IPython import embed
from StringIO import StringIO
import matplotlib.pyplot as plt

import imagenet_input as data_input
import resnet_split as resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('train_dataset', 'scripts/train_shuffle.txt', """Path to the ILSVRC2012 the training dataset list file""")
tf.app.flags.DEFINE_string('train_image_root', '/data1/common_datasets/imagenet_resized/', """Path to the root of ILSVRC2012 training images""")
tf.app.flags.DEFINE_string('val_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('val_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_integer('num_classes', 21841, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 7103405, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_val_instance', 7092449, """Number of val images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of GPUs.""")
tf.app.flags.DEFINE_integer('ngroups1', 1, """Grouping number on logits""")
tf.app.flags.DEFINE_integer('ngroups2', 1, """Grouping number on conv5_x""")
tf.app.flags.DEFINE_integer('ngroups3', 1, """Grouping number on conv4_x""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('gamma1', 0.001, """Overlap loss regularization parameter""")
tf.app.flags.DEFINE_float('gamma2', 0.001, """Weight split loss regularization parameter""")
tf.app.flags.DEFINE_float('gamma3', 0.001, """Uniform loss regularization paramter""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('val_interval', 1000, """Number of iterations to run a val""")
tf.app.flags.DEFINE_integer('val_iter', 100, """Number of iterations during a val""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS


def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tImageNet training root: %s' % FLAGS.train_image_root)
    print('\tImageNet training list: %s' % FLAGS.train_dataset)
    print('\tImageNet val root: %s' % FLAGS.val_image_root)
    print('\tImageNet val list: %s' % FLAGS.val_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of val images: %d' % FLAGS.num_val_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tNumber of GPUs: %d' % FLAGS.num_gpus)
    print('\tNumber of Groups: %d-%d-%d' % (FLAGS.ngroups3, FLAGS.ngroups2, FLAGS.ngroups1))
    print('\tBasemodel file: %s' % FLAGS.basemodel)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per validation: %d' % FLAGS.val_interval)
    print('\tSteps during validation: %d' % FLAGS.val_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        import multiprocessing
        num_threads = multiprocessing.cpu_count() / FLAGS.num_gpus
        print('Load ImageNet dataset(%d threads)' % num_threads)
        with tf.device('/cpu:0'):
            print('\tLoading training data from %s' % FLAGS.train_dataset)
            with tf.variable_scope('train_image'):
                train_images, train_labels = data_input.distorted_inputs(FLAGS.train_image_root, FLAGS.train_dataset
                                               , FLAGS.batch_size, True, num_threads=num_threads, num_sets=FLAGS.num_gpus)
            # tf.summary.image('images', train_images[0])
            print('\tLoading validation data from %s' % FLAGS.val_dataset)
            with tf.variable_scope('test_image'):
                val_images, val_labels = data_input.inputs(FLAGS.val_image_root, FLAGS.val_dataset
                                               , FLAGS.batch_size, False, num_threads=num_threads, num_sets=FLAGS.num_gpus)

        # Get splitted params
        if not FLAGS.basemodel:
            print('No basemodel found to load split params')
            sys.exit(-1)
        else:
            print('Load split params from %s' % FLAGS.basemodel)

            def get_perms(q_name, ngroups):
                split_q = reader.get_tensor(q_name)
                q_amax = np.argmax(split_q, axis=0)
                return [np.where(q_amax == i)[0] for i in range(ngroups)]

            reader = tf.train.NewCheckpointReader(FLAGS.basemodel)
            split_params = {}

            print('\tlogits...')
            base_logits_w = reader.get_tensor('logits/fc/weights')
            base_logits_b = reader.get_tensor('logits/fc/biases')
            split_p1_idxs = get_perms('group/split_p1/q', FLAGS.ngroups1)
            split_q1_idxs = get_perms('group/split_q1/q', FLAGS.ngroups1)

            logits_params = {'weights':[], 'biases':[], 'input_perms':[], 'output_perms':[]}
            for i in range(FLAGS.ngroups1):
                logits_params['weights'].append(base_logits_w[split_p1_idxs[i], :][:, split_q1_idxs[i]])
                logits_params['biases'].append(base_logits_b[split_q1_idxs[i]])
            logits_params['input_perms'] = split_p1_idxs
            logits_params['output_perms'] = split_q1_idxs
            split_params['logits'] = logits_params

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


                for i, unit_name in enumerate(['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'conv5_5', 'conv5_6']):
                    print('\t' + unit_name)
                    sp = {}
                    split_params[unit_name] = sp

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
        lr_decay_steps = map(float,FLAGS.lr_step_epoch.split(','))
        lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size/FLAGS.num_gpus for s in lr_decay_steps])
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            ngroups1=FLAGS.ngroups1,
                            ngroups2=FLAGS.ngroups2,
                            ngroups3=FLAGS.ngroups3,
                            split_params=split_params,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
        network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True)
        network_val.build_model()
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            # allow_soft_placement=True,
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
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
           saver.restore(sess, FLAGS.checkpoint)
           init_step = global_step.eval(session=sess)
           print('Load checkpoint %s' % FLAGS.checkpoint)
        elif FLAGS.basemodel:
            # Define a different saver to save model checkpoints
            # Select only base variables (exclude split layers)
            print('Load parameters from basemodel %s' % FLAGS.basemodel)
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                               not "logits" in var.name and
                               not "global_step" in var.name]
            if FLAGS.ngroups2 > 1:
                vars_restore = [var for var in vars_restore
                                if not "conv5_" in var.name]
            if FLAGS.ngroups3 > 1:
                vars_restore = [var for var in vars_restore
                                if not "conv4_" in var.name]
            saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)
            saver_restore.restore(sess, FLAGS.basemodel)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
                                                sess.graph)

        # Training!
        val_best_acc = 0.0
        for step in xrange(init_step, FLAGS.max_steps):
            # val
            if step % FLAGS.val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(FLAGS.val_iter):
                    loss_value, acc_value = sess.run([network_val.loss, network_val.acc],
                                feed_dict={network_val.is_train:False})
                    val_loss += loss_value
                    val_acc += acc_value
                val_loss /= FLAGS.val_iter
                val_acc /= FLAGS.val_iter
                val_best_acc = max(val_best_acc, val_acc)
                format_str = ('%s: (val)     step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, val_loss, val_acc))

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                summary_writer.add_summary(val_summary, step)
                summary_writer.flush()

            # Train
            lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, lr_decay_steps, step)
            start_time = time.time()
            _, loss_value, acc_value, train_summary_str = \
                    sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            # Display & Summary(training)
            if step % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
              char = sys.stdin.read(1)
              if char == 'b':
                embed()


def img_to_summary(img, tag="img"):
    s = StringIO()
    plt.imsave(s, img, cmap='bone', format='png')
    summary = tf.Summary.Value(tag=tag,
                               image=tf.Summary.Image(encoded_image_string=s.getvalue(),
                                                      height=img.shape[0],
                                                      width=img.shape[1]))
    return summary

def _merge_split_q(q, merge_idxs, name='merge'):
    ngroups, dim = q.shape
    max_idx = np.max(merge_idxs)
    temp_list = []
    for i in range(max_idx + 1):
        temp = []
        for j in range(ngroups):
            if merge_idxs[j] == i:
                 temp.append(q[j,:])
        temp_list.append(np.sum(temp, axis=0))
    ret = np.array(temp_list)

    return ret

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
