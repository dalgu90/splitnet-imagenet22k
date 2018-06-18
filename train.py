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
from tensorflow.python.client import timeline

import imagenet_input as data_input
import resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('train_dataset', 'scripts/train_shuffle.txt', """Path to the ImageNet training dataset list file""")
tf.app.flags.DEFINE_string('train_image_root', '/data1/common_datasets/imagenet_resized/', """Path to the root of ImageNet training images""")
tf.app.flags.DEFINE_string('val_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('val_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ImageNet test images""")
tf.app.flags.DEFINE_integer('num_classes', 21841, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 7103405, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_val_instance', 7092449, """Number of val images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch(per GPU).""")
tf.app.flags.DEFINE_integer('num_gpus', 4, """Number of GPUs.""")
tf.app.flags.DEFINE_integer('ngroups1', 1, """Grouping number on logits""")
tf.app.flags.DEFINE_integer('ngroups2', 1, """Grouping number on conv5_x""")
tf.app.flags.DEFINE_integer('ngroups3', 1, """Grouping number on conv4_x""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('gamma1', 1.0, """Overlap loss regularization parameter""")
tf.app.flags.DEFINE_float('gamma2', 1.0, """Weight split loss regularization parameter""")
tf.app.flags.DEFINE_float('gamma3', 1.0, """Uniform loss regularization paramter""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_boolean('bn_no_scale', False, """Whether not to use trainable gamma in BN layers.""")
tf.app.flags.DEFINE_boolean('weighted_group_loss', False, """Whether to normalize weight split loss where coeffs are propotional to its values.""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "5.0,10.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 500000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('val_interval', 1000, """Number of iterations to run a val""")
tf.app.flags.DEFINE_integer('val_iter', 100, """Number of iterations during a val""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_integer('group_summary_interval', None, """Interval for writing grouping visualization.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.96, """The fraction of GPU memory to be allocated""")
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
    print('\tOverlap loss weight: %f' % FLAGS.gamma1)
    print('\tWeight split loss weight: %f' % FLAGS.gamma2)
    print('\tUniform loss weight: %f' % FLAGS.gamma3)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tNo update on BN scale parameter: %d' % FLAGS.bn_no_scale)
    print('\tWeighted split loss: %d' % FLAGS.weighted_group_loss)
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
                            gamma1=FLAGS.gamma1,
                            gamma2=FLAGS.gamma2,
                            gamma3=FLAGS.gamma3,
                            momentum=FLAGS.momentum,
                            bn_no_scale=FLAGS.bn_no_scale,
                            weighted_group_loss=FLAGS.weighted_group_loss,
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
            # allow_soft_placement=False,
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            print('Load checkpoint %s' % FLAGS.checkpoint)
            saver.restore(sess, FLAGS.checkpoint)
            init_step = global_step.eval(session=sess)
        elif FLAGS.basemodel:
            # Define a different saver to save model checkpoints
            print('Load parameters from basemodel %s' % FLAGS.basemodel)
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                               not "group" in var.name and
                               not "global_step" in var.name]
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
            if step == 153:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_value, acc_value, train_summary_str = \
                        sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                                 feed_dict={network_train.is_train:True, network_train.lr:lr_value}
                                 , options=run_options, run_metadata=run_metadata)
                _ = sess.run(network_train.validity_op)
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(ctf)
                print('Wrote the timeline profile of %d iter training on %s' %(step, 'timeline.json'))
            else:
                _, loss_value, acc_value, train_summary_str = \
                        sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                                feed_dict={network_train.is_train:True, network_train.lr:lr_value})
                _ = sess.run(network_train.validity_op)
            duration = time.time() - start_time

            assert not np.isnan(loss_value)


            # Display & Summary(training)
            if step % FLAGS.display == 0 or step < 10:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
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

            # Does it work correctly?
            # if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
              # char = sys.stdin.read(1)
              # if char == 'b':
                # embed()

            # Add weights and groupings visualization
            filters = [64, [64, 256], [128, 512], [256, 1024], [512, 2048]]
            if FLAGS.group_summary_interval is not None:
                if step % FLAGS.group_summary_interval == 0:
                    img_summaries = []

                    if FLAGS.ngroups1 > 1:
                        logits_weights = get_var_value('logits/fc/weights', sess)
                        split_p1 = get_var_value('group/split_p1/q', sess)
                        split_q1 = get_var_value('group/split_q1/q', sess)
                        feature_indices = np.argsort(np.argmax(split_p1, axis=0))
                        class_indices = np.argsort(np.argmax(split_q1, axis=0))

                        img_summaries.append(img_to_summary(np.repeat(split_p1[:, feature_indices], 20, axis=0), 'split_p1'))
                        img_summaries.append(img_to_summary(np.repeat(split_q1[:, class_indices], 200, axis=0), 'split_q1'))
                        img_summaries.append(img_to_summary(np.abs(logits_weights[feature_indices, :][:, class_indices]), 'logits'))

                    if FLAGS.ngroups2 > 1:
                        conv5_1_shortcut = get_var_value('conv5_1/conv_shortcut/kernel', sess)
                        conv5_1_conv_1 = get_var_value('conv5_1/conv_1/kernel', sess)
                        conv5_1_conv_2 = get_var_value('conv5_1/conv_2/kernel', sess)
                        conv5_1_conv_3 = get_var_value('conv5_1/conv_3/kernel', sess)
                        conv5_2_conv_1 = get_var_value('conv5_2/conv_1/kernel', sess)
                        conv5_2_conv_2 = get_var_value('conv5_2/conv_2/kernel', sess)
                        conv5_2_conv_3 = get_var_value('conv5_2/conv_3/kernel', sess)
                        conv5_3_conv_1 = get_var_value('conv5_3/conv_1/kernel', sess)
                        conv5_3_conv_2 = get_var_value('conv5_3/conv_2/kernel', sess)
                        conv5_3_conv_3 = get_var_value('conv5_3/conv_3/kernel', sess)
                        split_p2 = get_var_value('group/split_p2/q', sess)
                        split_q2 = _merge_split_q(split_p1, _get_even_merge_idxs(FLAGS.ngroups1, FLAGS.ngroups2))
                        split_r211 = get_var_value('group/split_r211/q', sess)
                        split_r212 = get_var_value('group/split_r212/q', sess)
                        split_r221 = get_var_value('group/split_r221/q', sess)
                        split_r222 = get_var_value('group/split_r222/q', sess)
                        split_r231 = get_var_value('group/split_r231/q', sess)
                        split_r232 = get_var_value('group/split_r232/q', sess)
                        feature_indices1 = np.argsort(np.argmax(split_p2, axis=0))
                        feature_indices2 = np.argsort(np.argmax(split_q2, axis=0))
                        feature_indices3 = np.argsort(np.argmax(split_r211, axis=0))
                        feature_indices4 = np.argsort(np.argmax(split_r212, axis=0))
                        feature_indices5 = np.argsort(np.argmax(split_r221, axis=0))
                        feature_indices6 = np.argsort(np.argmax(split_r222, axis=0))
                        feature_indices7 = np.argsort(np.argmax(split_r231, axis=0))
                        feature_indices8 = np.argsort(np.argmax(split_r232, axis=0))
                        conv5_1_shortcut_img = np.abs(conv5_1_shortcut[:,:,feature_indices1,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[3][1], filters[4][1]))
                        conv5_1_conv_1_img = np.abs(conv5_1_conv_1[:,:,feature_indices1,:][:,:,:,feature_indices3].transpose([2,0,3,1]).reshape(filters[3][1], filters[4][0]))
                        conv5_1_conv_2_img = np.abs(conv5_1_conv_2[:,:,feature_indices3,:][:,:,:,feature_indices4].transpose([2,0,3,1]).reshape(filters[4][0] * 3, filters[4][0] * 3))
                        conv5_1_conv_3_img = np.abs(conv5_1_conv_3[:,:,feature_indices4,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[4][0], filters[4][1]))
                        conv5_2_conv_1_img = np.abs(conv5_2_conv_1[:,:,feature_indices2,:][:,:,:,feature_indices5].transpose([2,0,3,1]).reshape(filters[4][1], filters[4][0]))
                        conv5_2_conv_2_img = np.abs(conv5_2_conv_2[:,:,feature_indices5,:][:,:,:,feature_indices6].transpose([2,0,3,1]).reshape(filters[4][0] * 3, filters[4][0] * 3))
                        conv5_2_conv_3_img = np.abs(conv5_2_conv_3[:,:,feature_indices6,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[4][0], filters[4][1]))
                        conv5_3_conv_1_img = np.abs(conv5_3_conv_1[:,:,feature_indices2,:][:,:,:,feature_indices7].transpose([2,0,3,1]).reshape(filters[4][1], filters[4][0]))
                        conv5_3_conv_2_img = np.abs(conv5_3_conv_2[:,:,feature_indices7,:][:,:,:,feature_indices8].transpose([2,0,3,1]).reshape(filters[4][0] * 3, filters[4][0] * 3))
                        conv5_3_conv_3_img = np.abs(conv5_3_conv_3[:,:,feature_indices8,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[4][0], filters[4][1]))
                        img_summaries.append(img_to_summary(np.repeat(split_p2[:, feature_indices1], 20, axis=0), 'split_p2'))
                        img_summaries.append(img_to_summary(np.repeat(split_r211[:, feature_indices3], 20, axis=0), 'split_r211'))
                        img_summaries.append(img_to_summary(np.repeat(split_r212[:, feature_indices4], 20, axis=0), 'split_r212'))
                        img_summaries.append(img_to_summary(np.repeat(split_r221[:, feature_indices5], 20, axis=0), 'split_r221'))
                        img_summaries.append(img_to_summary(np.repeat(split_r222[:, feature_indices6], 20, axis=0), 'split_r222'))
                        img_summaries.append(img_to_summary(np.repeat(split_r231[:, feature_indices7], 20, axis=0), 'split_r231'))
                        img_summaries.append(img_to_summary(np.repeat(split_r232[:, feature_indices8], 20, axis=0), 'split_r232'))
                        img_summaries.append(img_to_summary(conv5_1_shortcut_img, 'conv5_1/shortcut'))
                        img_summaries.append(img_to_summary(conv5_1_conv_1_img, 'conv5_1/conv_1'))
                        img_summaries.append(img_to_summary(conv5_1_conv_2_img, 'conv5_1/conv_2'))
                        img_summaries.append(img_to_summary(conv5_1_conv_3_img, 'conv5_1/conv_3'))
                        img_summaries.append(img_to_summary(conv5_2_conv_1_img, 'conv5_2/conv_1'))
                        img_summaries.append(img_to_summary(conv5_2_conv_2_img, 'conv5_2/conv_2'))
                        img_summaries.append(img_to_summary(conv5_2_conv_3_img, 'conv5_2/conv_3'))
                        img_summaries.append(img_to_summary(conv5_3_conv_1_img, 'conv5_3/conv_1'))
                        img_summaries.append(img_to_summary(conv5_3_conv_2_img, 'conv5_3/conv_2'))
                        img_summaries.append(img_to_summary(conv5_3_conv_3_img, 'conv5_3/conv_3'))

                    # if FLAGS.ngroups3 > 1:
                        # conv4_1_shortcut = get_var_value('conv4_1/conv_shortcut/kernel', sess)
                        # conv4_1_conv_1 = get_var_value('conv4_1/conv_1/kernel', sess)
                        # conv4_1_conv_2 = get_var_value('conv4_1/conv_2/kernel', sess)
                        # conv4_2_conv_1 = get_var_value('conv4_2/conv_1/kernel', sess)
                        # conv4_2_conv_2 = get_var_value('conv4_2/conv_2/kernel', sess)
                        # split_p3 = get_var_value('group/split_p3/q', sess)
                        # split_q3 = _merge_split_q(split_p2, _get_even_merge_idxs(FLAGS.ngroups2, FLAGS.ngroups3))
                        # split_r31 = get_var_value('group/split_r31/q', sess)
                        # split_r32 = get_var_value('group/split_r32/q', sess)
                        # feature_indices1 = np.argsort(np.argmax(split_p3, axis=0))
                        # feature_indices2 = np.argsort(np.argmax(split_q3, axis=0))
                        # feature_indices3 = np.argsort(np.argmax(split_r31, axis=0))
                        # feature_indices4 = np.argsort(np.argmax(split_r32, axis=0))
                        # conv4_1_shortcut_img = np.abs(conv4_1_shortcut[:,:,feature_indices1,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[2], filters[3]))
                        # conv4_1_conv_1_img = np.abs(conv4_1_conv_1[:,:,feature_indices1,:][:,:,:,feature_indices3].transpose([2,0,3,1]).reshape(filters[2] * 3, filters[3] * 3))
                        # conv4_1_conv_2_img = np.abs(conv4_1_conv_2[:,:,feature_indices3,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[3] * 3, filters[3] * 3))
                        # conv4_2_conv_1_img = np.abs(conv4_2_conv_1[:,:,feature_indices2,:][:,:,:,feature_indices4].transpose([2,0,3,1]).reshape(filters[3] * 3, filters[3] * 3))
                        # conv4_2_conv_2_img = np.abs(conv4_2_conv_2[:,:,feature_indices4,:][:,:,:,feature_indices2].transpose([2,0,3,1]).reshape(filters[3] * 3, filters[3] * 3))
                        # img_summaries.append(img_to_summary(np.repeat(split_p3[:, feature_indices1], 20, axis=0), 'split_p3'))
                        # img_summaries.append(img_to_summary(np.repeat(split_r31[:, feature_indices3], 20, axis=0), 'split_r31'))
                        # img_summaries.append(img_to_summary(np.repeat(split_r32[:, feature_indices4], 20, axis=0), 'split_r32'))
                        # img_summaries.append(img_to_summary(conv4_1_shortcut_img, 'conv4_1/shortcut'))
                        # img_summaries.append(img_to_summary(conv4_1_conv_1_img, 'conv4_1/conv_1'))
                        # img_summaries.append(img_to_summary(conv4_1_conv_2_img, 'conv4_1/conv_2'))
                        # img_summaries.append(img_to_summary(conv4_2_conv_1_img, 'conv4_2/conv_1'))
                        # img_summaries.append(img_to_summary(conv4_2_conv_2_img, 'conv4_2/conv_2'))

                    if img_summaries:
                        img_summary = tf.Summary(value=img_summaries)
                        summary_writer.add_summary(img_summary, step)
                        summary_writer.flush()


def get_var_value(var_name, sess):
    return [var for var in tf.global_variables() if var_name in var.name][0].eval(session=sess)


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
                 temp.append(q[i,:])
        temp_list.append(np.sum(temp, axis=0))
    ret = np.array(temp_list)

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
