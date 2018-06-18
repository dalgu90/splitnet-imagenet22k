from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                    'batch_size, num_gpus, num_classes, weight_decay, momentum, finetune, '
                    'ngroups1, ngroups2, ngroups3, split_params')


class ResNet(object):
    def __init__(self, hp, images, labels, global_step, name=None, reuse_weights=False):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels # Input labels
        self._global_step = global_step
        self._name = name
        self._reuse_weights = reuse_weights
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

    def build_tower(self, images, labels):
        filters = [64, [64, 256], [128, 512], [256, 1024], [512, 2048]]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 1, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block_first(x, filters[1][0], filters[1][1], strides[1], name='conv2_1')
        x = self._residual_block(x, filters[1][0], filters[1][1], name='conv2_2')
        x = self._residual_block(x, filters[1][0], filters[1][1], name='conv2_3')

        # conv3_x
        x = self._residual_block_first(x, filters[2][0], filters[2][1], strides[2], name='conv3_1')
        x = self._residual_block(x, filters[2][0], filters[2][1], name='conv3_2')
        x = self._residual_block(x, filters[2][0], filters[2][1], name='conv3_3')
        x = self._residual_block(x, filters[2][0], filters[2][1], name='conv3_4')

        # conv4_x
        if self._hp.ngroups3 == 1:
            x = self._residual_block_first(x, filters[3][0], filters[3][1], strides[3], input_q=self.split_p3, output_q=self.split_q3, split_r1=self.split_r311, split_r2=self.split_r312, name='conv4_1')
            x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r321, split_r2=self.split_r322, name='conv4_2')
            x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r331, split_r2=self.split_r332, name='conv4_3')
            x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r341, split_r2=self.split_r342, name='conv4_4')
            x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r351, split_r2=self.split_r352, name='conv4_5')
            x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r361, split_r2=self.split_r362, name='conv4_6')
        else:
            for i, unit_name in enumerate(['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv4_5', 'conv4_6']):
                sp = self._hp.split_params[unit_name]
                if i == 0:  # conv4_1 only
                    x = self._residual_block_split_first(x, filters[3][0], filters[3][1], strides[3], sp['conv_shortcut'], sp['conv_1'], sp['conv_2'], sp['conv_3'],
                                                         sp['input_q_perms'], sp['output_q_perms'], sp['split_r1_perms'], sp['split_r2_perms'], name=unit_name)
                else:  # Otherwise
                    x = self._residual_block_split(x, filters[3][0], filters[3][1], sp['conv_1'], sp['conv_2'], sp['conv_3'],
                                                   sp['input_q_perms'], sp['split_r1_perms'], sp['split_r2_perms'], name=unit_namt)

        # conv5_x
        if self._hp.ngroups2 == 1:
            x = self._residual_block_first(x, filters[4][0], filters[4][1], strides[4], input_q=self.split_p2, output_q=self.split_q2, split_r1=self.split_r211, split_r2=self.split_r212, name='conv5_1')
            x = self._residual_block(x, filters[4][0], filters[4][1], input_q=self.split_q2, split_r1=self.split_r221, split_r2=self.split_r222, name='conv5_2')
            x = self._residual_block(x, filters[4][0], filters[4][1], input_q=self.split_q2, split_r1=self.split_r231, split_r2=self.split_r232, name='conv5_3')
        else:
            for i, unit_name in enumerate(['conv5_1', 'conv5_2', 'conv5_3']):
                sp = self._hp.split_params[unit_name]
                if i == 0:  # conv5_1 only
                    x = self._residual_block_split_first(x, filters[4][0], filters[4][1], strides[4], sp['conv_shortcut'], sp['conv_1'], sp['conv_2'], sp['conv_3'],
                                                         sp['input_q_perms'], sp['output_q_perms'], sp['split_r1_perms'], sp['split_r2_perms'], name=unit_name)
                else:  # Otherwise
                    x = self._residual_block_split(x, filters[4][0], filters[4][1], sp['conv_1'], sp['conv_2'], sp['conv_3'],
                                                   sp['input_q_perms'], sp['split_r1_perms'], sp['split_r2_perms'], name=unit_namt)

        # Logit
        logits_weights = self._hp.split_params['logits']['weights']
        logits_biases = self._hp.split_params['logits']['biases']
        logits_input_perms = self._hp.split_params['logits']['input_perms']
        logits_output_perms = self._hp.split_params['logits']['output_perms']
        print('\tBuilding unit: logits - %d split' % len(logits_weights))
        x_offset = 0
        x_list = []
        with tf.variable_scope('logits'):
            x = tf.reduce_mean(x, [1, 2])
            for i, (w, b, p) in enumerate(zip(logits_weights, logits_biases, logits_input_perms)):
                in_dim, out_dim = w.shape
                x_split = tf.transpose(tf.gather(tf.transpose(x), p))
                x_split = self._fc_with_init(x_split, out_dim, init_w=w, init_b=b, name='split%d' % (i+1))
                x_list.append(x_split)
                x_offset += in_dim
            x = tf.concat(x_list, 1)
            output_forward_idx = list(np.concatenate(logits_output_perms))
            output_inverse_idx = [output_forward_idx.index(i) for i in range(self._hp.num_classes)]
            x = tf.transpose(tf.gather(tf.transpose(x), output_inverse_idx))

        logits = x

        # Probs & preds & acc
        probs = tf.nn.softmax(x)
        preds = tf.to_int32(tf.argmax(logits, 1))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(preds, labels), ones, zeros)
        acc = tf.reduce_mean(correct)

        # Loss & acc
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=labels)
        loss = tf.reduce_mean(losses)

        return logits, preds, loss, acc


    def build_model(self):
        # Build towers for each GPU
        self._logits_list = []
        self._preds_list = []
        self._loss_list = []
        self._acc_list = []

        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d/' % i) as scope:
                    print('Build a tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    logits, preds, loss, acc = self.build_tower(self._images[i], self._labels[i])
                    self._logits_list.append(logits)
                    self._preds_list.append(preds)
                    self._loss_list.append(loss)
                    self._acc_list.append(acc)

        # Merge losses, accuracies of all GPUs
        with tf.device('/CPU:0'):
            self.logits = tf.concat(self._logits_list, 0, name="logits")
            self.preds = tf.concat(self._preds_list, 0, name="predictions")
            self.loss = tf.reduce_mean(self._loss_list, name="cross_entropy")
            tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy", self.loss)
            self.acc = tf.reduce_mean(self._acc_list, name="accuracy")
            tf.summary.scalar((self._name+"/" if self._name else "") + "accuracy", self.acc)


    def build_train_op(self):
        print('Building train ops')

        # Learning rate
        tf.summary.scalar((self._name+"/" if self._name else "") + 'learing_rate', self.lr)

        # Optimizer and gradients for each GPU
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        self._grads_and_vars_list = []

        # Computer gradients for each GPU
        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d/' % i) as scope:
                    print('Compute gradients of tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    losses = []

                    # Add l2 loss
                    with tf.variable_scope('l2_loss'):
                        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                        l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                        losses.append(l2_loss)

                    total_loss = self._loss_list[i] + tf.add_n(losses)

                    # Compute gradients of total loss
                    grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())

                    # Append gradients and vars
                    self._grads_and_vars_list.append(grads_and_vars)

        # Merge gradients
        print('Average gradients')
        with tf.device('/CPU:0'):
            grads_and_vars = self._average_gradients(self._grads_and_vars_list)

            # Finetuning
            if self._hp.finetune:
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if "group" in var.op.name or \
                            ("conv3_" in var.op.name and self._hp.ngroups3 > 1) or \
                            ("conv4_" in var.op.name and self._hp.ngroups2 > 1) or \
                            ("conv5_" in var.op.name) or \
                            "logits" in var.op.name:
                        print('\tScale up learning rate of % s by 10.0' % var.op.name)
                        grad = 10.0 * grad
                        grads_and_vars[idx] = (grad,var)

            # Apply gradient
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

            # Batch normalization moving average update
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(*(update_ops+[apply_grad_op]))


    def _residual_block_first(self, x, bt_channel, out_channel, strides,
                              input_q=None, output_q=None, split_r1=None, split_r2=None, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, input_q=input_q, output_q=output_q, name='conv_shortcut')
                shortcut = self._bn(shortcut, name='bn_shortcut')
            # Residual
            x = self._conv(x, 1, bt_channel, strides, input_q=input_q, output_q=split_r1, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, bt_channel, 1, input_q=split_r1, output_q=split_r2, name='conv_2')
            x = self._bn(x, name='bn_2')
            x = self._relu(x, name='relu_2')
            x = self._conv(x, 1, out_channel, 1, input_q=split_r2, output_q=output_q, name='conv_3')
            x = self._bn(x, name='bn_3')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu')
        return x

    def _residual_block(self, x, bt_channel, out_channel, input_q=None, split_r1=None, split_r2=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        assert out_channel == num_channel
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 1, bt_channel, 1, input_q=input_q, output_q=split_r1, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, bt_channel, 1, input_q=split_r1, output_q=split_r2, name='conv_2')
            x = self._bn(x, name='bn_2')
            x = self._relu(x, name='relu_2')
            x = self._conv(x, 1, out_channel, 1, input_q=split_r2, output_q=input_q, name='conv_3')
            x = self._bn(x, name='bn_3')

            x = x + shortcut
            x = self._relu(x, name='relu')
        return x

    def _residual_block_first_split(self, x, bt_channel, out_channel, strides, conv_shortcht_kernels, conv_1_kernels, conv_2_kernels, conv_3_kernels,
                              input_q_perms, output_q_perms, split_r1_perms, split_r2_perms, name="unit"):
        with tf.variable_scope(name) as scope:
            print('\tBuilding splitted residual unit: %s' % scope.name)
            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv_split(shortcut, out_channel, strides, conv_shortcut_kernels, input_q_perms, output_q_perms, name='conv_shortcut')
                shortcut = self._bn(shortcut, name='bn_shortcut')
            # Residual
            x = self._conv_split(x, bt_channel, strides, conv_1_kernels, input_q_perms, split_r1_perms, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv_split(x, bt_channel, 1, conv_2_kernels, split_r1_perms, split_r2_perms, name='conv_2')
            x = self._bn(x, name='bn_2')
            x = self._relu(x, name='relu_2')
            x = self._conv_split(x, out_channel, 1, conv_3_kernels, split_r2_perms, input_q_perms, name='conv_3')
            x = self._bn(x, name='bn_3')

            x = x + shortcut
            x = self._relu(x, name='relu')
        return x

    def _residual_block_split(self, x, bt_channel, out_channel, conv1_kernels, conv2_kernels, conv3_kernels,
                              input_q_perms, split_r1_perms, split_r2_perms, name="unit"):
        with tf.variable_scope(name) as scope:
            print('\tBuilding splitted residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv_split(x, bt_channel, 1, conv_1_kernels, input_q_perms, split_r1_perms, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv_split(x, bt_channel, 1, conv_2_kernels, split_r1_perms, split_r2_perms, name='conv_2')
            x = self._bn(x, name='bn_2')
            x = self._relu(x, name='relu_2')
            x = self._conv_split(x, out_channel, 1, conv_3_kernels, split_r2_perms, input_q_perms, name='conv_3')
            x = self._bn(x, name='bn_3')

            x = x + shortcut
            x = self._relu(x, name='relu')
        return x

    def _conv_split(self, x, out_channel, strides, kernels, input_perms, output_perms, name="unit"):
        b, w, h, in_channel = x.get_shape().as_list()
        x_list = []
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s - %d split' % (scope.name, len(kernels)))
            for i, (k, p) in enumerate(zip(kernels, input_perms)):
                kernel_size, in_dim, out_dim = k.shape[-3:]
                x_split = tf.transpose(tf.gather(tf.transpose(x, (3, 0, 1, 2)), p), (1, 2, 3, 0))
                x_split = self._conv_with_init(x_split, kernel_size, out_dim, strides, init_k=k, name="split%d"%(i+1))
                x_list.append(x_split)
        x = tf.concat(x_list, 3)
        output_forward_idx = list(np.concatenate(output_perms))
        output_inverse_idx = [output_forward_idx.index(i) for i in range(out_channel)]
        x = tf.transpose(tf.gather(tf.transpose(x, (3, 0, 1, 2)), output_inverse_idx), (1, 2, 3, 0))
        return x

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def _average_gradients2(self, tower_grads, group_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
          group_grad: List of (gradient, variable) tuples. The gradients are of
            the regularization(L2, group regularizations).
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        tower_average_grads = self._average_gradients(tower_grads)

        # Get all the variable names
        all_vars = set()
        for g, v in tower_average_grads + group_grads:
            all_vars.add(v)

        temp_grads = {v:[] for v in all_vars}
        for gv in tower_average_grads + group_grads:
            temp_grads[gv[1]].append(gv)

        average_grads = []
        for _, gvs in temp_grads.items():
            if len(gvs) == 1:
                if gvs[0][0] is not None:
                    average_grads.append(gvs[0])
            else:
                if gvs[0][0] is None:
                    average_grads.append(gvs[1])
                elif gvs[1][0] is None:
                    average_grads.append(gvs[0])
                else:
                    average_grads.append((gvs[0][0]+gvs[1][0], gvs[0][1]))

        return average_grads


    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _conv_with_init(self, x, filter_size, out_channel, stride, pad="SAME", init_k=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv_with_init(x, filter_size, out_channel, stride, pad, init_k, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc_with_init(self, x, out_dim, init_w=None, init_b=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc_with_init(x, out_dim, init_w, init_b, name)
        f = 2*(in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
