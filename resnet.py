from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                    'batch_size, num_gpus, num_classes, weight_decay, momentum, finetune, '
                    'ngroups1, ngroups2, ngroups3, gamma1, gamma2, gamma3, '
                    'bn_no_scale, weighted_group_loss')

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
            x = self._bn(x, no_scale=self._hp.bn_no_scale)
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
        x = self._residual_block_first(x, filters[3][0], filters[3][1], strides[3], input_q=self.split_p3, output_q=self.split_q3, split_r1=self.split_r311, split_r2=self.split_r312, name='conv4_1')
        x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r321, split_r2=self.split_r322, name='conv4_2')
        x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r331, split_r2=self.split_r332, name='conv4_3')
        x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r341, split_r2=self.split_r342, name='conv4_4')
        x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r351, split_r2=self.split_r352, name='conv4_5')
        x = self._residual_block(x, filters[3][0], filters[3][1], input_q=self.split_q3, split_r1=self.split_r361, split_r2=self.split_r362, name='conv4_6')

        # conv5_x
        x = self._residual_block_first(x, filters[4][0], filters[4][1], strides[4], input_q=self.split_p2, output_q=self.split_q2, split_r1=self.split_r211, split_r2=self.split_r212, name='conv5_1')
        x = self._residual_block(x, filters[4][0], filters[4][1], input_q=self.split_q2, split_r1=self.split_r221, split_r2=self.split_r222, name='conv5_2')
        x = self._residual_block(x, filters[4][0], filters[4][1], input_q=self.split_q2, split_r1=self.split_r231, split_r2=self.split_r232, name='conv5_3')

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            x = self._fc(x, self._hp.num_classes, input_q=self.split_p1, output_q=self.split_q1)

        logits = x

        # Probs & preds & acc
        # probs = tf.nn.softmax(x)
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
        # Grouping variables
        filters = [64, [64, 256], [128, 512], [256, 1024], [512, 2048]]

        with tf.variable_scope("group"):
            if self._reuse_weights:
                tf.get_variable_scope().reuse_variables()

            if self._hp.ngroups1 > 1:
                self.split_q1 = utils._get_split_q(self._hp.ngroups1, self._hp.num_classes, name='split_q1')
                if not self._reuse_weights: tf.summary.histogram("group/split_q1/", self.split_q1)
                self.split_p1 = utils._get_split_q(self._hp.ngroups1, filters[4][1], name='split_p1')
                if not self._reuse_weights: tf.summary.histogram("group/split_p1/", self.split_p1)
            else:
                self.split_p1 = None
                self.split_q1 = None

            if self._hp.ngroups2 > 1:
                self.split_q2 = utils._merge_split_q(self.split_p1, utils._get_even_merge_idxs(self._hp.ngroups1, self._hp.ngroups2), name='split_q2')
                if not self._reuse_weights: tf.summary.histogram("group/split_q2/", self.split_q2)
                self.split_p2 = utils._get_split_q(self._hp.ngroups2, filters[3][1], name='split_p2')
                if not self._reuse_weights: tf.summary.histogram("group/split_p2/", self.split_p2)
                self.split_r211 = utils._get_split_q(self._hp.ngroups2, filters[4][0], name='split_r211')
                self.split_r212 = utils._get_split_q(self._hp.ngroups2, filters[4][0], name='split_r212')
                self.split_r221 = utils._get_split_q(self._hp.ngroups2, filters[4][0], name='split_r221')
                self.split_r222 = utils._get_split_q(self._hp.ngroups2, filters[4][0], name='split_r222')
                self.split_r231 = utils._get_split_q(self._hp.ngroups2, filters[4][0], name='split_r231')
                self.split_r232 = utils._get_split_q(self._hp.ngroups2, filters[4][0], name='split_r232')
                if not self._reuse_weights: tf.summary.histogram("group/split_r211/", self.split_r211)
                if not self._reuse_weights: tf.summary.histogram("group/split_r212/", self.split_r212)
                if not self._reuse_weights: tf.summary.histogram("group/split_r221/", self.split_r221)
                if not self._reuse_weights: tf.summary.histogram("group/split_r222/", self.split_r222)
                if not self._reuse_weights: tf.summary.histogram("group/split_r231/", self.split_r231)
                if not self._reuse_weights: tf.summary.histogram("group/split_r232/", self.split_r232)
            else:
                self.split_p2 = None
                self.split_q2 = None
                self.split_r211 = None
                self.split_r212 = None
                self.split_r221 = None
                self.split_r222 = None
                self.split_r231 = None
                self.split_r232 = None

            if self._hp.ngroups3 > 1:
                self.split_q3 = utils._merge_split_q(self.split_p2, utils._get_even_merge_idxs(self._hp.ngroups2, self._hp.ngroups3), name='split_q3')
                if not self._reuse_weights: tf.summary.histogram("group/split_q3/", self.split_q3)
                self.split_p3 = utils._get_split_q(self._hp.ngroups3, filters[2][1], name='split_p3')
                if not self._reuse_weights: tf.summary.histogram("group/split_p3/", self.split_p3)
                self.split_r311 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r311')
                self.split_r312 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r312')
                self.split_r321 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r321')
                self.split_r322 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r322')
                self.split_r331 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r331')
                self.split_r332 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r332')
                self.split_r341 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r341')
                self.split_r342 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r342')
                self.split_r351 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r351')
                self.split_r352 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r352')
                self.split_r361 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r361')
                self.split_r362 = utils._get_split_q(self._hp.ngroups3, filters[3][0], name='split_r362')
                if not self._reuse_weights: tf.summary.histogram("group/split_r311/", self.split_r311)
                if not self._reuse_weights: tf.summary.histogram("group/split_r312/", self.split_r312)
                if not self._reuse_weights: tf.summary.histogram("group/split_r321/", self.split_r321)
                if not self._reuse_weights: tf.summary.histogram("group/split_r322/", self.split_r322)
                if not self._reuse_weights: tf.summary.histogram("group/split_r331/", self.split_r331)
                if not self._reuse_weights: tf.summary.histogram("group/split_r332/", self.split_r332)
                if not self._reuse_weights: tf.summary.histogram("group/split_r341/", self.split_r341)
                if not self._reuse_weights: tf.summary.histogram("group/split_r342/", self.split_r342)
                if not self._reuse_weights: tf.summary.histogram("group/split_r351/", self.split_r351)
                if not self._reuse_weights: tf.summary.histogram("group/split_r352/", self.split_r352)
                if not self._reuse_weights: tf.summary.histogram("group/split_r361/", self.split_r361)
                if not self._reuse_weights: tf.summary.histogram("group/split_r362/", self.split_r362)
            else:
                self.split_p3 = None
                self.split_q3 = None
                self.split_r311 = None
                self.split_r312 = None
                self.split_r321 = None
                self.split_r322 = None
                self.split_r331 = None
                self.split_r332 = None
                self.split_r341 = None
                self.split_r342 = None
                self.split_r351 = None
                self.split_r352 = None
                self.split_r361 = None
                self.split_r362 = None

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

    def build_train_op(self):  # Set computing the gradient wrt for the group assignment variables
        # Learning rate
        tf.summary.scalar((self._name+"/" if self._name else "") + 'learing_rate', self.lr)

        # Optimizer and gradients for each GPU
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        all_vars = tf.trainable_variables()
        group_vars = [v for v in all_vars if v.name.startswith('group/')]
        model_vars = [v for v in all_vars if not v.name.startswith('group/')]

        self._gv_list_task_loss = []
        self._gv_group_loss = []

        # Computer gradients for each GPU
        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d/' % i) as scope:
                    print('Compute gradients of tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    losses = []

                    # Add l2 loss
                    if self._hp.weight_decay > 0.0:
                        with tf.variable_scope('l2_loss'):
                            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                            l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                            losses.append(l2_loss)

                    # Add group split loss
                    with tf.variable_scope('group'):
                        if tf.get_collection('OVERLAP_LOSS') and self._hp.gamma1 > 0.0:
                            cost1 = tf.reduce_mean(tf.get_collection('OVERLAP_LOSS'))
                            cost1 = cost1 * self._hp.gamma1
                            if i == 0:
                                tf.summary.scalar('group/overlap_loss/', cost1)
                            losses.append(cost1)

                        if tf.get_collection('WEIGHT_SPLIT') and self._hp.gamma2 > 0.0:
                            if self._hp.weighted_group_loss:
                                reg_weights = [tf.stop_gradient(x) for x in tf.get_collection('WEIGHT_SPLIT')]
                                regs = [tf.stop_gradient(x) * x for x in tf.get_collection('WEIGHT_SPLIT')]
                                cost2 = tf.reduce_sum(regs) / tf.reduce_sum(reg_weights)
                            else:
                                cost2 = tf.reduce_mean(tf.get_collection('WEIGHT_SPLIT'))
                            cost2 = cost2 * self._hp.gamma2
                            if i == 0:
                                tf.summary.scalar('group/weight_split_loss/', cost2)
                            losses.append(cost2)

                        if tf.get_collection('UNIFORM_LOSS') and self._hp.gamma3 > 0.0:
                            cost3 = tf.reduce_mean(tf.get_collection('UNIFORM_LOSS'))
                            cost3 = cost3 * self._hp.gamma3
                            if i == 0:
                                tf.summary.scalar('group/group_uniform_loss/', cost3)
                            losses.append(cost3)

                    if losses:
                        total_loss = self._loss_list[i] + tf.add_n(losses)

                    # Compute gradients of total loss
                    grads_and_vars = opt.compute_gradients(self._loss_list[i], model_vars)

                    # Append gradients and vars
                    self._gv_list_task_loss.append(grads_and_vars)

        # Computer gradients of regularization loss
        # It needs one more GPU
        with tf.device('/GPU:%d' % self._hp.num_gpus), tf.variable_scope(tf.get_variable_scope()):
            with tf.name_scope('tower_group/') as scope:
                print('Compute gradients of regularization loss: %s' % scope)
                if self._reuse_weights:
                    tf.get_variable_scope().reuse_variables()

                losses = []

                # Add l2 loss
                if self._hp.weight_decay > 0.0:
                    with tf.variable_scope('l2_loss'):
                        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                        l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                        losses.append(l2_loss)

                # Add group split loss
                with tf.variable_scope('group'):
                    if tf.get_collection('OVERLAP_LOSS') and self._hp.gamma1 > 0.0:
                        cost1 = tf.reduce_mean(tf.get_collection('OVERLAP_LOSS'))
                        cost1 = cost1 * self._hp.gamma1
                        if i == 0:
                            tf.summary.scalar('group/overlap_loss/', cost1)
                        losses.append(cost1)

                    if tf.get_collection('WEIGHT_SPLIT') and self._hp.gamma2 > 0.0:
                        if self._hp.weighted_group_loss:
                            reg_weights = [tf.stop_gradient(x) for x in tf.get_collection('WEIGHT_SPLIT')]
                            regs = [tf.stop_gradient(x) * x for x in tf.get_collection('WEIGHT_SPLIT')]
                            cost2 = tf.reduce_sum(regs) / tf.reduce_sum(reg_weights)
                        else:
                            cost2 = tf.reduce_mean(tf.get_collection('WEIGHT_SPLIT'))
                        cost2 = cost2 * self._hp.gamma2
                        if i == 0:
                            tf.summary.scalar('group/weight_split_loss/', cost2)
                        losses.append(cost2)

                    if tf.get_collection('UNIFORM_LOSS') and self._hp.gamma3 > 0.0:
                        cost3 = tf.reduce_mean(tf.get_collection('UNIFORM_LOSS'))
                        cost3 = cost3 * self._hp.gamma3
                        if i == 0:
                            tf.summary.scalar('group/group_uniform_loss/', cost3)
                        losses.append(cost3)

                if losses:
                    # Compute gradients of total loss
                    total_loss = tf.add_n(losses)
                    self._gv_group_loss = opt.compute_gradients(total_loss, all_vars)


        # Merge gradients
        print('Average gradients')
        with tf.device('/CPU:0'):
            grads_and_vars = self._average_gradients2(self._gv_list_task_loss, self._gv_group_loss)

            # Finetuning
            if self._hp.finetune:
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if "group" in var.op.name or \
                            (("conv3_" in var.op.name) and self._hp.ngroups3 > 1) or \
                            (("conv4_" in var.op.name) and self._hp.ngroups2 > 1) or \
                            ("conv5_" in var.op.name) or \
                            "logits" in var.op.name:
                        print('\tScale up learning rate of % s by 10.0' % var.op.name)
                        grad = 10.0 * grad
                        grads_and_vars[idx] = (grad,var)

            # Reduced gradient
            eps = 1e-5
            for idx, (grad, var) in enumerate(grads_and_vars):
                if "group" in var.name:
                    print('\tApply reduced gradient on ' + var.name)
                    ngroups, dim = var.get_shape().as_list()
                    zeros = tf.zeros((ngroups,dim), dtype=tf.float32)
                    zeros_col = tf.zeros((ngroups,), dtype=tf.float32)
                    zeros_row = tf.zeros((dim,), dtype=tf.float32)
                    ones = tf.ones((ngroups,dim), dtype=tf.float32)
                    ones_col = tf.ones((ngroups,), dtype=tf.float32)
                    ones_row = tf.ones((dim,), dtype=tf.float32)

                    mu = tf.cast(tf.argmax(var, 0), dtype=tf.int32)
                    offset = tf.constant([ngroups*i for i in range(dim)], dtype=tf.int32)
                    mask = tf.cast(tf.transpose(tf.one_hot(mu, ngroups)), dtype=tf.bool)

                    grad_mu = tf.gather(tf.reshape(tf.transpose(grad), [-1]), mu + offset)
                    grad_mu_tile = tf.tile(tf.reshape(grad_mu, [1,-1]), [ngroups, 1])
                    grad_1 = grad - grad_mu_tile
                    grad_2 = tf.where(tf.logical_and(tf.less_equal(var, ones*eps),
                                                    tf.greater(grad_1, zeros))
                                    , zeros, grad_1)

                    grad_red_mu = -tf.reduce_sum(grad_2, 0)
                    grad_red_mu = tf.tile(tf.reshape(grad_red_mu, [1,-1]), [ngroups, 1])
                    grad_red = tf.where(mask, grad_red_mu, grad_2)

                    max_step_size = tf.where(tf.greater(grad_red, zeros), var/grad_red, ones)
                    max_step_size = tf.reduce_min(max_step_size, 0)
                    lr_mult = tf.where(tf.less(max_step_size, self.lr), max_step_size/self.lr, ones_row)
                    # lr_mult = tf.where(tf.less(max_step_size, self.lr*10.0), max_step_size/self.lr/10.0, ones_row)
                    grad_red = grad_red * lr_mult
                    grads_and_vars[idx] = (grad_red,var)

                    tf.summary.scalar(var.op.name+"/lr_mult_min", tf.reduce_min(lr_mult))
                    tf.summary.scalar(var.op.name+"/lr_mult_avg", tf.reduce_mean(lr_mult))
                    tf.summary.histogram(var.op.name+"/group_sum", tf.reduce_sum(var, 0))
                    tf.summary.scalar(var.op.name+"/sparsity", tf.nn.zero_fraction(var-eps))
                    tf.summary.histogram(var.op.name+"/grad", grad_red)

            # Apply gradient
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

            # Batch normalization moving average update
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(*(update_ops+[apply_grad_op]), name="train_op")


        # Build validity op
        print('Build validity op')
        with tf.device('/CPU:0'):
            # Force the group variable to be non-negative and sum-to-one
            validity_op_list = []
            with tf.name_scope("sum-to-one"):
                for var in tf.trainable_variables():
                    if "group" in var.name:
                        ngroups, dim = var.get_shape().as_list()
                        ones = tf.ones((ngroups, dim), dtype=tf.float32)
                        zeros = tf.zeros((ngroups, dim), dtype=tf.float32)
                        var_temp = tf.where(tf.less(var, ones*eps), ones*eps, var)  # non-negativity
                        var_temp = var_temp / tf.reduce_sum(var_temp, 0)  # sum-to-one
                        assign_op = var.assign(var_temp)
                        validity_op_list.append(assign_op)
            self.validity_op = tf.group(*validity_op_list, name="group_validity")


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

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn", no_scale=False):
        x = utils._bn(x, self.is_train, self._global_step, name, no_scale=no_scale)
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
