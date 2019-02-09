from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import attention_rnn
import rnn
import tfutils


class ModelBase(object):
    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        self.config = config
        self.is_training = is_training
        self.batch_size = batch_size = config.batch_size
        self.seq_length = seq_length = config.seq_length
        self.size = size = config.hidden_size
        self.vocab_size = vocab_size = config.vocab_size
        self.dropout_keep_rate = dropout_keep_rate

        self._input_data = input_data # = tf.placeholder(tf.int32, [seq_length, batch_size], name="inputs")
        self._targets = targets # = tf.placeholder(tf.int32, [seq_length, batch_size], name="targets")
        self._actual_lengths = lengths # tf.placeholder(tf.int32, [batch_size], name="actual_lengths")

        cell = self.create_cell()

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            self._embedding = embedding = tf.get_variable("embedding", [vocab_size, size])

            inputs = tf.gather(embedding, input_data)

        inputs = tf.nn.dropout(inputs, dropout_keep_rate)

        self._logits, self._predict, self._loss, self._final_state = self.output_and_loss(cell, inputs)
        self._cost = cost = tf.reduce_sum(self._loss) / batch_size

        print("Constructing Basic Model")

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def create_cell(self):
        raise NotImplementedError

    def rnn(self, cell, inputs):
        raise NotImplementedError

    def output_and_loss(self, cell, input):
        raise NotImplementedError

    @property
    def lr(self):
        return self._lr

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def logits(self):
        return self._logits

    @property
    def predict(self):
        return self._predict

    @property
    def loss(self):
        return self._loss

    @property
    def actual_lengths(self):
        return self._actual_lengths

    @property
    def is_attention_model(self):
        raise NotImplementedError

    @property
    def embedding_variable(self):
        return self._embedding


class BasicModel(ModelBase):
    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        super(BasicModel, self).__init__(is_training, config, targets, input_data, lengths, dropout_keep_rate)

    def create_cell(self, size=None):
        size = size or self.config.hidden_size

        lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True),
            output_keep_prob=self.dropout_keep_rate)] * self.config.num_layers, state_is_tuple=True)

        return lstm

    def rnn(self, cell, inputs):
        return tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=self.actual_lengths,
                                 initial_state=self._initial_state, time_major=True)

    def output_and_loss(self, cell, inputs):
        output, state = self.rnn(cell, inputs)

        output = tf.reshape(output, [-1, self.size])
        softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        predict = tf.nn.softmax(logits)

        labels = tf.reshape(self.targets, [self.batch_size * self.seq_length, 1])
        loss = tf.nn.sampled_softmax_loss(weights=tf.transpose(softmax_w), biases=softmax_b,labels=labels, inputs=output, num_sampled=self.config.num_samples, num_classes=self.vocab_size) \
            if self.config.num_samples > 0 else \
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.targets, [-1]))

        return logits, predict, loss, state

    @property
    def is_attention_model(self):
        return False


class AttentionModel(BasicModel):
    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        self._num_attns = len(config.attention)
        self._num_tasks = len(config.attention) + 1
        self._masks = masks #tf.placeholder(tf.bool,[config.seq_length, config.batch_size, len(config.attention)], name="masks")

        self._max_attention = config.max_attention
        self._lambda_type = config.lambda_type
        self._min_tensor = tf.ones([config.batch_size, self._max_attention]) * -1e-38

        super(AttentionModel, self).__init__(is_training, config, targets, input_data, lengths, dropout_keep_rate)
        print("Constructing Attention Model")

    @property
    def masks(self):
        return self._masks

    @property
    def num_tasks(self):
        return self._num_tasks

    def create_cell(self, size=None):
        cell = super(AttentionModel, self).create_cell()
        cell = attention_rnn.AttentionCell(cell, self._max_attention, self.size, self._num_attns,
                                           self._lambda_type, self._min_tensor)
        return cell

    def rnn(self, cell, inputs):
        inputs = tf.concat(axis=2, values=[inputs,
                            tf.cast(self._masks, tf.float32), # add mask
                            tf.cast(tf.expand_dims(self.input_data, 2), tf.float32)]) # add indexes of input data

        #return rnn.dynamic_attention_rnn(cell, inputs, self._max_attention, self.num_tasks, self.batch_size,
        #                                 sequence_length=self.actual_lengths, initial_state=self.initial_state)
        return rnn.attention_rnn(cell, inputs, self.seq_length, self.initial_state, self.batch_size,
                                 self.size, self._max_attention, self.num_tasks, sequence_length=self.actual_lengths)

    def output_and_loss(self, cell, inputs):
        def _attention_predict(alpha, attn_ids, batch_size, length, project_to):
            alpha = tf.reshape(alpha, [-1], name="att_reshape")
            attn_ids = tf.reshape(tf.cast(attn_ids, tf.int64), [-1, 1], name="att_id_reshape")
            initial_indices = tf.expand_dims(tfutils.tile_vector(tf.cast(tf.range(batch_size), tf.int64), length), 1,
                                             name="att_indices_expand")
            # (length*batch_size,1), with [[0][0][0][0][0][1][1][1][1][1]...]
            sp_indices = tf.concat(axis=1, values=[initial_indices, attn_ids], name="att_indices_concat")
            # (length*batch_size,2), with [[0,aid1][0,aid2]...]
            attention_probs = tf.sparse_to_dense(sp_indices, [batch_size, project_to], alpha, validate_indices=False,
                                                 name="att_sparse_to_dense")
            # maps the pointers in sp_indices to a matrix of [batch_size,project_to] based on the alpha values
            return attention_probs

        def weighted_average(inputs, weights):
            # inputs: (tasks, batch*t, vocab)
            # weights: (tasks, batch*t)
            # output: (batch*t, vocab)

            weights = tf.expand_dims(weights, 2)  # (tasks, batch*t, 1)
            weighted = inputs * weights  # (tasks, batch*t, vocab)
            return tf.reduce_sum(weighted, [0])

        output, alpha_tensor, attn_id_tensor, lmbda, state = self.rnn(cell, inputs)
        # inputs is a tensor of (steps, batch, size)
        output = tf.reshape(output, [-1, self.size], name="output_reshape")
        # (steps, batch, size) -> (steps*batch, size)

        lmbda = tf.reshape(lmbda, [-1, self.num_tasks], name="lmbda_reshape")  # (steps*batch, tasks)
        task_weights = tf.transpose(lmbda)

        # alpha_tensor has shape (steps, batch, num_attns, max_attention)
        alphas = [tf.reshape(alpha_tensor[:, :, t, :], [-1, self._max_attention]) for t in range(self.num_tasks-1)]
        # the ids of all the things attended over with alphas also shape (steps, batch, num_attns, max_attention)
        attn_ids = [tf.reshape(attn_id_tensor[:, :, t, :], [-1, self._max_attention]) for t in range(self.num_tasks-1)]
        # each item: (steps, batch, k) -> (steps*batch, k)

        softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])

        logits = tf.matmul(output, softmax_w, name="logits_matmul") + softmax_b
        standard_predict = tf.nn.softmax(logits, name="softmax")  # (steps*batch, vocab)
        attn_predict = [
            _attention_predict(alpha,
                               attn_id,
                               self.batch_size * self.seq_length,
                               self._max_attention, self.vocab_size)
            for alpha, attn_id in zip(alphas, attn_ids)]  # [(steps*batch, vocab)]

        prediction_tensor = tf.stack([standard_predict] + attn_predict)
        predict = weighted_average(prediction_tensor, task_weights)
        # size  (steps*batch, vocab)
        # are theses already normailzed probabilities?

        labels = tf.reshape(self.targets, [-1], name="label_reshape")

        # Compute the loss, by computing the cross_entropy for two different parts differently
        # Why can't this just be computed using `predict`? Shouldn't it be enough to compute xent on that?
        lm_cross_entropy = tf.nn.sampled_softmax_loss(weights=tf.transpose(softmax_w), biases=softmax_b,labels=tf.exapnd_dims(labels,1), inputs=output, num_sampled=self.config.num_samples, num_classes=self.vocab_size) \
            if self.config.num_samples > 0 else \
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        attn_cross_entropies = [tfutils.cross_entropy_from_indices(labels, attn_id, alpha,
                                                                   self.batch_size*self.seq_length, self._max_attention)
                                for attn_id, alpha in zip(attn_ids, alphas)]

        cross_entropies = tf.stack([lm_cross_entropy] + attn_cross_entropies) * task_weights
        cross_entropy = tf.reduce_sum(cross_entropies, [0])

        # Try to make gradient much simpler using newer built-in gradients in tensorflow
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=labels)
        

        return logits, predict, cross_entropy, state

    @property
    def is_attention_model(self):
        return True


class AttentionOverOutputModel(AttentionModel):
    """
    Language model output is double the size, with half the output being considered the language model
    and half considered the input to the attention mechanism
    """

    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        super(AttentionOverOutputModel, self).__init__(is_training, config, targets, input_data, lengths, dropout_keep_rate, masks)
        print("Constructing Attention over Output Model")

    def create_cell(self, size=None):
        lstm = BasicModel.create_cell(self, size=self.size * 2)
        cell = attention_rnn.AttentionOverOutputCell(lstm, self._max_attention, self.size,
                                                     self._num_attns, self._lambda_type, self._min_tensor)
        return cell

    def rnn(self, cell, inputs):
        output, alpha_tensor, attn_id_tensor, lmbda, state = \
            super(AttentionOverOutputModel, self).rnn(cell, inputs)

        output = tf.slice(output, [0, 0, 0], [-1, -1, self.size])
        return output, alpha_tensor, attn_id_tensor, lmbda, state


class AttentionKeyValueModel(AttentionModel):
    """
    Language model output is triple the size, with the first 3rd of the output being considered the language model
    the second 3rd, the state that drives the attention mechanism
    and the last 3rd the vector representations attended over
    """

    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        super(AttentionKeyValueModel, self).__init__(is_training, config, targets, input_data, lengths, dropout_keep_rate, masks)
        print("Constructing Attention Key Value Model")

    def create_cell(self, size=None):
        lstm = BasicModel.create_cell(self, size=self.size * 3)
        cell = attention_rnn.AttentionKeyValueCell(lstm, self._max_attention, self.size,
                                                   self._num_attns, self._lambda_type, self._min_tensor)
        return cell

    def rnn(self, cell, inputs):
        output, state = super(AttentionKeyValueModel, self).rnn(cell, inputs)
        output = tf.slice(output, [0, 0, 0], [-1, -1, self.size])
        return output, state


class AttentionWithoutLambdaModel(AttentionKeyValueModel):
    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        super(AttentionWithoutLambdaModel, self).__init__(is_training, config, targets, input_data, lengths, dropout_keep_rate, masks)
        print("Constructing Attention Without Lambda Model")

    def create_cell(self, size=None):
        lstm = BasicModel.create_cell(self, size=self.size * 3)
        cell = attention_rnn.AttentionWithoutLambdaCell(lstm, self._max_attention, self.size,
                                                        self._num_attns, self._lambda_type, self._min_tensor)
        return cell


class AttentionBaselineModel(AttentionModel):
    def __init__(self, is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None):
        super(AttentionBaselineModel, self).__init__(is_training, config, targets, input_data, lengths, dropout_keep_rate, masks)
        print("Constructing Attention Baseline Model")

    def create_cell(self, size=None):
        lstm = BasicModel.create_cell(self)
        cell = attention_rnn.AttentionBaselineCell(lstm, self._max_attention, self.size,
                                                   self._num_attns, self._lambda_type, self._min_tensor)
        return cell