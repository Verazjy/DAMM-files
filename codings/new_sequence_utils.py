import tensorflow as tf
import sys
import os
import re
from models.experimental.tf_revised_sequence_utils import sequence_input_layer, prune_invalid_ids
from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column import feature_column as fc
src = os.path.realpath(__file__)
for _ in 'model_lib/ranking/src/models'.split('/'):
    src = os.path.dirname(src)
    if not src in sys.path:
        sys.path.insert(0, src)
import utils
utils.add_custom_ops_python_path()
from share import tf_utils, shared_utils
import column_decorator as decorator
import network_utils
from collections import namedtuple
import inspect
import sequence_utils as seq_utils
from models.bert.modeling import transformer_model, create_attention_mask_from_input_mask
from tensorflow.estimator import ModeKeys
from sequence_utils import sequence_datum

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


sparse_embedding = namedtuple(  # similar to tf.IndexedSlices.
    'sparse_embedding', ['embedding_2d', 'term_indices', 'sequence_length'],
    verbose=False)


def segment_cartesian(embeddings_a, segment_ids_a, embeddings_b,
                      segment_ids_b):
    from segment_fn import segment_cross2d, segment_cross2d_grad
    segment_ids_a, segment_ids_b = map(tf.to_int32,
                                       [segment_ids_a, segment_ids_b])
    cross_a, cross_b = segment_cross2d(embeddings_a, embeddings_b,
                                       segment_ids_a, segment_ids_b)
    cross_a.set_shape([None, embeddings_a.get_shape()[1]])
    cross_b.set_shape([None, embeddings_b.get_shape()[1]])
    ids2d_a = tf.to_float(tf.reshape(segment_ids_a, [-1, 1]))
    ids2d_b = tf.to_float(tf.reshape(segment_ids_b, [-1, 1]))
    cross_ids_a, cross_ids_b = segment_cross2d(ids2d_a, ids2d_b, segment_ids_a,
                                               segment_ids_b)
    cross_ids_a = tf.reshape(cross_ids_a, [-1])
    del cross_ids_b
    return cross_a, cross_b, tf.to_int32(cross_ids_a)


def make_dense_3d(batch_embeddings_2d,
                  term_indices,
                  dense_shape_2d,
                  default_value=0.0):
    width = tf.shape(batch_embeddings_2d, out_type=tf.int64)[1]
    dense_shape_2d = tf.to_int64(dense_shape_2d)
    num_terms = tf.shape(term_indices)[0]
    dense_ids = tf_utils.sparse_tensor_to_dense(  # no backprop needed for values.
        tf.SparseTensor(
            indices=tf.to_int64(term_indices),
            values=tf.range(1, num_terms + 1),
            dense_shape=dense_shape_2d),
        default_value=0)
    default = tf.fill(tf.stack([1, width]), default_value)
    batch_embeddings_2d = tf.concat([default, batch_embeddings_2d], axis=0)
    return tf.gather(batch_embeddings_2d, dense_ids)


class SequenceNet(seq_utils.SequenceNet):
    def __init__(self, mode, config=None):
        super().__init__(mode, config=config)
        self._prune_invalid_ids = prune_invalid_ids(config)

    def seq_kernel(self, seq_layer_all):
        with tf_utils.device_or_none(self._model_config.gpu_device):
            config, model_config, mode = self._config, self._model_config, self._mode
            dropout_rate = eval(
                self._expt_flags.get(
                    'din_dropout', str(model_config.get(
                        'dropout', 0.0))))  #model_config.get('dropout', 0.0)
            layer_sizes = eval(
                self._expt_flags.get('seq_layer_sizes', '[64,32]'))
            last_layer_dropout = False
            last_layer_relu = eval(
                self._expt_flags.get('din_lastrelu', 'True'))
            seed = model_config.get('init_seed')
            kernel_initializer = shared_utils.weight_initializer_type(
                model_config,
                self._expt_flags.get('din_init_type', 'glorot_uniform'))
            for i, layer_size in enumerate(layer_sizes):
                seq_layer_all = tf.layers.dense(
                    seq_layer_all,
                    layer_size,
                    activation=tf.nn.relu if
                    (last_layer_relu or i < len(layer_sizes) - 1) else None,
                    name='f%d_seq' % (i + 1),
                    kernel_initializer=kernel_initializer)
                tf_utils.summary_histogram('seq_layer_%d' % i, seq_layer_all)
                if dropout_rate > 0.0 and (i < len(layer_sizes) - 1
                                           or last_layer_dropout):
                    seq_layer_all = tf.layers.dropout(
                        seq_layer_all,
                        dropout_rate,
                        training=(mode == ModeKeys.TRAIN),
                        seed=seed)
        return seq_layer_all

    def get_sparse_sequence_features(self, feature_columns, features, reuse):
        """Outputs either specialized sparse embedding namedtuples in the case
        of sequence embedding columns, or simply tf.SparseTensor in the case of
        sequence numeric (dense) columns.
        """
        config = self._config
        sparse_embeddings_or_tensors = {}
        feature_config = config.feature_config
        for fname, feature_column in feature_columns:
            fconfig = feature_config[fname]
            embedding_type = fconfig.get('embedding_type', 'combine')
            sequence_dense = isinstance(feature_column,
                                        sfc._SequenceNumericColumn)
            sequence_sparse = (embedding_type.startswith('sequence_')
                               and embedding_type.endswith('_sparse'))
            if sequence_sparse:  # special sparse_embeddings 2d value tensor.
                seq_datum = sequence_datum(
                    fname=fname, fcolumn=feature_column, fconfig=fconfig)
                sparse_embeddings_or_tensors[fname] = self.sparse_network(
                    features,
                    seq_datum,
                    scope='input_layer/sparse_%s' % fname,
                    reuse=reuse)
            elif sequence_dense:  # already a tf.SparseTensor
                feature = features[feature_column.name]
                assert isinstance(feature, tf.SparseTensor) and (
                    feature.values.dtype is tf.float32), feature
                sparse_embeddings_or_tensors[fname] = feature
        return sparse_embeddings_or_tensors

    def din_kernel(self, d_layer_all):
        config, model_config, mode = self._config, self._model_config, self._mode
        with tf_utils.device_or_none(self._model_config.gpu_device):
            dropout_rate = eval(
                self._expt_flags.get('din_dropout',
                                     str(model_config.get('dropout', 0.0))))
            layer_sizes = eval(
                self._expt_flags.get('din_layer_sizes', '[64,32,1]'))
            last_layer_dropout = False
            last_layer_relu = eval(
                self._expt_flags.get('din_lastrelu', 'False'))
            seed = model_config.get('init_seed')
            kernel_initializer = shared_utils.weight_initializer_type(
                model_config,
                self._expt_flags.get('din_init_type', 'glorot_uniform'))
            regularizer = tf.contrib.layers.l2_regularizer(
                model_config.regularizer_scale)
            for i, layer_size in enumerate(layer_sizes):
                activation_fn = tf.nn.relu if (
                    last_layer_relu or i < len(layer_sizes) - 1) else None
                if i == 0:
                    # layer_idx=None to match keras kernel/bias var names.
                    d_layer_all = network_utils.one_layer(d_layer_all,
                        layer_size, self._mode, regularizer=regularizer,
                        kernel_initializer=kernel_initializer, layer_idx=None,
                        activation_fn=activation_fn, name='f1_att')
                else:
                    d_layer_all = tf.layers.dense(
                        d_layer_all,
                        layer_size,
                        activation=activation_fn,
                        name='f%d_att' % (i + 1),
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer)
                tf_utils.summary_histogram('din_layer_%d' % i, d_layer_all)
                if dropout_rate > 0.0 and (i < len(layer_sizes) - 1
                                           or last_layer_dropout):
                    d_layer_all = tf.layers.dropout(
                        d_layer_all,
                        dropout_rate,
                        training=(mode == ModeKeys.TRAIN),
                        seed=seed)
        return d_layer_all

    def din_layer_2d(self, sequence_embeddings, extra_sequence_embeddings,
        sequence_length, attend_vector, extra_attend_vector, is_common=False):
        """attention_layer used in DIN, implemented with segment cross.

        Args:
            extra_sequence_embeddings: if not None, the same as sequence_embeddings,
                to be concatenated with embedding interactions as attention input.
            extra_attend_vector: similar to attend_vector, but additional.
            Note that the extra_ input tensors do not go through seq_kernel.
            is_common: whether the sequence_embeddings should be viewed as common
                features in a session, only available during serving.
        """
        # never time major.
        is_seq_kernel = eval(self._expt_flags.get('seq_kernel', 'False'))
        is_dot = eval(self._expt_flags.get('din_is_dot', 'False'))
        allow_empty_sequence = eval(self._expt_flags.get('allow_empty_sequence', 'True'))
        is_zero_attention = eval(
            self._expt_flags.get('is_zero_attention', 'False'))
        attention_type = self._expt_flags.get('attention_type', 'mlp')
        if is_seq_kernel:
            with tf.variable_scope('sequence_embedding', reuse=tf.AUTO_REUSE):
                sequence_embeddings = self.seq_kernel(sequence_embeddings)
            if attend_vector is not None:
                with tf.variable_scope('attend_vector', reuse=tf.AUTO_REUSE):
                    attend_vector = self.seq_kernel(attend_vector)
        config, mode = self._config, self._mode
        batch_size = tf.shape(attend_vector)[0]
        cross_a = sequence_embeddings
        cross_b = attend_vector
        if is_common:
            sequence_length = tf.tile(sequence_length, [batch_size])
            cross_a = tf.expand_dims(cross_a, 0)
            cross_b = tf.expand_dims(cross_b, 1)
            if extra_sequence_embeddings is not None:
                extra_sequence_embeddings = tf.expand_dims(extra_sequence_embeddings, 0)
            if extra_attend_vector is not None:
                extra_attend_vector = tf.expand_dims(extra_attend_vector, 1)
        sequence_width = tf.to_int32(tf_utils.reduce_max_with_zero(sequence_length))
        input_masks = tf.reshape(tf.sequence_mask(sequence_length, sequence_width), [-1])
        indices0 = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, sequence_width]),
            [-1, 1])
        indices1 = tf.reshape(
            tf.tile(tf.expand_dims(tf.range(sequence_width), 0), [batch_size, 1]),
            [-1, 1])
        term_indices = tf.boolean_mask(
            tf.concat([indices0, indices1], axis=1),
            input_masks)
        sequence_segment_ids = tf.to_int32(term_indices[:, 0])
        if attend_vector is None and extra_attend_vector is None:
            utils.debuginfo("attend_vector is None: skip attention kernel")
            assert not is_common, 'Cannot infer batch_size in this case!'
            return tf_utils.segment_sum_with_batch_size(
                sequence_embeddings, sequence_segment_ids, batch_size)

        if attend_vector is not None and not is_common:
            cross_b = tf.gather(cross_b, sequence_segment_ids)
        if extra_attend_vector is not None and not is_common:
            extra_attend_vector = tf.gather(extra_attend_vector, sequence_segment_ids)
        if attention_type == 'mlp':
            # D: embedding dim, B: batch_size, L: sequence length, LB: sum_{i=1}^B L_i.
            # (B, 1, D) or (LB, D)
            din_concat_input = [cross_b] if cross_b is not None else []
            din_concat_input.append(cross_a)    # (L, D) or (LB, D)
            if is_dot and cross_b is not None:
                if is_common:
                    x = tf.transpose(cross_b, [2, 0, 1])   # (B, 1, D) -> (D, B, 1)
                    y = tf.transpose(cross_a, [2, 0, 1])    # (1, L, D) -> (D, 1, L)
                    din_concat_input.append(tf.transpose(tf.matmul(
                        x, y), [1, 2, 0])) # (B, L, D)
                    print('din_concat_input: ', din_concat_input)
                else:
                    din_concat_input.append(cross_b * cross_a)  # ()
            if extra_sequence_embeddings is not None:
                din_concat_input.append(extra_sequence_embeddings)
            if extra_attend_vector is not None:
                din_concat_input.append(extra_attend_vector)
            if not is_common:
                din_concat_input = tf.concat(din_concat_input, axis=-1)
            d_layer_all = self.din_kernel(din_concat_input)
        elif attention_type == 'dot':
            d_layer_all = tf.reduce_sum(cross_a * cross_b, axis=-1)
        dense_shape_2d = tf.stack([batch_size, sequence_width], axis=0)
        if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
            d_layer_all = tf_utils.sparse_tensor_to_dense(
                tf.SparseTensor(
                    indices=tf.to_int64(term_indices),
                    values=tf.reshape(d_layer_all, [-1]),
                    dense_shape=tf.to_int64(dense_shape_2d)),
                default_value=-2.**32 + 1)
        else:
            d_layer_all = tf_utils.sparse_tensor_to_dense(
                tf.SparseTensor(
                    indices=tf.to_int64(term_indices),
                    values=tf.reshape(d_layer_all, [-1]),
                    dense_shape=tf.to_int64(dense_shape_2d)),
                default_value=-2.**32 + 1)
        outputs = tf.reshape(d_layer_all,
                             tf.stack([batch_size, 1, sequence_width], axis=0))
        if is_zero_attention:
            attend_0 = tf.zeros([batch_size, 1, 1])
            outputs = tf.concat([attend_0, outputs], axis=-1)
        outputs = tf.div(outputs, tf.sqrt(tf.to_float(tf.shape(sequence_embeddings)[1])))
        results = tf.nn.softmax(outputs)
        tf_utils.summary_histogram('attention_score', results)
        if is_zero_attention:
            zero_weight = results[:, :, 0]
            tf_utils.summary_histogram('attention_score_zero', zero_weight)
            results = results[:, :, 1:]
        results = tf.boolean_mask(tf.reshape(results, [-1]), input_masks)
        if is_common:
            sequence_embeddings = tf.tile(sequence_embeddings, [batch_size, 1])
        results = sequence_embeddings * tf.reshape(results, [-1, 1])
        results = tf_utils.segment_sum_with_batch_size(
            results, sequence_segment_ids, batch_size)
        if is_dot and attend_vector is not None:
            results = tf.concat([results, results * attend_vector], axis=1)
        return results

    def din_attention_layer(self,
                            inputs,
                            sequence_length,
                            attend_vector,
                            regularizer,
                            time_major=False):
        """attention_layer used in DIN."""
        is_seq_kernel = eval(self._expt_flags.get('seq_kernel', 'False'))
        is_dot = eval(self._expt_flags.get('din_is_dot', 'False'))
        is_zero_attention = eval(
            self._expt_flags.get('is_zero_attention', 'False'))
        if is_seq_kernel:
            with tf.variable_scope('sequence_embedding', reuse=tf.AUTO_REUSE):
                inputs = self.seq_kernel(inputs)
            if attend_vector is not None:
                with tf.variable_scope('attend_vector', reuse=tf.AUTO_REUSE):
                    attend_vector = self.seq_kernel(attend_vector)
        config, mode = self._config, self._mode
        if time_major:
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        attend_vector_units_length = attend_vector.get_shape().as_list()[-1]
        input_dims = tf.shape(inputs)
        batch_size = tf.shape(sequence_length)[0]
        attend_vector_tile = tf.tile(attend_vector, [1, input_dims[1]])
        attend_vector_tile = tf.reshape(
            attend_vector_tile,
            [input_dims[0], input_dims[1], attend_vector_units_length])
        if is_dot:
            d_layer_all = tf.concat(
                [attend_vector_tile, inputs, attend_vector_tile * inputs],
                axis=-1)
        else:
            d_layer_all = tf.concat([attend_vector_tile, inputs], axis=-1)
        d_layer_all = self.din_kernel(d_layer_all)
        outputs = tf.reshape(d_layer_all, [input_dims[0], 1, input_dims[1]])
        input_masks = tf.sequence_mask(sequence_length, input_dims[1])
        input_masks = tf.expand_dims(input_masks, 1)
        paddings = tf.ones_like(outputs) * (-2**32 + 1)
        outputs = tf.where(input_masks, outputs, paddings)
        if is_zero_attention:
            attend_0 = tf.zeros([batch_size, 1, 1])
            outputs = tf.concat([attend_0, outputs], axis=-1)
        outputs = tf.div(outputs,
                         tf.sqrt(tf.cast(input_dims[-1], dtype=tf.float32)))
        results = tf.cond(
            tf.not_equal(input_dims[1], 0),
            true_fn=lambda: tf.nn.softmax(outputs),
            false_fn=lambda: outputs)
        tf_utils.summary_histogram('attention_score', results)
        if is_zero_attention:
            zero_weight = results[:, :, 0]
            tf_utils.summary_histogram('attention_score_zero', zero_weight)
            results = results[:, :, 1:]
        results2 = tf.matmul(results,
                             inputs)  # per row matmul of last two dims.
        # Use as_list() to get a number intead of a tensor. Parameters in next
        # hidden layer need fixed shape to initialize.
        results2 = tf.reshape(
            results2,
            [input_dims[0], inputs.get_shape().as_list()[-1]])
        if is_dot:
            results2 = tf.concat([results2, results2 * attend_vector], axis=1)
        return results2

    def sequence_embeddings_and_ids(self, features, seq_data):
        mode = self._mode
        embeddings_and_ids = []
        assert seq_data
        for seq_datum in seq_data:
            fcolumn, fconfig = seq_datum.fcolumn, seq_datum.fconfig
            embeddings, sparse_ids, id_sequence_length = sequence_input_layer(
                features, [(seq_datum.fname, fcolumn)],
                config=self._config,
                mode=mode)[0]
            if fconfig.sequence_decorator:
                for dec in fconfig.sequence_decorator.split(';'):
                    embeddings = eval('decorator.%s' % dec)(embeddings)
            embeddings_and_ids.append((embeddings, sparse_ids,
                                       id_sequence_length))
        return embeddings_and_ids

    def concat_sequence_input_layer(self, features, seq_data):
        mode = self._mode
        nets = []
        embeddings_and_ids = self.sequence_embeddings_and_ids(
            features, seq_data)
        first_feature = next(iter(features.values()))
        if isinstance(first_feature, tf.Tensor):
            input_bs = tf.shape(first_feature)[0]
        else:
            assert isinstance(first_feature, tf.SparseTensor), first_feature
            input_bs = first_feature.dense_shape[0]
        zeros = tf.fill(
            tf.reshape(input_bs, [1]), tf.constant(0, dtype=tf.int64))
        sequence_length = zeros
        total_width = 0
        for sd, (net, sparse_ids, id_sequence_length) in zip(
                seq_data, embeddings_and_ids):
            if isinstance(sd.fcolumn, sfc._SequenceNumericColumn):
                total_width += net.get_shape()[1]
                nets.append(net)
                continue
            if self._prune_invalid_ids:
                indices = sparse_ids.indices[:, :2]
                shape = sparse_ids.dense_shape[:2]
                nets.append(make_dense_3d(net, indices, shape))
            else:
                nets.append(net)
            total_width += sd.fcolumn.dimension
            sequence_length = tf.maximum(sequence_length, id_sequence_length)
        nets_concat = tf.concat(nets, -1)
        if not self._prune_invalid_ids:
            nets_concat = make_dense_3d(nets_concat, sparse_ids.indices[:, :2],
                                        sparse_ids.dense_shape[:2])
        # This is needed for tf.layers.Dense in din_attention_layer.
        nets_concat.set_shape([None, None, total_width])
        return nets_concat, sequence_length

    def concat_sequence_2d(self, features, seq_data):
        embeddings_and_ids = self.sequence_embeddings_and_ids(
            features, seq_data)
        max_length = tf_utils.reduce_max_with_zero(
            tf.stack(
                [tf.to_int32(length) for _, _, length in embeddings_and_ids],
                axis=1),
            axis=1)
        if self._prune_invalid_ids:
            cum_length = tf.cumsum(max_length, exclusive=True)
            sum_length = cum_length[-1] + max_length[-1]
        nets = []
        widths = []
        extra_nets = []
        extra_widths = []
        for sd, (net, sparse_ids, id_sequence_length) in zip(
                seq_data, embeddings_and_ids):
            width = net.get_shape()[1]
            if sd.flag1 == 1:
                widths.append(width)
            if sd.flag2 == 1:
                extra_widths.append(width)
            if isinstance(sd.fcolumn, sfc._SequenceNumericColumn):
                if sd.flag1 == 1:
                    nets.append(net)
                if sd.flag2 == 1:
                    extra_nets.append(net)
                continue
            #width = sd.fcolumn.dimension
            #widths.append(width)
            if self._prune_invalid_ids:
                row_idx = tf.to_int32(sparse_ids.indices[:, 0])
                seq_idx = tf.to_int32(sparse_ids.indices[:, 1])
                row_base = tf.to_int32(tf.gather(cum_length, row_idx))
                batch_seq_idx = tf.reshape(
                    tf.tile(
                        tf.reshape(row_base + seq_idx, [-1, 1]), [1, width]),
                    [-1, 1])
                emb_idx = tf.reshape(
                    tf.tile(
                        tf.reshape(tf.range(width), [1, -1]),
                        [tf.size(row_idx), 1]), [-1, 1])
                indices = tf.to_int64(
                    tf.concat([batch_seq_idx, emb_idx], axis=1))
                dense_shape = tf.to_int64(tf.stack([sum_length, width]))
                if sd.flag1 == 1:
                    nets.append(
                        tf_utils.sparse_tensor_to_dense(
                            tf.SparseTensor(
                                indices=indices,
                                values=tf.reshape(net, [-1]),
                                dense_shape=dense_shape)))
                if sd.flag2 == 1:
                    extra_nets.append(
                        tf_utils.sparse_tensor_to_dense(
                            tf.SparseTensor(
                                indices=indices,
                                values=tf.reshape(net, [-1]),
                                dense_shape=dense_shape)))
            else:
                if sd.flag1 == 1:
                    nets.append(net)
                if sd.flag2 == 1:
                    extra_nets.append(net)
        net = tf.concat(nets, axis=1)
        net.set_shape([None, sum(widths)])
        extra_net = None
        if len(extra_nets) > 0:
            extra_net = tf.concat(extra_nets, axis=1)
            extra_net.set_shape([None, sum(extra_widths)])
        return net, extra_net, max_length

    def transformer_network(self,
                            features,
                            seq_data,
                            scope,
                            reuse=False,
                            bow_tensors=None):
        config, mode, model_config = self._config, self._mode, self._model_config
        assert bow_tensors is not None
        with tf.variable_scope(scope, reuse=reuse):
            first_fconfig = seq_data[0].fconfig
            regularizer = tf.contrib.layers.l2_regularizer(
                model_config.regularizer_scale)
            attend_vector = seq_utils.get_attend_vector(
                first_fconfig, bow_tensors, model_config.siamese)
            net, sequence_length = self.concat_sequence_input_layer(
                features, seq_data)
            input_dims = tf.shape(net)
            input_masks = tf.sequence_mask(sequence_length, input_dims[1])
            attention_masks = create_attention_mask_from_input_mask(
                input_masks, input_masks)
            net = transformer_model(
                net,
                attention_masks,
                hidden_size=net.get_shape().as_list()[-1],
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=32)
            net = self.din_attention_layer(net, sequence_length, attend_vector,
                                           regularizer)
            return net

    # Override base class method.
    def attention_network(self,
                          features,
                          seq_data,
                          scope,
                          reuse=False,
                          bow_tensors=None):
        config, mode, model_config = self._config, self._mode, self._model_config
        expt_flags = dict(
            t.split(':')
            for t in filter(None, config.experimental_flags.split(';')))
        reshape_3d = eval(
            expt_flags.get('attention_use_3d_embeddings', 'False'))
        assert bow_tensors is not None
        with tf.variable_scope(scope, reuse=reuse):
            first_fconfig = seq_data[0].fconfig
            attend_vector = seq_utils.get_attend_vector(
                first_fconfig, bow_tensors, model_config.siamese)
            extra_attend_vector = seq_utils.get_extra_attend_vector(
                first_fconfig, bow_tensors, model_config.siamese)
            regularizer = tf.contrib.layers.l2_regularizer(
                model_config.regularizer_scale)
            if reshape_3d:
                net, sequence_length = self.concat_sequence_input_layer(
                    features, seq_data)
                net = self.din_attention_layer(net, sequence_length,
                                               attend_vector, regularizer)
            else:
                sequence_embeddings, extra_sequence_embeddings, sequence_length = \
                    self.concat_sequence_2d(features, seq_data)
                is_common = all(sd.fname in self._common_features for sd in seq_data)
                net = self.din_layer_2d(sequence_embeddings,
                                        extra_sequence_embeddings,
                                        sequence_length,
                                        attend_vector,
                                        extra_attend_vector,
                                        is_common=is_common)
            return net

    def sparse_network(self, features, seq_datum, scope, reuse=False):
        """A pass through network to output a sparse object for the sequence.

        Args:
            seq_datum: a sequence_datum namedtuple object.
        """
        with tf.variable_scope(scope, reuse=reuse):
            tmp = self.sequence_embeddings_and_ids(features, [seq_datum])[0]
            embeddings, sparse_ids, seq_length = tmp
            return sparse_embedding(embeddings, tf.to_int32(
                sparse_ids.indices), seq_length)
