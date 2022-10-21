import tensorflow as tf
from tensorflow.estimator import ModeKeys
import sys
import os
import re
from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as sfc
src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)
import utils
from share import tf_utils
import column_decorator as decorator
import network_utils
import re
from collections import namedtuple

thismodule = sys.modules[__name__]
class sequence_datum:
    def __init__(self, fname=None, fcolumn=None, fconfig=None, flag1=None, flag2=None):
        self.fname, self.fcolumn, self.fconfig, self.flag1, self.flag2 = (
            fname, fcolumn, fconfig, flag1, flag2)


def extract_feature_groups(config):
    feature_to_concat_name = {}
    for concat_name, v in config.concat_sequence_features.items():
        for fname in v.features:
            feature_to_concat_name[fname] = (concat_name, 1, 0)
        for fname in v.extra_features:
            if fname in feature_to_concat_name:
                feature_to_concat_name[fname] = (concat_name, 1, 1)
            else:
                feature_to_concat_name[fname] = (concat_name, 0, 1)
    return feature_to_concat_name

class SequenceNet:
    def __init__(self, mode, config=None):
        self._config = config
        self._expt_flags = config and dict(kv.split(':') for kv in filter(
            None, config.experimental_flags.split(';')))
        self._model_config = config and config.model_config
        self._mode = mode
        # common features not supported for train/eval/predict.
        self._common_features = config and config.get(
            'export_mode') and config.common_features or []

    def _tile_common(self, bow_tensors):
        common_features = self._config.common_features or []
        common_fnames = [fn for fn in bow_tensors if fn in common_features]
        regular_fnames = [fn for fn in bow_tensors if fn not in common_features]
        if not (common_fnames and regular_fnames):
            return
        batch_size = tf.shape(bow_tensors[regular_fnames[0]])[0]
        common_tensors = {k: tf.tile(bow_tensors[k], [batch_size, 1]) for k in
            common_fnames}
        bow_tensors.update(common_tensors)

    def get_combined_sequence_features(self, feature_columns, features, reuse,
        bow_tensors):
        mode, config = self._mode, self._config
        feature_to_concat_name = extract_feature_groups(config)
        sequence_concat_data = {}
        feature_config = config.feature_config
        model_config = config.model_config
        num_codistill = model_config.num_codistill
        for fname, feature_column in utils.siamese_sorted(feature_columns):
            fconfig = feature_config[fname]
            embedding_type = fconfig.get('embedding_type', 'combine')
            if not embedding_type.startswith('sequence_'):
                continue
            seq_type = embedding_type.split('_')[-1]
            assert seq_type in ['cnn', 'rnn', 'attention','transformer']
            group_value = feature_to_concat_name.get(fname, (fname, 1, 0))
            utils.debuginfo("group_value: {}".format(group_value))
            sequence_concat_data.setdefault((group_value[0], seq_type), []).append(
                sequence_datum(
                    fname=fname, fcolumn=feature_column, fconfig=fconfig, flag1=group_value[1], flag2=group_value[2]))
        if not sequence_concat_data:
            return {}
        sequence_tensors = {}
        for seq_index, ((group_name, seq_type), seq_data) in enumerate(
                sorted(sequence_concat_data.items())):
            assert (len(set(t.fconfig.sequence_output_decorator for t in
                seq_data)) == 1), (group_name, seq_type, seq_data)
            first_fconfig = seq_data[0].fconfig
            if model_config.siamese and any(n.endswith('_b') for n in bow_tensors):
                bow_tensors = {re.sub('_b$', '_a', k): v for k, v in bow_tensors.items()}
            self._tile_common(bow_tensors)
            column_tensor = getattr(self, '%s_network' % seq_type)( # attention_network
                features,
                seq_data,
                scope='input_layer/%s_f%d' % (seq_type, seq_index),
                reuse=reuse,
                bow_tensors=bow_tensors)
            if first_fconfig.sequence_output_decorator:
                for dec in first_fconfig.sequence_output_decorator.split(';'):
                    column_tensor = eval('decorator.%s' % dec)(column_tensor)
            if num_codistill > 1 and not first_fconfig.get('indicator', True):
                column_tensor = tf.split(
                    column_tensor, num_or_size_splits=num_codistill, axis=1)
            # tf_utils.summary_histogram("sequence_input_layer/"+group_name, column_tensor)
            sequence_tensors[group_name] = column_tensor
        return sequence_tensors

    def rnn_network(self, features, seq_data, scope, reuse=False,
        bow_tensors=None):
        mode, config, expt_flags = self._mode, self._config, self._expt_flags
        with tf.variable_scope(scope, reuse=reuse):
            regularizer = tf.contrib.layers.l2_regularizer(
                config.model_config.regularizer_scale)
            net, _ = concat_sequence_input_layer(features, seq_data, mode)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.embedding_dim)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, net, dtype=tf.float32)
            return states[1]

    def cnn_network(self,
                    features,
                    seq_data,
                    scope,
                    reuse=False,
                    bow_tensors=None,
                    output_dense=False):
        mode, config, expt_flags = self._mode, self._config, self._expt_flags
        with tf.variable_scope(scope, reuse=reuse):
            # Convolutional Layer #1
            regularizer = tf.contrib.layers.l2_regularizer(
                config.model_config.regularizer_scale)
            net, _ = concat_sequence_input_layer(features, seq_data, mode)

            net = tf.layers.dropout(
                inputs=net, rate=0.0, training=(mode == ModeKeys.TRAIN))
            conv_blocks = []
            for sz in map(int,
                          expt_flags.get('cnn_kernel_sizes', '1,2,3').split(',')):
                conv = tf.layers.conv1d(
                    inputs=net,
                    filters=config.embedding_dim,
                    kernel_size=sz,
                    kernel_regularizer=regularizer)
                if eval(expt_flags.get('cnn_batch_normalization', '1')):
                    conv = tf.layers.batch_normalization(
                        inputs=conv, training=(mode == ModeKeys.TRAIN))
                conv = tf.nn.relu(conv)
                conv = tf.reduce_max(conv, reduction_indices=[1])
                conv = tf.layers.dropout(
                    inputs=conv, rate=0.2, training=(mode == ModeKeys.TRAIN))
                conv_blocks.append(conv)
            cnn_output = tf.concat(conv_blocks, axis=1)
            if output_dense:
                dense_net = tf.layers.dense(
                    cnn_output,
                    units=config.embedding_dim,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                )
                dense_net = tf.layers.batch_normalization(
                    inputs=dense_net, training=(mode == ModeKeys.TRAIN))
                dense_net = tf.nn.relu(dense_net)
                cnn_output = tf.layers.dropout(
                    inputs=dense_net, rate=0.1, training=(mode == ModeKeys.TRAIN))
            return cnn_output

    def attention_network(self,
                          features,
                          seq_data,
                          scope,
                          reuse=False,
                          bow_tensors=None):
        mode, config = self._mode, self._config
        assert bow_tensors
        model_config = config.model_config
        with tf.variable_scope(scope, reuse=reuse):
            first_fconfig = seq_data[0].fconfig
            attend_vector = get_attend_vector(
                first_fconfig, bow_tensors, model_config.siamese)
            net, sequence_length = concat_sequence_input_layer(
                features, seq_data, mode)
            regularizer = tf.contrib.layers.l2_regularizer(
                model_config.regularizer_scale)
            net = din_attention_layer(net, sequence_length, attend_vector,
                                      regularizer, mode, config)
            return net


def concat_sequence_input_layer(features, seq_data, mode):
    nets = []
    for seq_datum in seq_data:
        fcolumn, fconfig = seq_datum.fcolumn, seq_datum.fconfig
        net, sequence_length = sfc.sequence_input_layer(features, [fcolumn])
        if fconfig.sequence_decorator:
            for dec in fconfig.sequence_decorator.split(';'):
                net = eval('decorator.%s' % dec)(net)
        nets.append(net)
    return tf.concat(nets, axis=-1), sequence_length


def get_attend_vector(fconfig, fname_to_tensor, siamese=True):
    attend_names = fconfig.get('attend_name')
    if attend_names is None or len(attend_names) == 0:
        return None
    if siamese:
        attend_names = [re.sub('_b$', '_a', name) for name in attend_names]
    utils.debuginfo(attend_names)
    if attend_names[0] == 'side_a':
        attend_vectors = [fname_to_tensor[fname]
            for fname in utils.siamese_sorted(fname_to_tensor) if fname !='price1aft_a']
    else:
        attend_vectors = [fname_to_tensor[fname] for fname in attend_names]
        #[fname_to_tensor[fname] for fname in utils.siamese_sorted(attend_names)]
    return tf.concat(attend_vectors, axis=1)


def get_extra_attend_vector(fconfig, fname_to_tensor, siamese=True):
    attend_names = fconfig.get('extra_attend_name')
    if attend_names is None:
        return None
    if siamese:
        attend_names = [re.sub('_b$', '_a', name) for name in attend_names]
    utils.debuginfo(attend_names)
    attend_vectors = [fname_to_tensor[fname] for fname in attend_names]
    return tf.concat(attend_vectors, axis=1)


def din_attention_layer(inputs,
                        sequence_length,
                        attend_vector,
                        regularizer,
                        mode,
                        config,
                        time_major=False):
    """attention_layer used in DIN."""
    model_config = config.model_config
    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
    attend_vector_units_length = attend_vector.get_shape().as_list()[-1]
    input_dims = tf.shape(inputs)
    attend_vector_tile = tf.tile(attend_vector, [1, input_dims[1]])
    #attend_vector_tile-inputs,attend_vector_tile*inputs
    attend_vector_tile = tf.reshape(
        attend_vector_tile,
        [input_dims[0], input_dims[1], attend_vector_units_length])
    d_layer_all = tf.concat([attend_vector_tile, inputs], axis=-1)
    dropout_rate = model_config.get('dropout', 0.0)
    expt_flags = dict(
        kv.split(':')
        for kv in filter(None, config.experimental_flags.split(';')))
    layer_sizes = eval(expt_flags.get('din_layer_sizes', '[64,32,1]'))
    last_layer_dropout = False
    last_layer_relu = True
    seed = model_config.get('init_seed')
    kernel_initializer =tf.glorot_uniform_initializer(seed=seed)   #shared_utils.weight_initializer(model_config)  #tf.glorot_uniform_initializer(seed=seed)

    for i, layer_size in enumerate(layer_sizes):
        d_layer_all = tf.layers.dense(
            d_layer_all,
            layer_size,
            activation=tf.nn.relu if
             (last_layer_relu or i < len(layer_sizes) - 1) else None,
            name='f%d_att' % (i + 1),
            kernel_initializer=kernel_initializer)
        if dropout_rate > 0.0 and (i < len(layer_sizes) - 1
                                   or last_layer_dropout):
            d_layer_all = tf.layers.dropout(
                d_layer_all,
                dropout_rate,
                training=(mode == ModeKeys.TRAIN),
                seed=seed)
    # outputs = network_utils.get_all_layers(din_all,[64,32,1],
    # regularizer, model_config, mode, network_utils.prelu)

    outputs = tf.reshape(d_layer_all, [input_dims[0], 1, input_dims[1]])
    input_masks = tf.sequence_mask(sequence_length, input_dims[1])
    input_masks = tf.expand_dims(input_masks, 1)
    paddings = tf.ones_like(outputs) * (-2**32 + 1)
    outputs = tf.where(input_masks, outputs, paddings)
    outputs = tf.div(outputs, tf.sqrt(
        tf.cast(input_dims[-1], dtype=tf.float32)))
    # outputs = outputs / (inputs.get_shape().as_list()[-1] ** 0.5)
    results = tf.cond(
        tf.not_equal(input_dims[1], 0),
        true_fn=lambda: tf.nn.softmax(outputs),
        false_fn=lambda: outputs)
    results = tf.matmul(results, inputs)
    # Use as_list() to get a number intead of a tensor. Parameters in next
    # hidden layer need fixed shape to initialize.
    results = tf.reshape(
        results,
        [input_dims[0], inputs.get_shape().as_list()[-1]])
    return results
