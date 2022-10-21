from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.estimator import ModeKeys
import sys
import os
import re
import functools
from tensorflow.python.feature_column import feature_column as fc
src = os.path.abspath(__file__)
for _ in 'model_lib/ranking/src/models'.split('/'):
    src = os.path.dirname(src)
    if src not in sys.path:
        sys.path.insert(0, src)
import column_decorator as decorator
import experimental_utils as expt_utils
import utils
from share import tf_utils, shared_utils
from share import light_weight_utils as lw_utils
import simple_input_columns
from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as sfc
from tf_revised_lib import tf_optimizer


# Refactored from tensorflow/python/feature_column/feature_column.py
def _internal_output_tensors(features,
                             feature_columns,
                             scope='input_layer',
                             trainable=True,
                             reuse=True):
    if feature_columns:
        feature_columns = list(
            zip([t[0] for t in feature_columns],
                fc._normalize_feature_columns(
                    [t[1] for t in feature_columns])))
    weight_collections = [
        tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES
    ]
    with tf.variable_scope(scope, values=features.values(), reuse=reuse):
        builder = fc._LazyBuilder(features)
        output_tensors = []
        ordered_columns = []
        for fname, column in utils.siamese_sorted(feature_columns):
            ordered_columns.append(column)
            with tf.variable_scope(
                    column._var_scope_name, reuse=tf.AUTO_REUSE):
                if isinstance(column, sfc._SequenceNumericColumn):
                    # avoid _get_sequence_dense_tensor to keep SparseTensor.
                    output_tensors.append((fname, builder.get(column)))
                else:
                    assert isinstance(column, fc._DenseColumn), (fname, column)
                    tensor = column._get_dense_tensor(
                        builder,
                        weight_collections=weight_collections,
                        trainable=trainable)
                    num_elements = column._variable_shape.num_elements(
                    )  # py var
                    batch_size = tf.shape(tensor)[0]
                    output_tensors.append(
                        (fname,tf.reshape(tensor, shape=(batch_size, num_elements))))

        fc._verify_static_batch_size_equality([t[1] for t in output_tensors],
                                              ordered_columns)
        return output_tensors


def output_tensors(*args, **kwargs):
    return tf.make_template(
        'input_layer', _internal_output_tensors,
        create_scope_now_=True)(*args, **kwargs)


def _simple_feature_columns(fconfigs, num_codistill):
    return not any(
        fc.get('embedding_type', 'combine').startswith('sequence_')
        for fc in fconfigs) and all(
            set([
                fc.get('decorator', 'identity'),
                fc.get('sequence_decorator', 'identity')
            ]) == set(['identity']) for fc in fconfigs) and num_codistill == 1


def process_feature_columns(features,
                            feature_columns,
                            config=None,
                            mode=None,
                            reuse=False,
                            new_seq_utils=True,
                            sparse_output=False):
    """Apply some post-processing steps to each feature columns.

    Post-processings are of the following 3 varieties:
    1. sequence model sub-network: (CNN, RNN)
    2. feature_decorator: zero, stop_gradient, fixed width sub-network etc
    3. codistillation: splitting each embedding column into n pieces

    Returns:
        If no post-processing, a simple dictionary; else a list of dictionaries.
    """
    if not feature_columns:
        return [{}]
    feature_config = config.feature_config
    seq_index = 0
    fconfigs = list(feature_config.values())
    num_codistill = config.model_config.num_codistill
    # TODO(jyj): get rid of below now that reused scope can be specified.

    if config.model_config.siamese:  # normalize feature name to _a side, since
        # in Siamese mode process_feature_columns takes only one side at a time.
        feature_columns2 = {
            re.sub('_b$', '_a', fname): fcol
            for fname, fcol in feature_columns
        }
        assert len(feature_columns2) == len(feature_columns), (
            'Duplicate '
            'siamese keys in %s' % ([k for k, v in feature_columns]))
        feature_columns = utils.siamese_sorted(list(feature_columns2.items()))
    if _simple_feature_columns(fconfigs, num_codistill):
        fnames_tensors = _internal_output_tensors(
            features, feature_columns, scope='input_layer', reuse=reuse)
        return [dict(fnames_tensors)]

    fixed_width_tensors = {}
    # first go through fixed_width columns.
    for fname, feature_column in feature_columns:
        fconfig = feature_config[fname]
        fstat = config['feature_stats'][fname]
        if fconfig.get('embedding_type', 'combine').startswith('sequence') or (
                fstat['feature_column_type'] == 'variable_length'
                and fstat['data_type'] == 'float'):
            continue

        _, column_tensor = _internal_output_tensors(
            features, [(fname, feature_column)],
            scope='input_layer',
            reuse=reuse)[0]
        # column_tensor = fc.InputLayer([feature_column])(features)
        # TODO(jyj): remove this block.
        if isinstance(column_tensor, tf.SparseTensor):
            values = column_tensor.values
            for dec in fconfig.sequence_decorator.split(';'):
                values = eval('decorator.' + dec)(values)
            column_tensor = tf.SparseTensor(column_tensor.indices, values,
                                            column_tensor.dense_shape)
        for dec in fconfig.decorator.split(';'):  # includes embedding_dropout
            column_tensor = eval('decorator.%s' % dec)(column_tensor)
        if num_codistill > 1 and not fconfig.get(
                'indicator') and fstat['feature_type'] == 'sparse':
            column_tensor = tf.split(
                column_tensor, num_or_size_splits=num_codistill, axis=1)
        tf_utils.summary_histogram("input_layer/" + fname, column_tensor)
        fixed_width_tensors[fname] = column_tensor
    if new_seq_utils:
        from models import new_sequence_utils as seq_utils
    else:
        from models import sequence_utils as seq_utils
    seq_net = seq_utils.SequenceNet(mode, config)
    if sparse_output:
        assert new_seq_utils
        sparse_embeddings_or_tensors = seq_net.get_sparse_sequence_features(
            feature_columns, features, reuse)
        tensors = {**fixed_width_tensors, **sparse_embeddings_or_tensors}
    else:  # represents sequence embedding features as 3d tensors.
        combined_sequence_tensors = seq_net.get_combined_sequence_features(
            feature_columns, features, reuse, fixed_width_tensors)
        tensors = {**fixed_width_tensors, **combined_sequence_tensors}

    ret = [{
        k: t if not isinstance(t, list) else t[i]
        for k, t in utils.siamese_sorted(tensors.items())
    } for i in range(num_codistill)]
    assert len(set(tuple(sorted(v.keys())) for v in ret)) == 1
    return ret


def prelu(_x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable(
            "prelu",
            shape=_x.get_shape()[-1],
            dtype=_x.dtype,
            initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def batch_attention_indices(query_seg_sizes, item_seg_sizes):
    tiled = tf.tile(
        tf.reshape(item_seg_sizes, [-1, 1]),
        [1, tf.reduce_max(query_seg_sizes)])
    mask = tf.sequence_mask(query_seg_sizes, tf.reduce_max(query_seg_sizes))
    lengths = tf.boolean_mask(tiled, mask)  # 1d
    tiled2 = tf.tile(
        tf.reshape(tf.range(tf.reduce_sum(query_seg_sizes)), [-1, 1]),
        [1, tf.reduce_max(item_seg_sizes)])
    mask2 = tf.sequence_mask(lengths, tf.reduce_max(item_seg_sizes))
    row_indices = tf.boolean_mask(tiled2, mask2)
    range2 = tf.tile(
        tf.reshape(tf.range(tf.reduce_max(item_seg_sizes)), [1, -1]),
        [tf.reduce_sum(query_seg_sizes), 1])
    col_indices = tf.boolean_mask(range2, mask2)
    return tf.stack([row_indices, col_indices], axis=1)


def segment_sizes(segment_ids, num_segments):
    return tf.sparse_segment_sum(
        tf.ones_like(segment_ids),
        tf.range(tf.size(segment_ids)),
        segment_ids,
        num_segments=num_segments)


class NetworkLibBase:
    def __init__(self, regularizer, config, mode, aux_data=None):
        self._regularizer = regularizer
        self._config = config
        self._model_config = config.model_config
        self._expt_flags = dict(
            kv.split(':')
            for kv in filter(None, config.experimental_flags.split(';')))
        self._mode = mode
        self._aux_data = aux_data


def sliced_matmul(nets, kernel):
    if len(nets) == 1:
        return tf.matmul(nets[0], kernel)
    widths = [int(net.get_shape()[-1]) for net in nets]
    kernels = tf.split(kernel, axis=0, num_or_size_splits=widths)
    with tf.variable_scope('sliced_matmul'):
        ret = []
        for i, (k, n) in enumerate(zip(kernels, nets)):
            # Below hacks ensure kernel is not saved, but initialized.
            kernel = tf.get_variable('kernel_%d' % i, shape=[widths[i],
                int(k.get_shape()[-1])], dtype=tf.float32,
                initializer=tf.zeros_initializer(), collections=[])
            tf.add_to_collection(
                tf.GraphKeys.TABLE_INITIALIZERS, kernel.assign(k))
            ret.append(tf.matmul(n, kernel))
        return sum(ret)


def one_layer(net, units, mode, regularizer=None, kernel_initializer=None,
    layer_idx=None, activation_fn=None, name=None, normalize_kernel=False,
    batch_normalize=False, group_normalize=False, freeze_layer=False,
    dropout_rate=0.0):
    """Compute one layer of MLP, similar to tf.keras.dense.
    Args:
        net: either a dense input tensor or a list of dense tensors. In the
        latter case we split the kernel along axis 0 and multiple the input
        separately before adding them at the end, to save some flops.
    """
    # To be consistent with keras.Dense naming convention.
    suffix = '' if layer_idx is None else '_{}'.format(layer_idx)
    if isinstance(net, tf.Tensor):
        tf_utils.summary_histogram('layer%s' % suffix, net)
        net = [net]
    input_dim = sum(t.get_shape()[-1] for t in net)
    if freeze_layer:
        regularizer = None
    with lw_utils.ctx_or_none(tf.variable_scope, name, name is not None):
        kernel = tf.get_variable(
            'kernel%s' % suffix, shape=[input_dim, units],
            dtype=tf.float32, initializer=kernel_initializer,
            regularizer=regularizer)
        if normalize_kernel:
            kernel = tf.nn.l2_normalize(kernel, axis=1)
        bias = tf.get_variable(
            'bias%s' % suffix,
            shape=[units],
            initializer=tf.zeros_initializer(),
            regularizer=regularizer)
        if freeze_layer:
            kernel = tf.stop_gradient(kernel)
            bias = tf.stop_gradient(bias)
        net = tf.add(sliced_matmul(net, kernel), bias)
        if activation_fn:
            if group_normalize:
                net = tf.contrib.layers.group_norm(inputs=net)
            if batch_normalize:
                net = tf.layers.batch_normalization(
                    inputs=net, training=(mode == ModeKeys.TRAIN))
            net = activation_fn(net)
            if dropout_rate > 0.0:
                net = tf.layers.dropout(
                    net, dropout_rate, training=(mode == ModeKeys.TRAIN))
    tf_utils.summary_histogram('kernel%s' % suffix, kernel)
    tf_utils.summary_histogram('bias%s' % suffix, bias)
    return net


class NetworkLib(NetworkLibBase):
    def one_layer(self, net, units, layer_idx, activation_fn=None,
        kernel_initializer=None):
        model_config, mode = self._model_config, self._mode
        normalize_kernel = model_config.normalize_kernel and (units > 1 or not
            eval(str(self._expt_flags.get('skip_trivial_kernel_normalization'))))
        batch_normalization = model_config.get('batch_normalization', False)
        freezed_layers = model_config.freezed_layers
        freeze_layer = freezed_layers and layer_idx in freezed_layers
        if kernel_initializer is None:
            kernel_initializer = shared_utils.weight_initializer(model_config)
        return one_layer(net, units, self._mode, regularizer=self._regularizer,
            kernel_initializer=kernel_initializer, layer_idx=layer_idx,
            activation_fn=activation_fn, normalize_kernel=normalize_kernel,
            freeze_layer=freeze_layer, group_normalize=False,
            dropout_rate=getattr(model_config, 'dropout', 0.0),
            batch_normalize=batch_normalization)

    def _maybe_add_layer(self, net, units, layer_idx):
        if net.shape[1] != units:
            return self.one_layer(net, units, layer_idx, activation_fn=None)
        return net

    def get_all_layers(self,
                       net,
                       layer_sizes,
                       activation_fn=tf.nn.relu,
                       resnet_lookbacks=[]):
        def _index_exists(list_obj, idx):
            return idx < len(list_obj) and idx >= -len(list_obj)

        model_config, mode = self._model_config, self._mode
        all_layers = [net]
        # if getattr(model_config, 'batch_normalization', False):
        #     net = tf.layers.batch_normalization(
        #         inputs=net, training=(mode == ModeKeys.TRAIN))
        for i, units in enumerate(layer_sizes):
            activator = (i < len(layer_sizes) - 1) and activation_fn
            net = self.one_layer(net, units, i, activation_fn=activator)
            avg = True
            valid_indices = [
                j for j in resnet_lookbacks if _index_exists(all_layers, j)
            ]
            if valid_indices:  # use large negative adhoc layer idx.
                net += sum(
                    self._maybe_add_layer(all_layers[j], units, -i * 1000 + j)
                    for j in valid_indices) / (len(valid_indices)
                                               if avg else 1)
                if avg:
                    net /= 2
            all_layers.append(net)
        tf_utils.summary_histogram('layer_%d' % len(layer_sizes), net)
        return all_layers

    def _feed_forward_base(self, net, layer_sizes, **kwargs):
        """Compute the output of a feed forward layer from the input net.

        Args:
            net: batch x input_layer tensor.
            regularizer: regularizer layer.
            model_config: hyperparameter attrdict object.
            mode: TRAIN, EVAL, or PREDICT.
        Return:
            net: batch x last layer tensor.
        """
        model_config, layer_sizes = self._model_config, list(layer_sizes)
        utils.debuginfo('layer_sizes = %s' % layer_sizes)
        update_aux = kwargs.pop('update_aux', True)
        all_layers = self.get_all_layers(net, layer_sizes, tf.nn.relu,
                                         **kwargs)
        if self._aux_data is not None and update_aux:
            self._aux_data['all_layers'] = all_layers
        if model_config.use_linear_links:
            ret = expt_utils.linear_combine(all_layers, self._regularizer)
            tf_utils.summary_histogram('linear_link_final_logit', ret)
            return ret
        else:
            return all_layers[-1]

    def feed_forward(self, net):
        return self._feed_forward_base(
            net,
            list(self._model_config.hidden_units) + [1])

    def feed_forward_multihead(self, net):
        ret = []
        for i in range(12):
            with tf.variable_scope('head_%d' % i):
                ret.append(self.feed_forward(net))
        return tf.add_n(ret) / len(ret)

    def feed_forward_resnet(self, net):
        return self._feed_forward_base(
            net,
            list(self._model_config.hidden_units) + [1],
            resnet_lookbacks=[-1])

    def feed_forward_with_reconstruction(self, net):
        ret = self.feed_forward(net)
        reconstr_config = self._model_config.reconstruction_config
        penult_layer = self._aux_data['all_layers'][-2]
        hidden_units = list(reconstr_config.hidden_units) + [net.get_shape()[1]]
        with tf.variable_scope('reconstruction_layer'):
            self._aux_data['reconstruction_input'] = self._feed_forward_base(
                penult_layer, hidden_units, update_aux=False)
        return ret

    def feed_forward_embed(self, net):
        return tf.nn.relu(
            self._feed_forward_base(net, self._model_config.hidden_units))

    def feed_forward_embed_no_relu(self, net):
        return self._feed_forward_base(net, self._model_config.hidden_units)

    def _deep_dot_base(self, item, query):
        output_dict = self._aux_data.setdefault('output_dict', {})
        output_dict.update({'item_embedding': item, 'query_embedding': query})
        ret = tf.reduce_sum(tf.multiply(item, query), axis=1, keepdims=True)
        self._aux_data['all_layers'] = [ret]
        return ret

    def feed_forward_attention(self, net):
        model_config = self._model_config
        width = net.get_shape()[1]
        if model_config.combine_query_first and model_config.add_term_positions:
            embedding_dim = (width - 3) // 2
            splits = tf.stack([embedding_dim, embedding_dim + 3], axis=0)
            item, query = tf.split(net, num_or_size_splits=splits, axis=1)
        else:
            item, query = tf.split(net, num_or_size_splits=2, axis=1)
            embedding_dim = width // 2
        layer_sizes = self._model_config.hidden_units
        aux = self._aux_data
        if len(layer_sizes) > 1:
            tf.logging.warning(
                'Attention uses > 1 layers: %s!' % str(layer_sizes))
        with tf.variable_scope('item'):
            item = self._feed_forward_base(item, layer_sizes)
        with tf.variable_scope('query'):
            query = self._feed_forward_base(query, layer_sizes)
        # total number of query tokens in the batch.
        item_seg_sizes = segment_sizes(aux['item_seg_id'], aux['num_segments'])
        query_seg_sizes = segment_sizes(aux['query_seg_id'],
                                        aux['num_segments'])
        # cross_seg_sizes = segment_sizes(aux['cross_seg_id'], aux['num_segments'])
        # query_seg_sizes = cross_seg_sizes / item_seg_sizes
        total_segments = tf.reduce_sum(query_seg_sizes)
        batch_width = tf.reduce_max(item_seg_sizes)
        attention_scores = tf.reshape(self._deep_dot_base(item, query), [-1])
        attention_scores /= tf.sqrt(tf.to_float(embedding_dim))
        dense_shape = tf.stack([total_segments, batch_width], axis=0)
        indices = batch_attention_indices(query_seg_sizes, item_seg_sizes)
        attention_probs = tf.sparse_softmax(
            tf.SparseTensor(
                tf.to_int64(indices), attention_scores,
                tf.to_int64(dense_shape)))
        return tf.reshape(attention_probs.values, [-1, 1]) * item

    def deep_dot(self, net):
        item, query = tf.split(net, num_or_size_splits=2, axis=1)
        return self._deep_dot_base(item, query)

    def cosine_similarity(self, net):
        item, query = tf.split(net, num_or_size_splits=2, axis=1)
        item = tf.nn.l2_normalize(item, axis=1)
        query = tf.nn.l2_normalize(query, axis=1)
        return self._deep_dot_base(item, query)

    def shifted_deep_dot(self, net):
        net = self.deep_dot(net)
        bias = tf.get_variable(
            'bias_-1',
            shape=[1],
            initializer=tf.zeros_initializer(),
            regularizer=self._regularizer)
        if -1 in (self._model_config.freezed_layers or []):
            bias = tf.stop_gradient(bias)
        tf_utils.summary_histogram('bias_-1', bias)
        net += bias
        return net

    def n_way_deep_dot(self, num_split=2):
        def f(net):
            tmp = tf.split(net, num_or_size_splits=num_split, axis=1)
            ret = tmp[0]
            for t in tmp[1:]:
                ret = tf.multiply(ret, t)
            ret = tf.reduce_sum(ret, axis=1, keepdims=True)
            self._aux_data['all_layers'] = [ret]
            return ret

        return f

    def embed_item_query(self, input_dict, labels, weights=None):
        """Embed item and query separately into 2 vectors of the same dimension.

        Args:
            input_dict: a dictionary of batched dense tensors.
            labels: orig. label tensor, to be expanded for batch_negatives, etc.

        Returns:
            net: concatenation of item and query embeddings along axis 1.
        """
        assert 'item_a' in input_dict, ('Only Siamese or pointwise model '
            'supported! But found {}'.format(input_dict))
        assert 'query' in input_dict
        config, model_config, mode = self._config, self._model_config, self._mode
        assert model_config.item_query_separate_columns
        with tf.variable_scope('item'):
            item_layers = self.get_all_layers(input_dict['item_a'],
                                              model_config.item_layer_sizes,
                                              tf.nn.relu)
        with tf.variable_scope('query'):
            query_layers = self.get_all_layers(input_dict['query'],
                                               model_config.query_layer_sizes,
                                               tf.nn.relu)
        training_weights, logit_fn = weights, None
        if mode == ModeKeys.PREDICT and config.common_features:
            assert set(k for k in config.feature_config if not
                k.endswith(('_a', '_b'))) == set(config.common_features)
            dynamic_bs = tf.shape(item_layers[-1])[0]
            query_layers = [tf.tile(ql, [dynamic_bs / tf.shape(ql)[0], 1])
                for ql in query_layers]
        query, item = query_layers[-1], item_layers[-1]
        if 'concat_all_layers' in config.experimental_flags:
            # Make sure all layers have the same dimension b/t query and item.
            query, item = tf.concat(query_layers, 1), tf.concat(item_layers, 1)
        if mode != ModeKeys.PREDICT and model_config.add_batch_negatives:
            tmp = expt_utils.add_batch_negatives(query, item, labels, config,
                                                 mode)
            net, labels, training_weights, logit_fn = tmp
            if isinstance(weights, tf.Tensor):
                training_weights = tf.concat(
                    [weights, training_weights[tf.shape(weights)[0]:, :]],
                    axis=0)
        else:
            net = tf.concat([item_layers[-1], query_layers[-1]], 1)
        self._aux_data.update({
            'query_layers': query_layers,
            'item_layers': item_layers
        })
        return net, labels, training_weights, logit_fn


def maybe_swap_embedding_ab(input_dict, labels, config, mode):
    model_config = config.model_config
    if model_config.siamese or mode != ModeKeys.TRAIN or (
            not model_config.random_flip_ab):
        return input_dict, labels
    random_u = tf.random_uniform([tf.shape(labels)[0]], seed=config.shuffle_seed)
    random_bit = tf.greater(random_u, 0.5)
    neutral = float(int(model_config.train_loss_fn != 'mean_squared_error'))
    labels, _ = tf_utils.random_flip_ab(random_bit, labels, neutral - labels)
    # query was already double-stacked, so swapping acts trivially. Doing this
    # for all values also support more general input_dict than query/item type.
    input_dict = {
        k: tf.concat(
            tf_utils.random_flip_ab(
                random_bit, *tf.split(v, num_or_size_splits=2, axis=1)),
            axis=1)
        for k, v in input_dict.items()
    }
    return input_dict, labels


def maybe_flip_ab(train_data, config, mode, aux_data=None):
    if simple_pass(mode, config):
        return
    labels, logits = train_data['labels'], train_data['logits']
    model_config = config.model_config
    # assert model_config.siamese   # now covered by maybe_flip_embedding_ab
    random_u = tf.random_uniform([tf.shape(labels)[0]], seed=config.shuffle_seed)
    random_bit = tf.greater(random_u, 0.5)
    neutral = float(int(model_config.train_loss_fn != 'mean_squared_error'))
    labels, _ = tf_utils.random_flip_ab(random_bit, labels, neutral - labels)
    logits, _ = tf_utils.random_flip_ab(random_bit, logits, -logits)
    if aux_data and model_config.siamese and model_config.softmax_loss:
        batch_neg_columns = config.batch_size - 1
        if model_config.batch_negative_columns is not None:
            batch_neg_columns = model_config.batch_negative_columns
        nets = tf.split(
            aux_data['net'], num_or_size_splits=batch_neg_columns + 2, axis=0)
        net_a = tf.tile(nets[0], [batch_neg_columns + 1, 1])
        net_b = tf.concat(nets[1:], axis=0)
        net_a, net_b = tf_utils.random_flip_ab(random_bit, net_a, net_b)
        aux_data['net'] = tf.concat([net_a, net_b], axis=0)
    train_data.update({'labels': labels, 'logits': logits})


def maybe_apply_regularization(loss, aux_data, model_config, mode):
    if mode == ModeKeys.TRAIN:
        if 'regularizer' in aux_data and (model_config.regularizer_scale > 0.0
            and model_config.hidden_units):
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(
                aux_data['regularizer'], reg_variables)
            loss += reg_term
        if 'reconstruction_input' in aux_data:
            aux_data['reconstruction_loss'] = tf.nn.l2_loss(
                aux_data['reconstruction_input'] - aux_data['net'])
            tf_utils.summary_scalar('reconstruction_loss', aux_data['reconstruction_loss'])
            loss += model_config.reconstruction_config.loss_weight * aux_data[
                'reconstruction_loss']
    return loss


def get_optimizer(config):
    embed_var_list = [
        var for var in tf.trainable_variables() if 'embed' in var.name
    ]
    #for var in embed_var_list:
    #    tf_utils.summary_histogram(var.name, var)
    model_config = config.model_config
    learning_rate = model_config.learning_rate
    multiplier_schedule = model_config.learning_rate_multiplier_schedule
    expt_flags = dict(
        kv.split(':')
        for kv in filter(None, config.experimental_flags.split(';')))

    if multiplier_schedule:
        learning_rate *= tf_utils._interpolate(eval(multiplier_schedule))

    def capitalize_first_letter(s):
        return s[0].upper() + s[1:]  # RMSProp -> RMSProp, adam -> Adam.

    optimizer_name = capitalize_first_letter(model_config.optimizer)
    assert optimizer_name in [
        'Adagrad', 'Adam', 'Adadelta', 'RMSProp', 'LazyAdamW', 'LazyAdam',
        'MaskedAdam', 'AdamNoam','CustomLazyAdamW'
    ] or (optimizer_name.startswith('FastAdagrad'))
    train_lib = tf.train
    optimizer_kwargs = {}
    embedding_clip = eval(expt_flags.get('embedding_clip', '0.01'))
    if optimizer_name == 'LazyAdamW':
        LazyAdamWOptimizer = tf_optimizer.extend_with_decoupled_weight_decay(
            tf_optimizer.LazyAdamOptimizer)
        optimizer = LazyAdamWOptimizer(
            weight_decay=0.001 * learning_rate,
            embedding_clip=embedding_clip,
            learning_rate=learning_rate,
            epsilon=0.1,
            exclude_from_weight_decay=[
                'LayerNorm', 'layer_norm', 'bias', 'BatchNorm',
                'batch_norm'
            ])
        return optimizer
    elif optimizer_name == 'CustomLazyAdamW':
        from tf_revised_lib import CustomLazyAdam
        optimizer = CustomLazyAdam.CustomLazyAdamOptimizer(
            weight_decay=0.001,
            learning_rate=learning_rate,
            exclude_from_weight_decay=[
                'LayerNorm', 'layer_norm', 'bias', 'BatchNorm',
                'batch_norm','attention'
            ])
        return optimizer
    elif optimizer_name in ['LazyAdam', 'MaskedAdam']:
        train_lib = tf_optimizer
        optimizer_kwargs.update({'embedding_clip': embedding_clip})
        optimizer_kwargs.update({'epsilon': 0.1})
    elif optimizer_name == 'AdamNoam':
        step = tf.cast(tf.train.get_global_step(), tf.float32)
        arg1 = tf.math.rsqrt(step)
        warmup_steps = 4000
        arg2 = step * (warmup_steps ** -1.5)
        learning_rate = learning_rate * tf.math.minimum(arg1, arg2)
        optimizer_name = 'Adam'

    elif optimizer_name.startswith('FastAdagrad'):
        from models.experimental import fast_adagrad_optimizer
        train_lib = fast_adagrad_optimizer
        optimizer_kwargs['config'] = config   # dense update for MirroredStrategy.
    if '(' in optimizer_name:   # e.g. FastAdagrad(dense_only=True)
        optimizer_kwargs.update(eval(
            'dict({})'.format(optimizer_name.split('(')[1].split(')')[0])))
        optimizer_name = optimizer_name.split('(')[0]

    optimizer = getattr(train_lib, optimizer_name + 'Optimizer')(
        learning_rate=learning_rate, **optimizer_kwargs)
    return optimizer


def construct_train_op(loss, optimizer, model_config):
    if model_config.random_embedding_only_update > 0.0:
        var_list = [v for v in tf.trainable_variables() if 'embed' in v.name]
        train_op = tf.cond(tf.greater(tf.random_uniform([]),
            model_config.random_embedding_only_update),
            true_fn=lambda: optimizer.minimize(
                loss, global_step=tf.train.get_global_step()),
            false_fn=lambda: optimizer.minimize(
                loss, global_step=tf.train.get_global_step(), var_list=var_list))
    elif tf_utils.cuda_ops_only():
        train_op = optimizer.minimize(loss,
            global_step=tf.train.get_global_step(),
            colocate_gradients_with_ops=True)
    else:
        if model_config.gradient_clipping <= 0.0:
            model_config.gradient_clipping = None
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=model_config.learning_rate,
            optimizer=optimizer,
            summaries=['gradients', 'gradient_norm'],
            clip_gradients=model_config.gradient_clipping,
            name='')
    return train_op


def simple_pass(mode, config):
    return mode == ModeKeys.PREDICT or (not config.model_config.siamese
        ) or tf_utils.siamese_session_eval(config, mode)


def debug_shapes(features):
    """This is useful for debugging output tensor shapes from custom dataset."""
    for k, v in features.items():
        if isinstance(v, dict):
            v['dense_shape'] = tf.Print(
                v['dense_shape'], [v['dense_shape']], message='%s shape' % k)
        else:
            features[k] = tf.Print(v, [tf.shape(v)], message='deb %s' % k)


# Truncate each row of a 2d SparseTensor from the end. Make sure the indices
# are contiguous from the left.
def _extract_last_n_per_row(sp_tensor, last_n):
    seq_len = tf_utils.sequence_length_from_sparse_tensor(sp_tensor)
    rows, cols = tf.unstack(sp_tensor.indices, num=2, axis=1)
    col_mask = tf.greater_equal(cols, tf.gather(seq_len, rows) - last_n)
    rows = tf.boolean_mask(rows, col_mask)
    row_widths = tf_utils.segment_sum_with_batch_size(tf.ones_like(rows), rows,
        tf.shape(sp_tensor)[0])
    cols = tf_utils.segment_inner_range(row_widths, out_idx=tf.int64)
    indices = tf.stack([rows, cols], axis=1)
    values = tf.boolean_mask(sp_tensor.values, col_mask)
    shape = tf.concat([sp_tensor.dense_shape[:1], [last_n]], axis=0)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


# NOTE(jyj): this kind of hardcoded logic should be avoided in the future;
# instead, implement it with configuration.
def _truncate_behavior_sequence(expt_flags, features):
    behav_cut_size = eval(str(expt_flags.get('behav_cut_size', '0.0')))
    if behav_cut_size <= 0:
        return
    utils.debuginfo('behav_cut_size is %s ' % (behav_cut_size))
    for k, v in features.items():
        if k.startswith('user_behaviour_fco_yestcontx'
            ) and v.dense_shape.get_shape().as_list()[0] == 2:
            utils.debuginfo('cut feature:{}, dtype:{}, shape:{}'.format(
                k, v.dtype, v.dense_shape))
            features[k] = _extract_last_n_per_row(v, behav_cut_size)


def input_layer_columns(features,
                        feature_columns,
                        config,
                        mode=None,
                        sparse_output=False):
    """Performs embedding lookup and query/item feature grouping."""
    if config.reader == 'FusedDataset':  # avoid serialization of SparseTensor.
        for k, v in features.items():
            if isinstance(v, dict):
                features[k] = tf.SparseTensor(**v)
    expt_flags = dict(kv.split(':') for kv in filter(
        None, config.experimental_flags.split(';')))

    _truncate_behavior_sequence(expt_flags, features)
    fconfigs = config.feature_config
    for k, v in fconfigs.items():
        temp_fname = getattr(v, "copy_from", None)
        if temp_fname:
            features[k] = features[temp_fname]
            utils.debuginfo("feature:%s copy from %s"%(k, temp_fname))
    sorted_columns = utils.siamese_sorted(feature_columns.items())

    utils.debuginfo('simple_input_columns: %s' % config.simple_input_columns)
    if config.simple_input_columns:
        with tf_utils.device_or_none(config.model_config.gpu_device):
            return simple_input_columns.build_simple_columns(
                features, feature_columns, config, mode=mode)
    model_config = config.model_config
    process_columns = functools.partial(
        process_feature_columns,
        config=config,
        mode=mode,
        sparse_output=sparse_output,
        reuse=tf.AUTO_REUSE,
        new_seq_utils=eval(str(expt_flags.get('new_seq_utils', 'True'))))
    # sorting to ensure corresponding features match between item_a and item_b.
    sorted_columns = utils.siamese_sorted(feature_columns.items())
    if tf_utils.siamese_session_eval(config, mode):
        sorted_columns = [(k, v) for k, v in sorted_columns
                          if not k.endswith('_b')]
    if model_config.item_query_separate_columns:
        assert any(k.endswith(('_a', '_b')) for k, v in sorted_columns), (
            'TODO(jyj): implement query/item separation for pointwise data.')
        query_columns = [(k, v) for k, v in sorted_columns
                         if not k.endswith(('_a', '_b'))]
        query = process_columns(features, query_columns)
        item_a_columns = [(k, v) for k, v in sorted_columns
                          if k.endswith('_a')]
        item_a = process_columns(features, item_a_columns)
        item_b_columns = [(k, v) for k, v in sorted_columns
                          if k.endswith('_b')]
        if not item_b_columns:
            return {'query': query, 'item_a': item_a}
        item_b = process_columns(features, item_b_columns, reuse=True)
        return {'query': query, 'item_a': item_a, 'item_b': item_b}

    else:  # For generic model without query/item features.
        if not model_config.siamese or not any(
                k.endswith(('_a', '_b')) for k, _ in sorted_columns):
            return {'side_a': process_columns(features, sorted_columns)}
        else:
            feature_columns_a = [(k, v) for k, v in sorted_columns
                                 if not k.endswith('_b')]
            side_a = process_columns(features, feature_columns_a)
            if not any(k.endswith('_b') for k, _ in sorted_columns):
                return {'side_a': side_a}
            feature_columns_b = [(k, v) for k, v in sorted_columns
                                 if not k.endswith('_a')]
            side_b = process_columns(features, feature_columns_b, reuse=True)
            return {'side_a': side_a, 'side_b': side_b}


def flatten_nested_features(container):
    # Supports arbitrary nested list/dict with dense tensors as leaf nodes.
    # useful for output of input_layer_columns keyed by query, item_a, all,
    # etc, as well as lists from codistillation input columns.
    if isinstance(container, dict):
        for k, v in utils.siamese_sorted(container.items()):
            for i in flatten_nested_features(v):
                yield i
    elif type(container) is tuple or type(container) is list:
        for i in container:
            for j in flatten_nested_features(i):
                yield j
    else:
        from models import new_sequence_utils as seq_utils
        assert isinstance(container, (tf.Tensor, seq_utils.sparse_embedding))
        yield container


def input_layer(columns, config, mode=None):
    empty_tensor = next(flatten_nested_features(columns))[:, :0]

    def debug(k, v):
        w = list(flatten_nested_features(v))
        if len(w) == 0:
            utils.debuginfo('No features under key %s!' % k)
            return empty_tensor
        assert all(isinstance(t, tf.Tensor) for t in w), (k, w, v)
        return tf.concat(w, 1)

    ret = {k: debug(k, v) for k, v in columns.items()}
    model_config = config.model_config
    if 'item_b' in ret or 'side_b' in ret:
        axis = int(not model_config.siamese)
        if model_config.item_query_separate_columns:
            query = tf.tile(ret['query'],
                            [2, 1]) if model_config.siamese else ret['query']
            return {
                'query': query,
                'item_a': tf.concat([ret['item_a'], ret['item_b']], axis)
            }
        else:
            return {'side_a': tf.concat([ret['side_a'], ret['side_b']], axis)}
    return ret


def output_layer(net, config, mode):
    if (config.model_config.siamese
            and (mode != ModeKeys.PREDICT or config.siamese_predict)
            and not tf_utils.siamese_session_eval(config, mode)):
        net_a, net_b = tf.split(net, num_or_size_splits=2, axis=0)
        net = net_a - net_b
    return net