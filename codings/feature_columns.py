from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
import sys
import os
import functools
import tensorflow as tf
from tensorflow.python.platform import gfile
import utils
from tensorflow.contrib.feature_column.python.feature_column import \
sequence_feature_column as sfc
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.estimator import ModeKeys
import feature_groups as fg
from tf_revised_lib import tf_revised_utils as tf_rev
from models.column_decorator import log_x_plus_one, log_numeric_plus_one

src = os.path.abspath(__file__)
for _ in ['src', 'ranking', 'model_lib']:
    src = os.path.dirname(src)
    if src not in sys.path:
        sys.path.insert(0, src)
from share import shared_utils, tf_utils
from tf_revised_lib.tf_revised_utils import customized_embedding_column, \
customized_shared_embedding_columns


def get_capped_min_max_norm(x, fmin, fmax):
    return (tf.maximum(fmin, tf.minimum(tf.cast(x, tf.float32), fmax)) -
            fmin) / (fmax - fmin)


def log_min_max(x, fmin, fmax):
    if isinstance(x, tf.SparseTensor):
        x_value = x.values
    else:
        x_value = x
    x_log = log_x_plus_one(x_value)
    fmin_log = log_x_plus_one(fmin)
    fmax_log = log_x_plus_one(fmax)
    x_value = get_capped_min_max_norm(x_log, fmin_log, fmax_log)
    if isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(
            indices=x.indices, values=x_value, dense_shape=x.dense_shape)
    else:
        return x_value


def min_max_combo(x, fmin, fmax):
    fmin, fmax = map(float, [fmin, fmax])
    assert fmin <= fmax
    min_max_norm = lambda y: get_capped_min_max_norm(y, fmin, fmax)
    triple_combo = lambda y: [f(min_max_norm(y)) for f in [tf.square,
        tf.sqrt, lambda z: z]]
    return tf.stack(
        triple_combo(x) + triple_combo(tf.abs(x))
        # + [tf.square(log_x_plus_one(x)),
        # tf.sqrt(log_x_plus_one(x))]
        ,
        axis=1)


def _check_permission(
        file_path):  # https://stackoverflow.com/questions/1861836
    if not file_path.startswith('hdfs://'):
        assert os.stat(file_path).st_mode & 32, 'No permission: ' + file_path
    else:
        assert gfile.Exists(file_path), file_path
    return file_path


def parse_time(x, default_value=0.0):
    if isinstance(x, tf.SparseTensor):
        #####fill empty rows here since it is hard to fill empty rows in 3d sparse tenosr
        x, _ = tf_utils.sparse_fill_empty_rows(x, default_value)
        x_value = x.values
    else:
        x_value = x
    sec = tf.mod(x_value, 60) / 60.0
    min_x = tf.floordiv(x_value, 60)
    minu = tf.mod(min_x, 60) / 60.0
    hour_x = tf.floordiv(min_x, 60)
    hour = tf.mod(hour_x, 24) / 24.0
    day_x = tf.floordiv(hour_x, 24)
    day = tf.mod(day_x, 30) / 30.0
    month = tf.floordiv(day_x, 30) / 12.0
    x_value = tf.stack([sec, minu, hour, day, month], axis=1)
    if isinstance(x, tf.SparseTensor):
        indices0 = tf.tile(x.indices, [1, 5])
        indices0 = tf.reshape(indices0, shape=[-1, 2])
        indices1 = tf.tile(
            tf.range(0, 5, dtype=tf.int64), [tf.shape(x.indices)[0]])
        indices1 = tf.reshape(indices1, shape=[-1, 1])
        indices = tf.concat([indices0, indices1], axis=1)
        x_value = tf.reshape(x_value, shape=[-1])
        return tf.SparseTensor(
            indices=indices,
            values=x_value,
            dense_shape=tf.concat([x.dense_shape, [5]], axis=0))
    else:
        return x_value


class ColumnBuilder:
    def __init__(self, mode, config, feature_stats):
        self._mode = mode
        self._export = mode in [
            ModeKeys.PREDICT, 'export'] and not config.siamese_predict
        self._config = config
        self._feature_stats = feature_stats
        self._feature_config = config.feature_config
        self._composite_config = config.get('composite_feature_config', {})

    def extract_feature_info(self, colname, fgconfig):
        # get info for all related feature_names
        fnames = fgconfig['feature_names']
        fstats = [getattr(self._feature_stats, name) for name in fnames]
        fkeys = [
            utils.get_feature_key(
                stat, name, export=self._export, config=self._config)
            for name, stat in zip(fnames, fstats)
        ]
        tf_types = [shared_utils.data_type[stat.data_type] for stat in fstats]
        fconfigs = [
            self._feature_config.get(fname, self._composite_config.get(fname))
            for fname in fnames
        ]
        assert all(cfg is not None for cfg in fconfigs)
        # is_seqs = [fc.get('embedding_type', '').startswith(
        #     'sequence') for fc in fconfigs]
        return fnames, fstats, fkeys, tf_types, fconfigs

    # TODO (wenyun): move the feature columns to a folder when it gets too many
    def build_indicator_column(self, colname, fgconfig):
        fnames, fstats, fkeys, tf_types, _ = self.extract_feature_info(
            colname, fgconfig)
        ret = {}
        for fn, stat, key, typ in zip(fnames, fstats, fkeys, tf_types):
            indicator_column = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=key,
                    vocabulary_file=_check_permission(stat.vocab_file),
                    vocabulary_size=fgconfig['vocab_size'],
                    num_oov_buckets=fgconfig['embedding_oov_buckets'],
                    default_value=None,
                    dtype=typ))
            ret[fn] = indicator_column
        return ret

    def _embedding_params(self, fstat, fgconfig, model_config):
        is_sequence = fgconfig['feature_column'] == 'sequence_embedding'
        params = {
            'dimension': fgconfig['embedding_dim'],
            'combiner': None if is_sequence else fgconfig.get('embedding_combiner', 'sqrtn')
        }
        embedding_file = fgconfig.get('embedding_file',
                                      model_config.get('embedding_file', None))
        if embedding_file:
            pre_trained_embedding_dim = fgconfig.get(
                'pre_trained_embedding_dim', params['dimension'])
            random_embedding_dim = params[
                'dimension'] - pre_trained_embedding_dim
            params['initializer'] = shared_utils.pre_trained_initializer(
                pre_trained_embedding_dim,
                embedding_file,  # may not exist
                _check_permission(fstat.vocab_file),
                random_embedding_dim)
            params['trainable'] = True  # random_embedding_dim > 0
        else:
            ckpt_to_load_from = fgconfig.get('ckpt_to_load_from', None)
            tensor_name_in_ckpt = fgconfig.get('tensor_name_in_ckpt', None)
            if ckpt_to_load_from and tensor_name_in_ckpt:
                params['ckpt_to_load_from'] = ckpt_to_load_from
                params['tensor_name_in_ckpt'] = tensor_name_in_ckpt
                params['trainable'] = False
                utils.debuginfo('ckpt_to_load_from: {}'.format(ckpt_to_load_from))
            else:
                params['initializer'] = shared_utils.weight_initializer(
                    model_config, embedding=True)
                params['trainable'] = True
        return params

    def build_embedding_column(self, colname, fgconfig):
        config = self._config
        def column_fn(fstat, is_sequence=False):
            parser = fstat.get('tokenizer', 'default_parser')
            id_input = self._config.lookup_embedding_ids
            id_input |= parser.split('(')[0].endswith('_with_vocab_parser')
            id_input &= not fgconfig.get('keep_raw_feature')
            if is_sequence:
                if config.contrib_dense_table or id_input:
                    return functools.partial(
                        tf_rev.revised_sequence_categorical_column_with_vocabulary_file,
                        config=self._config, mode=self._mode, id_input=id_input)
                return sfc.sequence_categorical_column_with_vocabulary_file
            elif id_input:
                return tf_rev.categorical_column_with_vocabulary_ids
            elif config.contrib_dense_table:
                return functools.partial(
                    tf_rev.revised_categorical_column_with_vocabulary_file,
                    config=self._config, mode=self._mode)
            else:
                return tf.feature_column.categorical_column_with_vocabulary_file

        fnames, fstats, fkeys, tf_types, fconfigs = self.extract_feature_info(
            colname, fgconfig)
        categorical_columns = [column_fn(
            stat, is_sequence=fconfig.embedding_type.startswith('sequence'))(
                    key=key,
                    vocabulary_file=_check_permission(stat.vocab_file),
                    vocabulary_size=fgconfig['vocab_size'],
                    default_value=None,
                    dtype=type_,
                    num_oov_buckets=fgconfig['embedding_oov_buckets'])
            for name, stat, key, type_, fconfig in zip(fnames, fstats, fkeys,
                                                       tf_types, fconfigs)
        ]
        params = self._embedding_params(fstats[0], fgconfig,
                                        self._config.model_config)
        fgconfig['mode'] = self._mode
        column_kwargs = {'feature_config': fgconfig}
        custom_column = not fgconfig.get('dedup_vocab') or fgconfig.get(
            'num_partitions', 1) > 1
        if fgconfig.get('shared_embedding'):
            embedding_columns = customized_shared_embedding_columns(
                categorical_columns, custom=custom_column,
                shared_embedding_collection_name=colname,
                **params, **column_kwargs)
            return dict(zip(fnames, embedding_columns))
        else:
            assert len(fnames) == 1
            column_fn = functools.partial(
                customized_embedding_column, **column_kwargs
            ) if custom_column else tf.feature_column.embedding_column
            embedding_column = column_fn(categorical_columns[0], **params)
            return {fnames[0]: embedding_column}

    def build_numeric_column(self, colname, fgconfig):
        fnames, fstats, fkeys, tf_types, fconfigs = self.extract_feature_info(
            colname, fgconfig)
        result = {}

        def numeric_column(idx, **kwargs):
            is_sequence = fstats[idx]['feature_column_type'] == 'variable_length'
            return (sfc.sequence_numeric_column if is_sequence else
                    (tf.feature_column.numeric_column))(
                        fkeys[idx],
                        dtype=tf_types[idx],
                        default_value=fstats[idx].default_value,
                        **kwargs)
        # normalize_numeric means normalization done within FusedDataset, except
        # those marked keep_raw_feature (for computing training weight, etc).
        normalization = fgconfig.get('normalization') if (fgconfig.get(
        'keep_raw_feature') or not self._config.normalize_numeric) else None
        for i in range(len(fnames)):
            shape = (fstats[0].get('row_width', 1), )
            if not normalization: # could be normalized in FusedDataset.
                if fgconfig.get('normalization') == 'bucketization':
                    shape = (len(fstats[0].buckets) + 1,)
                numeric_feature = numeric_column(i, shape=shape)
            elif normalization == 'bucketization':
                numeric_feature = tf.feature_column.bucketized_column(
                    numeric_column(i), fstats[i].buckets)
            elif normalization == 'standardization':
                numeric_feature = numeric_column(
                    i,
                    normalizer_fn=
                    lambda x: (tf.to_float(x) - fstats[i].mean) / fstats[i].std
                )
            elif normalization == 'min_max':
                fmin, fmax = map(float, [fstats[i].min, fstats[i].max])
                assert fmax >= fmin
                numeric_feature = numeric_column(
                    i,
                    normalizer_fn=
                    lambda x: get_capped_min_max_norm(x, fmin, fmax))
            elif normalization == 'log':
                numeric_feature = numeric_column(
                    i, normalizer_fn=log_x_plus_one)
            elif normalization == 'log_bucketization':
                numeric_feature = tf.feature_column.bucketized_column(
                    numeric_column(i, normalizer_fn=log_x_plus_one),
                    [log_numeric_plus_one(x) for x in fstats[i].buckets])
            elif normalization == 'log_min_max':
                fmin, fmax = map(float, [fstats[i].min, fstats[i].max])
                assert fmax >= fmin
                numeric_feature = numeric_column(
                    i, normalizer_fn=lambda x: log_min_max(x, fmin, fmax))
            elif normalization.startswith('dummy_constant='):
                dummy_constant = float(fgconfig['normalization'].split('=')[1])
                numeric_feature = numeric_column(
                    i,
                    normalizer_fn=
                    lambda x: tf.to_float(x) * 0.0 + dummy_constant)
            elif normalization == 'min_max_combo':
                numeric_feature = numeric_column(
                    i,
                    shape=(6, ),
                    normalizer_fn=
                    lambda x: min_max_combo(x, fstats[i].min, fstats[i].max))
            elif normalization == 'parse_time':
                numeric_feature = numeric_column(
                    i,
                    shape=(5, ),
                    normalizer_fn=
                    lambda x: parse_time(x, fstats[i].default_value))
            else:
                raise ValueError('Unknown feature normalization option!')
            result[fnames[i]] = numeric_feature
        return result


def build_model_columns(config, mode):
    export = mode in [ModeKeys.PREDICT, 'export'] and not config.siamese_predict
    feature_groups = fg.aggregate_columns(config, export=export)
    columns = {}
    builder = ColumnBuilder(mode, config, config.feature_stats)
    for colname, fgconfig in feature_groups.items():
        feature_column = fgconfig['feature_column']
        assert feature_column in ['indicator', 'embedding', 'numeric']
        columns.update(getattr(builder, 'build_%s_column' % feature_column)(
            colname, fgconfig))
    # utils.debuginfo('columns = %s' % columns.keys())
    return columns
