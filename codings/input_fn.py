from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attrdict
import copy
import datetime
from component_lib import label_fn
from component_lib import filter_fn
import os
import re
import sys
import tensorflow as tf
src = os.path.realpath(__file__)
for _ in range(2):
    src = os.path.dirname(src)
    if not src in sys.path:
        sys.path.insert(0, src)
import utils
from tensorflow.estimator import ModeKeys
import random
from share import tf_utils, shared_utils
import json
import input_processors
from schemas.feature_schema import dense_feature_schema
from schemas.config_schema import config_schema


hdfs_cache = {} # hdfs file IO can be very unreliable, so read only once.


def augment_feature_config(config):
    feature_config = config['feature_config']
    default_dense_feature = dense_feature_schema.parse({'export': True})
    for tag in config.auxiliary_tags:
        feature_config[tag] = default_dense_feature


def feature_parse_spec(config, mode=ModeKeys.TRAIN):
    """Specification for all features, including components for label."""
    feature_specs = {}
    feature_config = config.feature_config
    features_to_parse = set(k for k, v in feature_config.items()
                            if mode != ModeKeys.PREDICT
                            or v['export'] or config.siamese_predict)
    # label components will be popped during training.
    if mode != ModeKeys.PREDICT:
        assert all(
            t in features_to_parse
            for t in config.label_transform.components), (features_to_parse,
                                                          config)
    for fname in features_to_parse:
        fstat = getattr(config.feature_stats, fname)
        fconfig = feature_config.get(fname)
        fkey = utils.get_feature_key(
            fstat,
            fname,
            export=(mode == ModeKeys.PREDICT and not config.siamese_predict),
            config=config)
        if fkey in feature_specs:
            # This should only happen with siamese network.
            utils.debuginfo(
                'repeated fkey %s in feature_specs %s' % (fkey, feature_specs))
            assert getattr(config.model_config, 'siamese', False)
            continue
        dtype, ctype, default_val = (fstat.data_type,
                                     fstat.feature_column_type,
                                     fstat.get('default_value'))
        export_mode = config.get('export_mode')
        vocab_file = fstat.get('vocab_file')

        if export_mode:
            if fstat.data_type == 'string': # id/float_list always variable_len
                if vocab_file and config.lookup_embedding_ids:
                    # tokenize outside tf model during serving.
                    dtype, ctype, default_val = 'int64', 'variable_length', None
                else:   # otherwise tokenize within the model during serving.
                    ctype, default_val = 'fixed_length', None
        elif fconfig.get('tokenize_during_training'):   # training/eval/predict.
            # The alternative is pre-tokenized vocab id list in tfrecord.
            dtype, ctype, default_val = 'string', 'fixed_length', None

        if ctype == 'fixed_length':
            width = fstat.get('row_width', 1)
            shape = [] if width == 1 else [width]
            default_val = default_val if width == 1 else [default_val] * width
            feature_specs[fkey] = tf.FixedLenFeature(shape,
                dtype=shared_utils.data_type[dtype], default_value=default_val)
        elif ctype == 'variable_length':
            feature_specs[fkey] = tf.VarLenFeature(
                dtype=shared_utils.data_type[dtype])
        else:
            raise Exception('wrong feature column type: %s' % ctype)
    return feature_specs


def parser(record, features_to_parse, config, mode=ModeKeys.TRAIN):
    if config.reader == 'TFRecordDataset':
        parsed = tf.parse_single_example(record, features_to_parse)
    else:
        assert config.reader == 'TextLineDataset'
        parsed = input_processors.parse_tsv(record, config, mode=mode)
    for fname, v in config.feature_config.items():
        fstat = getattr(config.feature_stats, fname)
    if mode == ModeKeys.PREDICT:
        return parsed
    label_transform = config.label_transform
    component_keys = [
        utils.get_feature_key(
            getattr(config.feature_stats, fname),
            fname,
            export=(mode == ModeKeys.PREDICT and not config.siamese_predict),
            config=config) for fname in label_transform.components
    ]
    label_func = label_fn._get_label_func(label_transform.type)
    return parsed, tf.reshape(label_func(component_keys, parsed), [-1])


def shuffle_and_assign_filenames(config, args):
    random.seed(a=config.shuffle_seed, version=2)
    filenames = utils.glob_files(config.train_dataset_files)
    if config.shuffle:  # otherwise files come in natural sorting order.
        assert config.each_worker_sees_all_data > -1
        random.shuffle(filenames)
    files_per_worker = len(filenames) // args.num_tasks
    if config.each_worker_sees_all_data > 0:  # default = 1.
        starting_file = files_per_worker * args.worker_index    # div strategy.
        filenames = filenames[starting_file:] + filenames[:starting_file]
    else:  # ensure different worker sees different files, using mod strategy.
        filenames = filenames[args.worker_index::args.num_tasks]
    if args.verbose and args.role in ['chief', 'worker']:
        utils.debuginfo('train filenames for %s: %d files like %s' %
                        (args.role_and_index, len(filenames), filenames[:1]))
    return filenames


def apply_reader(filenames, config):
    if config.num_readers_per_worker > 1:  # default = 1.
        file_set = tf.data.Dataset.from_tensor_slices(filenames)
        data_set = file_set.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: getattr(tf.data, config.reader)(filename),
                cycle_length=config.num_readers_per_worker))
    else:
        data_set = getattr(tf.data, config.reader)(filenames)
    return data_set


def create_fused_dataset(config, filenames, mode=ModeKeys.TRAIN):
    assert config.dataset_header_file
    assert not config.skip_header
    utils.add_custom_ops_python_path()
    import fused_dataset
    try:
        tsv_header_str = utils.fopen(config.dataset_header_file).readline().strip()
        hdfs_cache['tsv_header_str'] = tsv_header_str
    except:
        tsv_header_str = hdfs_cache.get('tsv_header_str')
        assert tsv_header_str, (config.dataset_header_file, hdfs_cache, utils.get_ip())
    label_transform = config['label_transform']
    if label_transform and label_transform['components']:
        if mode == ModeKeys.PREDICT and not config.session_sampler:
            utils.debuginfo('Make sure labels are not part of predict features!')
        label_fn._validate_label_transform(label_transform)
    else:
        config['feature_config'] = {
            k: v for k, v in config.feature_config.items() if v.get('export', True)}
    # Will use lookup_dataset_reader if config contains feature_tables.
    num_parallel_calls = config.num_parallel_calls
    if mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL and config.batch_size > 0:
        num_parallel_calls = 1
    data_set = fused_dataset.FusedDataset(
        filenames,
        json.dumps(config.feature_stats),
        json.dumps(config),
        tsv_header_str,
        label_transform,
        config.batch_size,
        num_parallel_calls,
        epochs=config.epochs,
        take_steps=config.take_steps is None and -1 or config.take_steps,
        # use distributed mode for training and file delimited eval (bs = -2).
        num_buffer_threads=num_parallel_calls,
        buffers_per_thread=config.prefetch_batches,
        lookup_embedding_ids=config.lookup_embedding_ids,
        drop_remainder=config.dataset_drop_remainder
        and mode != ModeKeys.PREDICT,
        # aux features like price for training weights will be normalized later.
        normalize_numeric=config.normalize_numeric, is_training=mode == ModeKeys.TRAIN)
    return data_set


def create_fused_dataset_with_failover(config, filenames, args, mode=ModeKeys.TRAIN):
    assert config.dataset_header_file
    assert not config.skip_header
    utils.add_custom_ops_python_path()
    utils.add_custom_ops_python_path(
        base_dir='external/tensorflow/tensorflow/core/custom_ops/dds')
    import fused_dataset
    import dds_ops
    try:
        tsv_header_str = utils.fopen(config.dataset_header_file).readline().strip()
        hdfs_cache['tsv_header_str'] = tsv_header_str
    except:
        tsv_header_str = hdfs_cache['tsv_header_str']
    label_transform = config['label_transform']

    if label_transform and label_transform['components']:
        if mode == ModeKeys.PREDICT: # non-session tsv shouldn't have label during predict
            assert config.session_sampler
        label_fn._validate_label_transform(label_transform)
    else:
        assert mode == ModeKeys.PREDICT, mode

    # Will use lookup_dataset_reader if config contains feature_tables.
    num_parallel_calls = config.num_parallel_calls
    if mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL and config.batch_size > 0:
        num_parallel_calls = 1

    def _flat_map_fn(f):
        data_set = fused_dataset.FusedDataset(
            f,
            json.dumps(config.feature_stats),
            json.dumps(config),
            tsv_header_str,
            label_transform,
            config.batch_size,
            1,
            1,
            take_steps=config.take_steps is None and -1 or config.take_steps,
            async_input=False,
            # use distributed mode for training and file delimited eval (bs = -2).
            num_buffer_threads=1,
            buffers_per_thread=config.prefetch_batches,
            lookup_embedding_ids=config.lookup_embedding_ids,
            drop_remainder=config.dataset_drop_remainder
            and mode != ModeKeys.PREDICT,
            # aux features like price for training weights will be normalized later.
            normalize_numeric=config.normalize_numeric, is_training=mode == ModeKeys.TRAIN,
            decode_fn_ver='v2')
        return data_set

    if args.dds_address == '':
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.repeat(config.epochs)
        dataset = dds_ops.ParallelInterleaveWithFailoverDataset(
            dataset, datackpt_server_address='',
            task_index=0, map_func=_flat_map_fn, cycle_length=num_parallel_calls,
            block_length=2, sloppy=True, buffer_output_elements=8,
            prefetch_input_elements=0)
    else:
        worker_id = int(args.task_index)
        dataset = dds_ops.DataGrpcGeneratorDataset(
                    task_index=worker_id, task_name='worker', dds_address=args.dds_address)
        dataset = dds_ops.ParallelInterleaveWithFailoverDataset(
            dataset, datackpt_server_address=args.dds_address,
            task_index=worker_id, map_func=_flat_map_fn, cycle_length=num_parallel_calls,
            block_length=2, sloppy=True, buffer_output_elements=8,
            prefetch_input_elements=0)

    dataset = dataset.prefetch(128)
    return dataset


def train_input_fn(config, args):
    filenames = shuffle_and_assign_filenames(config, args)
    utils.debuginfo('filenames for %s: %d files like %s' %
                    (ModeKeys.TRAIN, len(filenames), filenames[:1]))
    augment_feature_config(config)
    if config.reader == 'FusedDataset':
        return (create_fused_dataset_with_failover(config, filenames, args)
                if config.failover_mode else
                create_fused_dataset(config, filenames))# .map(
        # lambda parsed, label: (parsed, label), num_parallel_calls=None)

    data_set = apply_reader(filenames, config)
    if config.cache_input:
        data_set = data_set.cache()
    data_set = data_set.repeat(config.epochs)
    if config.shuffle:
        data_set = data_set.shuffle(buffer_size=config.buffer_size)
    features_to_parse = feature_parse_spec(config)
    data_set = data_set.map(
        lambda record: parser(record, features_to_parse, config),
        num_parallel_calls=config.get('num_parallel_calls', None))
    if config.filter_fn:
        for ff in config.filter_fn.split(';'):
            data_set = data_set.filter(eval('filter_fn.%s' % ff))
    data_set = data_set.batch(
        config.batch_size, drop_remainder=config.dataset_drop_remainder)
    if config.take_steps:
        data_set = data_set.take(config.take_steps)
    data_set = input_processors.batch_process(data_set, ModeKeys.TRAIN, config)
    data_set = data_set.prefetch(buffer_size=config.prefetch_batches)
    return data_set


def sequential_forward_input_fn(config, args, filenames, mode=None):
    try:
        if isinstance(filenames, str):  # otherwise a list.
            filenames = utils.glob_files(filenames)
        hdfs_cache['filenames'] = filenames
    except:
        assert 'filenames' in hdfs_cache, filenames
        filenames = hdfs_cache['filenames']
    batch_size = config.batch_size
    utils.debuginfo('filenames for %s: %d files like %s' %
                    (mode, len(filenames), filenames[:1]))
    config['epochs'] = 1
    if mode == ModeKeys.EVAL and not filenames[0].startswith(
            'hdfs') and os.path.getsize(filenames[0]) * len(
                filenames) < 1e5 and config.dataset_drop_remainder:
        utils.debuginfo('Do not drop remainder for very small eval dataset.')
        config['dataset_drop_remainder'] = False
    augment_feature_config(config)
    if config.reader == 'FusedDataset':
        return (create_fused_dataset_with_failover(config, filenames, args, mode=mode)
                if config.failover_mode else
                create_fused_dataset(config, filenames, mode=mode))
    else:
        assert not config.lookup_embedding_ids
    if config.skip_header:
        assert config.reader == 'TextLineDataset'
        assert len(filenames) == 1
    if not config.failover_mode:
        data_set = getattr(tf.data, config.reader)(filenames)
        if config.skip_header:
            data_set = data_set.skip(1)
        if config.epochs > 1:
            data_set = data_set.repeat(config.epochs)
    else:
        utils.add_custom_ops_python_path(
            base_dir='external/tensorflow/tensorflow/core/custom_ops/dds')
        import dds_ops
        def _flat_map_fn(f):
            data_set = getattr(tf.data, config.reader)(f)
            if config.skip_header:
                data_set = data_set.skip(1)
            return data_set

        if isinstance(filenames, tuple):
            filenames = list(filenames)
        data_set = tf.data.Dataset.from_tensor_slices(filenames)
        if config.epochs > 1:
            data_set = data_set.repeat(config.epochs)
        data_set = dds_ops.FlatMapWithFailoverDataset(
            data_set, datackpt_server_address='', task_index=0,
            map_func=_flat_map_fn)
    features_to_parse = feature_parse_spec(config, mode)
    data_set = data_set.map(
        lambda record: parser(record, features_to_parse, config, mode=mode),
        num_parallel_calls=config.get('num_parallel_calls', None))
    if config.filter_fn:
        for ff in config.filter_fn.split(';'):
            data_set = data_set.filter(eval('filter_fn.%s' % ff))
    #data_set = data_set.cache(filename='eval_tmp.cache')
    data_set = data_set.batch(
        config.batch_size,
        drop_remainder=config.dataset_drop_remainder and mode == ModeKeys.EVAL)
    if mode == ModeKeys.PREDICT and config.predict_batches > 0:
        data_set = data_set.take(config.predict_batches)
    return input_processors.batch_process(data_set, mode, config)


def eval_input_fn(config, args):
    if config.eval_style == 'siamese_pointwise':
        config.feature_config = {
            k: v
            for k, v in config.feature_config.items() if not k.endswith('_b')
        }
    return sequential_forward_input_fn(
        config, args, config.eval_dataset_files, mode=ModeKeys.EVAL)


def predict_input_fn(config, args):
    if not config.siamese_predict:
        config.feature_config = {
            k: v
            for k, v in config.feature_config.items() if not k.endswith('_b')
        }
    return sequential_forward_input_fn(
        config, args, config.test_dataset_files, mode=ModeKeys.PREDICT)
