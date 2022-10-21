"""
Manual distributed run:
python src/run.py train_and_evaluate -c model_config.json -s 20 \
--chief_ip=10.252.199.4 --worker_ips=10.252.199.4,10.252.199.4 \
--role_and_index=chief:0 --port_start=2222
python src/run.py train_and_evaluate -c model_config.json -s 20 \
--chief_ip=10.252.199.4 --worker_ips=10.252.199.4,10.252.199.4 \
--role_and_index=worker:0 --port_start=2222
python src/run.py train_and_evaluate -c model_config.json -s 20 \
--chief_ip=10.252.199.4 --worker_ips=10.252.199.4,10.252.199.4 \
--role_and_index=worker:1 --port_start=2222

Automated distributed run: see src/run_distributed.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import sys
import time
import tensorflow as tf
from tensorflow.estimator import ModeKeys
from tensorflow.python.training import device_setter
from tensorflow.python.training import training
from tensorflow.contrib.training.python.training import device_setter as device_setter_lib
src = os.path.abspath(__file__)
for _ in ['src', 'ranking', 'model_lib']:
    src = os.path.dirname(src)
    if src not in sys.path:
        sys.path.insert(0, src)
import config as conf
import feature_columns
import utils
import importlib
import pickle
import checkpoint_utils as ckpt_utils
import numpy as np
from tf_revised_lib import tf_estimator
from tf_revised_lib import tf_run_config
from tf_revised_lib import tf_estimator_training
from analysis import compute_metrics
from itertools import accumulate

src = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(src, '../..'))
from ranking.src import input_fn
from share import fs_utils, tf_utils, shared_utils
from tensorflow.python.platform import tf_logging as logging
tf.logging.set_verbosity(tf.logging.INFO)


def pre_steps(args, mode=ModeKeys.TRAIN):
    config = conf.parse_config(
        args.config, args.param, mode=mode, role_and_index=args.role_and_index)
    if args.port_start is not None:
        assert args.chief_ip
        cluster = utils.construct_cluster(args, config)
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': cluster,
            'task': {
                'type': args.role,
                'index': args.index
            }
        })
    if mode == 'export':
        # TODO(jyj): enable these with custom id lookup.
        # config['lookup_embedding_ids'] = False
        config['cuda_only'] == args.gpu_serving
    cpu_serving = not args.gpu_serving and mode == 'export'
    if config.cuda_only:
        tf_utils.global_disable_non_cuda_ops()
    if args.role in ['evaluator', 'chief'] and args.index > 0 and args.verbose:
        config2 = copy.deepcopy(config)
        feature_stats = config2.pop('feature_stats')  # reduce verbosity.
        utils.debuginfo('Final config for %s = %s' %
                        (args.role_and_index, json.dumps(config2, indent=4)))
    model_config = config.model_config
    gpu_device_indices = [t for t in filter(
        None, args.gpu_device_indices.split(',')) if int(t) > -1]
    num_gpus = len(gpu_device_indices)
    if cpu_serving or num_gpus == 0:
        config['model_config']['gpu_device'] = '/device:CPU:0'
    if args.role == 'ps' and model_config.optimizer.startswith('FastAdagrad'):
        # parameter server loads ops from a pbtxt, hence only knows ops defined
        # in _pywrap_tensorflow_internal.so.
        shared_utils.build_bazel_binary('training_ops.so')
        setattr(config, 'model_config', model_config)
        # override -1 set in main()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device_indices[args.index % num_gpus]
    utils.debuginfo('gpu_device = ' + model_config.gpu_device)
    utils.transfer_args_to_config(args, config)
    columns = None
    # to differentiate from ModeKeys.PREDICT
    config['export_mode'] = mode == 'export'  # export_mode not in config_schema.
    if not config.simple_input_columns:
        columns = feature_columns.build_model_columns(config, mode=mode)
    train_distribute = None
    local_eval = set(filter(None, args.evaluator_ips.split(','))).issubset(
        [None, 'localhost', utils.get_ip()])
    if model_config.get('train_distribute_strategy') == 'MirroredStrategy':
        if mode == ModeKeys.TRAIN:
            if local_eval:  # otherwise assume evaluator on a different host.
                num_gpus = max(num_gpus - 1, 0) # reserve last one for eval.
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device_indices[:-1])
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device_indices)
            train_distribute = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
        elif mode == ModeKeys.EVAL: # override main, to put all evals on the same gpu.
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device_indices[-1]
            if not local_eval and args.role == 'evaluator':
                # chief/worker also has EVAL mode.
                assert config.model_path.startswith('/mnt/')
    elif model_config.get('train_distribute_strategy') == 'MultiWorkerMirroredStrategy':
        assert local_eval, 'Not implemented yet.'
        if mode == ModeKeys.TRAIN:
            if args.role == 'chief':
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device_indices[:-1])
            elif args.role == 'worker':
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device_indices)
            train_distribute = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
                cluster_resolver=None
            )
        elif mode == ModeKeys.EVAL:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device_indices[-1]
    run_config = tf.estimator.RunConfig()
    device_filters = [
        '/job:ps', '/job:%s/task:%d' % (run_config.task_type, run_config.task_id)
    ]
    if run_config.task_type == 'ps':
        device_filters = None
    session_config = tf.ConfigProto(
        device_filters= device_filters if mode==ModeKeys.TRAIN else None,
        device_count={'GPU': 0 if cpu_serving else num_gpus},
        allow_soft_placement=not tf_utils.cuda_ops_only(),
        log_device_placement=False, # otherwise this would spew a ton of logs.
        gpu_options=None if cpu_serving else tf.GPUOptions(allow_growth=True))
    run_config = tf_run_config.RunConfig(
        save_checkpoints_secs=args.ckpt_secs,
        session_config=session_config, train_distribute=train_distribute,
        keep_checkpoint_max=None)
    if run_config.task_type == 'chief':
      args.task_index = 0
    else:
      args.task_index = run_config.task_id + 1

    if config.ps_balance_strategy:
        if run_config.task_type:
            worker_device = '/job:%s/task:%d' % (run_config.task_type, run_config.task_id)
        else:
            worker_device = '/job:worker'
        if run_config.num_ps_replicas > 0:
            device_fn = training.replica_device_setter(
                ps_tasks=run_config.num_ps_replicas,
                worker_device=worker_device,
                merge_devices=True,
                ps_ops=list(device_setter.STANDARD_PS_OPS),
                cluster=run_config.cluster_spec,
                ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                    run_config.num_ps_replicas,
                    device_setter_lib.byte_size_load_fn))
            run_config = run_config.replace(device_fn=device_fn)
    model = importlib.import_module('models.' + (config.model or args.model))
    ws = None
    if config.warm_start_checkpoint_path:
        var_name_to_vocab_info = None
        if config.var_name_to_vocab_info:
            var_name_to_vocab_info = {
                k: tf.train.VocabInfo(v.new_vocab_file, v.new_vocab_size,
                                      v.num_oov_buckets, v.old_vocab_file, -1)
                                     # tf.glorot_uniform_initializer())
                for k, v in config.var_name_to_vocab_info.items()
            }
        vars_to_warm_start = config.vars_to_warm_start
        if len(config.vars_to_warm_start_list) > 0:
            vars_to_warm_start = list(config.vars_to_warm_start_list)
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=config.warm_start_checkpoint_path,
            vars_to_warm_start=vars_to_warm_start,
            var_name_to_prev_var_name=config.var_name_to_prev_var_name,
            var_name_to_vocab_info=var_name_to_vocab_info)
    predictor = tf_estimator.Estimator(
        model_fn=model.model_fn,
        params={
            'feature_columns': columns,
            'config': config,
            'args': args,
        },
        model_dir=config.model_path,
        log_dir=config.tensorboard_dir,
        config=run_config,
        warm_start_from=ws)
    return config, predictor, columns


def find_best_ckpt(args, config):
    metrics_steps = ckpt_utils.get_metrics_dictionary(
        args.log_file, config.model_path)
    assert len(args.best_metric_to_export.split(':')) == 2
    best_ckpts = [t[1] for t in ckpt_utils.topk_ckpts_all_metrics(
        [args.best_metric_to_export.split(':') + [1]], metrics_steps,
        config.model_path)]
    if best_ckpts:
        best_ckpt = best_ckpts[0]
    else:
        utils.debuginfo(
            'No checkpoints found using topk metrics; use default!')
        best_ckpt = None
    return best_ckpt


def get_checkpoint_path(ckpt, config):
    if ckpt is None:
        return None
    else:
        assert isinstance(ckpt, int), ckpt
        return os.path.join(config.model_path, 'model.ckpt-%d' % ckpt)


def export_best_ckpt(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = (args.gpu_device_indices
                                          if args.gpu_serving else '-1')
    print('gpu_serving = %s' % args.gpu_serving)
    config, predictor, export_columns = pre_steps(args, mode='export')
    # add pvid to fconfig for rerank
    if args.mode == 'session_export':
        input_fn.augment_feature_config(config)
    if args.checkpoint_to_export > -1:
        best_ckpt = args.checkpoint_to_export
    else:
        best_ckpt = find_best_ckpt(args, config)
    utils.debuginfo('checkpoint_to_export = %s' % best_ckpt)
    if config.export_embedding: # to record uuid embedding in RNN models.
        from models.experimental.sequence_models import rnn
        tmp_dir = rnn.warmstart_for_embedding_export(args, config, best_ckpt)
        config2, predictor2, export_columns = pre_steps(args, mode=ModeKeys.TRAIN)
        predictor2.train(input_fn=lambda: input_fn.train_input_fn(config2, args))
        rnn.move_ckpt_for_embedding_export(tmp_dir, config.model_path, best_ckpt)
    servable_model_path, ckpt_path = ckpt_utils.export_model_graph(
        predictor, config, config.model_path, ckpt=best_ckpt)
    if args.tps_test_file:
        from gen_tps_test_file import gen_tps_test_file
        gen_tps_test_file(
            args.tps_test_file,
            args.config,
            os.path.join(config.model_path, servable_model_path),
            config,
            count=2000)
    return servable_model_path, ckpt_path


def train(args):
    config, predictor, _ = pre_steps(args, mode=ModeKeys.TRAIN)
    # Train model.
    predictor.train(input_fn=lambda: input_fn.train_input_fn(config, args))


def ordinal(n):
    return '%d%s' % (n, {
        1: 'st',
        2: 'nd',
        3: 'rd'
    }.get(n if n < 20 else n % 10, 'th'))


def train_and_evaluate(args):
    # in distributed mode, continuous eval uses config rather than eval_config.
    config, predictor, _ = pre_steps(args, mode=ModeKeys.EVAL if
        args.role == 'evaluator' else ModeKeys.TRAIN)
    eval_config, _, _ = pre_steps(args, mode=ModeKeys.EVAL)
    # Train model and evaluate
    train_hooks = []
    if config.stop_at_steps:
        train_hooks.append(
            tf.train.StopAtStepHook(num_steps=config.stop_at_steps))
    if config.add_timeline:
        train_hooks.append(tf.train.ProfilerHook(
            save_steps=100, output_dir='timeline'))
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn.train_input_fn(config, args),
        hooks=train_hooks,
        max_steps=args.max_steps)
    eval_hooks = []
    if args.role in ['chief', 'evaluator']:
        if 'secondary_eval' in args.topk_metrics:
            assert (eval_config.secondary_eval_config_file or
                eval_config.secondary_eval_config_files)
        eval_hooks.append(ckpt_utils.CkptDeletionHook(
            args.log_file, args.topk_metrics, eval_config.model_path,
            args.role_and_index))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn.eval_input_fn(eval_config, args),
        hooks=eval_hooks,
        throttle_secs=args.throttle_secs,
        # None to evaluate in all evaluation data.
        steps=eval_config.eval_steps,
        name=ordinal(args.index + 1) if
        (args.role == 'evaluator' and args.index > 0) else None,
        # Secondary eval not available at the beginning.
        start_delay_secs=max(10 * (args.index + 1), args.throttle_secs))
    tf_estimator_training.train_and_evaluate(predictor, train_spec, eval_spec)
    if args.role_and_index == 'chief:0':
        export_best_ckpt(args)


def evaluate(args):
    config, predictor, _ = pre_steps(args, mode=ModeKeys.EVAL)
    if args.checkpoint_to_eval > -1:
        ckpt = args.checkpoint_to_eval
    else:
        ckpt = find_best_ckpt(args, config)
    if args.mode == 'evaluate_via_inference':
        from inference import inference_eval
        metrics = inference_eval.evaluate(ckpt, config, args)
    else:
        metrics = predictor.evaluate(
            input_fn=lambda: input_fn.eval_input_fn(config, args),
            checkpoint_path=get_checkpoint_path(ckpt, config),
            steps=config.eval_steps,
            name='standalone')  # avoid mangling tfevent
    utils.debuginfo('metrics = %s' % metrics)
    return metrics


def _format_output(prediction_output, n_rows=1):
    if isinstance(prediction_output, bytes):
        return [prediction_output.decode('utf-8')] * n_rows
    elif type(prediction_output) is np.ndarray:
        ret = prediction_output.tolist()
        if len(ret) > 1:  # embeddings.
            return ['|'.join(map(str, ret))]
        else:
            return ret
    else:
        return [prediction_output] * n_rows


def _file_indices(args, config, test_files, worker_idx):
    global_num_files = len(test_files)
    assert worker_idx < args.num_workers
    def get_sharded_file(idx, file_base=config.prediction_file + '.tsv'):
        return '%s-%05d-of-%05d' % (file_base, idx, global_num_files)
    indices = []
    for idx in range(global_num_files):
        sharded_file = get_sharded_file(idx)
        if not fs_utils.exists(sharded_file + '.DONE'):
            indices.append(idx)
        else:
            with fs_utils.fopen(sharded_file + '.DONE') as f:
                tmp = f.readline().strip('\r\n').split('\t')
            # start_timestamp should be the same for all run.py instances.
            if len(tmp) < 5 or args.start_timestamp < float(tmp[4]):
                indices.append(idx)
            elif worker_idx == 0:
                utils.debuginfo('Skip %s marked DONE ..' % sharded_file)
    assert indices, 'All test files predicted already!'
    return indices[worker_idx::args.num_workers]


def predict(args):
    utils.debuginfo('os.environ["CUDA_VISIBLE_DEVICES"] = %s; ip address = %s'
        % (os.environ['CUDA_VISIBLE_DEVICES'], utils.get_ip()))
    config, predictor, _ = pre_steps(args, mode=ModeKeys.PREDICT)
    file_idx, test_file_indices, global_num_files = 0, [0], 1
    test_files = utils.glob_files(config.test_dataset_files)
    global_num_files = len(test_files)
    if args.distributed_predict:
        test_file_indices = _file_indices(args, config, test_files, args.index)
        test_files = [test_files[k] for k in test_file_indices]
    if args.checkpoint_to_predict > -1:
        ckpt = args.checkpoint_to_predict
    else:
        ckpt = find_best_ckpt(args, config)
    config['test_dataset_files'] = test_files
    utils.debuginfo('Predicting using checkpoint %s' % ckpt)
    parent = '/'.join(config.prediction_file.split('/')[:-1])
    if not fs_utils.exists(parent):
        fs_utils.mkdir(parent)

    tsv_file = config.prediction_file + '.tsv'

    # format must agree with sharded_file in run_distributed.py.
    def output_tsv(idx):
        return '%s-%05d-of-%05d' % (tsv_file, idx, global_num_files)

    if config.session_sampler == 'all_items':
        input_reader = utils.MultiFileSessionTsvReader(test_files,
            config.dataset_header_file, config.session_columns_offset)
    else:
        input_reader = utils.MultiFileTextReader(test_files)
    if global_num_files > 1:
        line_counts = input_reader.count_output_lines()
        utils.debuginfo('role_and_index: %s; test files line counts: %s' %
            (args.role_and_index, list(zip(test_files, line_counts))))
        line_cumsum = list(accumulate(line_counts))
        num_lines = line_cumsum[file_idx]  # file_idx = 0.
        for idx in test_file_indices:
            if fs_utils.exists(output_tsv(idx)):
                fs_utils.rm(output_tsv(idx))
    elif fs_utils.exists(tsv_file):
        fs_utils.rm(tsv_file)

    predictions = []

    def maybe_dump_pickle(idx, predictions):
        if args.dump_predict_pickle:
            pickle_file = config.prediction_file
            if global_num_files > 1:
                pickle_file += '-%05d-of-%05d' % (idx, global_num_files)
            with fs_utils.fopen(pickle_file, 'wb') as h:
                pickle.dump(predictions, h)

    def mark_done(file_idx, row_cnt):
        with fs_utils.fopen(output_tsv(test_file_indices[file_idx]) + '.DONE',
                  'w') as f3:
            f3.write('\t'.join(map(str, [utils.get_ip(), args.role_and_index,
                file_idx, row_cnt - (0 if file_idx == 0 else
                line_cumsum[file_idx - 1]), time.time()])))

    f = fs_utils.fopen(output_tsv(test_file_indices[file_idx]) if
        global_num_files > 1 else tsv_file, 'w')
    predict_hooks = []
    if config.add_timeline and args.index == 0:
        predict_hooks.append(tf_estimator.PredictProfilerHook(
            save_steps=100, output_dir='predict_timeline', max_steps=1000))
    for row_cnt, prediction in enumerate(predictor.predict(
        input_fn=lambda: input_fn.predict_input_fn(config, args),
        checkpoint_path=get_checkpoint_path(ckpt, config), hooks=predict_hooks)):
        if row_cnt == 0:
            keys = sorted(prediction.keys())
            header_line = '\t'.join(keys)
            if config.predict_tsv_with_input_columns:
                with fs_utils.fopen(config['dataset_header_file']) as f2:
                    input_headers = f2.readline().strip().split('\t')
                header_line += '\t' + '\t'.join(input_headers)
            if global_num_files == 1:
                f.write(header_line + '\n')
            else:
                header_file = config.prediction_file + '.header'
                if test_file_indices[file_idx] == 0:
                    with fs_utils.fopen(header_file, 'w') as g:
                        g.write(header_line)
                elif fs_utils.exists(header_file):
                    with fs_utils.fopen(header_file) as g:
                        assert g.readline().strip('\r\n') == header_line
        if global_num_files > 1:
            if row_cnt == num_lines:
                maybe_dump_pickle(test_file_indices[file_idx], predictions)
                predictions = []
                utils.debuginfo('%s finished predicting %s' %
                                (args.role_and_index, test_files[file_idx]))
                mark_done(file_idx, row_cnt)
                file_idx += 1
                num_lines = line_cumsum[file_idx]
                f.close()
                f = fs_utils.fopen(
                    output_tsv(test_file_indices[file_idx]), 'w')
        # e.g. prediction = {'class_ids': array([[0]], dtype=int32),
        # 'probabilities': array([0.21103612], dtype=float32), 'logits':
        # array([-1.3186913], dtype=float32)}
        tmp = list(zip(*[_format_output(prediction[k], 1) for k in keys]))
        assert len(tmp) == 1
        if config.predict_tsv_with_input_columns:
            input_columns = input_reader.get_next()
            tmp = [list(t) + s for t, s in zip(tmp, [input_columns])]
        f.write('\n'.join('\t'.join(map(str, row)) for row in tmp) + '\n')
        if args.dump_predict_pickle:
            predictions.append(prediction)
    # end for
    f.close()
    mark_done(file_idx, row_cnt + 1)
    maybe_dump_pickle(test_file_indices[file_idx], predictions)

    return predictions  # Only last predictions, needed for non-distributed UT.


def main():
    # sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    parser = utils.init_run_parser()
    args, _ = parser.parse_known_args()
    if args.verbose:
        utils.debuginfo('args: %s' % str(args))
    if 'TF_CONFIG' in os.environ:   # running on 9N.
        tf_json = json.loads(os.environ['TF_CONFIG'])
        args.role = tf_json.get("task").get("type")
        args.index = int(tf_json.get("task").get("index"))
        args.role_and_index = args.role + ":" + str(args.index)
        print('TF_CONFIG = %s' % os.environ.get('TF_CONFIG'))
        logging.info('TF_CONFIG = %s\nargs.role = %s\n' %
                     (os.environ.get('TF_CONFIG'), str(args.role)))
        if args.mode == 'predict':
            os.environ.pop('TF_CONFIG')
    if 'BAZEL_REBUILD_SECS' not in os.environ and args.bazel_rebuild_secs:
        os.environ['BAZEL_REBUILD_SECS'] = str(args.bazel_rebuild_secs)
    utils.prepare_model_dir(args)
    utils.augment_run_args(args)
    gpu_device_indices = list(filter(None, args.gpu_device_indices.split(',')))
    if args.mode == 'export' or args.worker_index is None:
        gpu_index = '-1'
    else:
        is_eval = int(args.role == 'evaluator')
        gpu_index = (1 - 2 * is_eval) * args.worker_index - is_eval
        gpu_index = gpu_device_indices[gpu_index % len(gpu_device_indices)]
    if args.worker_gpu_override_map:
        parts = [t.split(':') for t in args.worker_gpu_override_map.split(',')]
        gpu_index = {
            '%s:%s' % (role, index): gpu_idx
            for role, index, gpu_idx in parts
        }.get(args.role_and_index, gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index

    def run_job():
        if args.mode == 'train':
            train(args)
        elif args.mode == 'gen_fake_train':
            config, _, _ = pre_steps(args, mode='export')
            tf_utils.add_fake_train('/tmp/fake_train', config, config['feature_stats'])
        elif args.mode == 'train_and_evaluate':
            train_and_evaluate(args)
        elif args.mode.startswith('evaluate'):
            evaluate(args)
        elif args.mode == 'predict':
            predict(args)
        elif args.mode in ['export', 'session_export']:
            assert args.role_and_index == 'chief:0', args.role_and_index
            export_best_ckpt(args)
        elif args.mode == 'session_evaluate':
            predict(args)
            config = conf.parse_config(args.config, None)
            prediction_tsv = '%s.tsv' % config.prediction_file
            metric_file = '%s.joined.tsv' % config.prediction_file
            model_config = config.model_config
            prediction_header_map = ','.join(['probabilities_'+head+':'+head
                for head in model_config.session_eval_heads]) if (
                model_config.session_eval_heads) else 'probabilities:probabilities'
            tsv_header_file = None
            tsv_file_to_join = config.session_validate_tsv_with_header
            compute_metrics.join_tsv_file(prediction_tsv, metric_file,
                                          prediction_header_map, config,
                                          tsv_header_file, tsv_file_to_join)
            groupby_fea = 'pvid,query,uid'
            if model_config.task_weights:
                compute_metrics.eval_multi_task_model(metric_file, 'eval',
                                            groupby_fea.split(','),
                                            compute_metrics.metrics, False,
                                            None, model_config.task_weights,
                                            model_config.session_eval_heads)
            else:
                compute_metrics.eval_one_model(metric_file, 'eval',
                                           groupby_fea.split(','),
                                           compute_metrics.metrics, False,
                                           None, model_config.session_eval_heads)
        elif args.mode == 'rerank_session_evaluate':
            predict(args)
            config = conf.parse_config(args.config, None)
            metric_file = '%s.tsv' % config.prediction_file
            # groupby_fea = 'pvid,query,uid'
            groupby_fea = 'pvid'
            compute_metrics.eval_one_model(metric_file, 'eval',
                                           groupby_fea.split(','),
                                           compute_metrics.metrics, False,
                                           None)

        else:
            raise Exception('unsupported mode: %s' % args.mode)

    if args.profile_dir:
        with tf.contrib.tfprof.ProfileContext(args.profile_dir):
            run_job()
    else:
        run_job()


if __name__ == '__main__':
    main()