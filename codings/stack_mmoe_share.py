from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.estimator import ModeKeys
import os, sys
src = os.path.realpath(__file__)
for _ in 'model_lib/ranking/src/models/experimental/moe_models'.split('/'):
    src = os.path.dirname(src)
    if src not in sys.path:
        sys.path.insert(0, src)
import network_utils
from component_lib import loss_fn
import utils
from share import tf_utils
import dnn_logloss
from adjust_loss_weight import get_weights
import multi_tower
from component_lib import loss_fn
import multi_task_metric_utils
import multi_task_utils

def train_eval_spec(loss, metrics, config, mode, **kwargs):
    if mode == ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics, **kwargs)
    # Create training op.
    assert mode == ModeKeys.TRAIN
    optimizer = network_utils.get_optimizer(config)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # ensure all other weights stay frozen during uuid embedding export.
        train_op = network_utils.construct_train_op(loss, optimizer,
            config.model_config) if not config.export_embedding else tf.no_op()
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op, **kwargs)

def predict_spec(predictions, config, args, features):
    # predict_spec.auxiliary_tags are uuid, pvid, etc, not available at serving time.
    if args.mode != 'export' and config.auxiliary_tags:
        for fname in config.auxiliary_tags:
            feature = getattr(config.feature_stats, fname)
            fkey = utils.get_feature_key(
                feature, fname, export=True, config=config)
            predictions[fname] = features[fkey]
    return tf.estimator.EstimatorSpec(
        ModeKeys.PREDICT, predictions=predictions)

multi_task_name = ['ctr', 'ctavr', 'ctcvr', 'ctr_share', 'ctavr_share', 'ctcvr_share']
multi_task_threshold = [0, 1, 2]
class StackMMoEAfterForwardNet(dnn_logloss.AfterForwardNet):
    def get_loss(self, train_data, aux_data):
        labels, logits = train_data['labels'], train_data['logits']
        model_config = self._model_config
        multi_task_labels = multi_task_utils.multi_task_label_transform(
            labels, multi_task_threshold)
        if logits.get_shape()[1] > len(multi_task_labels):
            multi_task_labels = multi_task_labels+multi_task_labels
        assert len(multi_task_labels) == logits.get_shape()[1]

        # loss_name = ['ctr_loss', 'ctavr_loss', 'ctcvr_loss']
        losses = []
        for head_num in range(logits.get_shape()[1]):
            this_loss = eval('loss_fn.%s' % model_config.train_loss_fn)(
                multi_task_labels[head_num],
                logits[:, head_num:head_num + 1],
                weights=train_data['weights'])
            losses.append(this_loss)
            train_data['%s_loss'%multi_task_name[head_num]] = this_loss

        for head_num in range(0, logits.get_shape()[1]):
          tf_utils.summary_histogram('loss_%s' % head_num, losses[head_num])
        if self._mode != ModeKeys.TRAIN:
            loss = tf.reduce_sum(tf.stack(losses, axis=0), axis=0)
        else:
            if model_config.get('weighted_loss', False):
                loss = multi_task_utils.uncertainty_weighting_loss(
                    losses, multi_task_name)
            else:
                loss = tf.reduce_sum(tf.stack(losses, axis=0), axis=0)
        tf_utils.summary_histogram('loss_final', loss)
        if 'regularizer' in aux_data:
            loss = network_utils.maybe_apply_regularization(
                loss, aux_data, model_config, self._mode)
        return loss

    def after_forward_pass(self, train_data, aux_data, features):
        config, args, mode = self._config, self._args, self._mode

        model_config = config.model_config
        predictions = self.get_predictions(
            train_data['logits'],
            aux_data,
            input_dict=config.export_embedding and aux_data['input_dict'],
            output_dict=config.export_embedding and aux_data.get('output_dict'))
        if mode == ModeKeys.PREDICT:
            return predict_spec(predictions, config, args, features)
        # Compute loss.
        with utils.ctx_or_none(tf.name_scope, 'secondary_eval_%d' % args.index,
                               args.role == 'evaluator' and args.index > 0):
            loss = self.get_loss(train_data, aux_data)
            multi_args = {
                'heads' : ['ctr', 'ctcvr', 'ctavr', 'ctcvr_1_5'],
                'label_threshold_ctcvr' : 2.0,
                'label_threshold_ctr' : 0.0,
                'label_threshold_ctavr' : 1.0,
                'label_threshold_ctcvr_1_5' : 1.5,
                'metrics' : {
                                'loss_ctr': tf.metrics.mean(train_data['ctr_loss']),
                                'loss_ctavr': tf.metrics.mean(train_data['ctavr_loss']),
                                'loss_ctcvr': tf.metrics.mean(train_data['ctcvr_loss'])
                            }
            }
            metrics = multi_task_metric_utils.all_metrics(
                train_data['labels'], predictions, features,
                self._config, self._mode, self._args, multi_args)
            if config.export_embedding:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(
                    tf.train.get_global_step(), args.checkpoint_to_export))
            return train_eval_spec(
                loss, metrics, config, mode, predictions=predictions)

    def get_predictions(self, logits, aux_data, input_dict=None, output_dict=None):
        # calc prob
        assert len(multi_task_name) == logits.get_shape()[1]
        rets = {}
        multi_task_prob = {}
        epsilon = 1e-12
        for head_num in range(logits.get_shape()[1]):
            multi_task_prob['%s_prob'%multi_task_name[head_num]] = tf.nn.sigmoid(logits[:, head_num:head_num+1])
            if self._mode == ModeKeys.PREDICT:
                ret = {
                    'probabilities_%s'%multi_task_name[head_num]:
                    tf.log(
                        tf.maximum(multi_task_prob['%s_prob'%multi_task_name[head_num]], epsilon))
                }
            else:
                ret = {
                    'probabilities_%s'%multi_task_name[head_num]:
                    multi_task_prob['%s_prob'%multi_task_name[head_num]]
                }
            rets.update(ret)
        if self._mode == ModeKeys.PREDICT:
            gmv_head_value = tf.reshape(aux_data['price_head'], [tf.shape(multi_task_prob['ctr_prob'])[0], 1])
            gmv = tf.log(tf.abs(gmv_head_value) + 1)
            rets.update({'probabilities_cvr': gmv})
            #rets.update({'channel': aux_data['channel']})
            #rets.update({'gate_output': aux_data['gate_output']})
        else:
            rets.update({'probabilities_ctcvr_1_5':
                        multi_task_prob['ctcvr_prob']})
 
        #add gate ouput
        #rets.update({'query_gate_0': aux_data['query_gate_0']})
        #rets.update({'query_gate_1': aux_data['query_gate_1']})
        #rets.update({'query_gate_2': aux_data['query_gate_2']})
        #rets.update({'query_gate_3': aux_data['query_gate_3']})

        #rets.update({'cid_gate_0': aux_data['cid_gate_0']})
        #rets.update({'cid_gate_1': aux_data['cid_gate_1']})
        #rets.update({'cid_gate_2': aux_data['cid_gate_2']})
        #rets.update({'cid_gate_3': aux_data['cid_gate_3']})

        #rets.update({'user_gate_0': aux_data['user_gate_0']})
        #rets.update({'user_gate_1': aux_data['user_gate_1']})
        #rets.update({'user_gate_2': aux_data['user_gate_2']})
        #rets.update({'user_gate_3': aux_data['user_gate_3']})
        #end add

        return rets

def model_fn(features, labels, mode, params):
    config = params['config']
    model_config = config.model_config
    feature_columns = params['feature_columns']
    args = params['args']
    utils.debuginfo(
        'role_and_index = %s; CUDA_VISIBLE_DEVICES = %s' %
        (config.role_and_index, os.environ['CUDA_VISIBLE_DEVICES']))
    with tf.device('/device:CPU:0'):
        # Output sparse_embeddings objects for sequence embedding columns.
        columns = network_utils.input_layer_columns(
            features, feature_columns, config, mode=mode, sparse_output=False)
        weights = get_weights(labels, columns, features, config, args, mode)
        if not network_utils.simple_pass(mode,
                                         config) or config.siamese_predict:
            columns = multi_tower.merge_siamese_ab(columns, model_config, mode)
        utils.debuginfo(columns)
        input_dict = multi_tower.flatten_columns_as_dict(columns)
    with tf.device(model_config.gpu_device):
        input_dict, labels = network_utils.maybe_swap_embedding_ab(
            input_dict, labels, config, mode)
        forward_net = multi_tower.MultiTowerForwardNet(
            config, mode, weights, original_features=features)
        train_data, aux_data = forward_net.forward_pass(input_dict, labels)
        aux_data['price_head'] = features[config.feature_stats['price_a']['feature_id']]
        #aux_data['channel'] = features[config.feature_stats['channel_a']['feature_id']]
        after_net = StackMMoEAfterForwardNet(config, args, mode)
        return after_net.after_forward_pass(train_data, aux_data, features)
