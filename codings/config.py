import json
import copy
import sys
import argparse
from schemas.config_schema import config_schema
from attrdict import AttrDict
from collections import MutableMapping
import os
import re
src = os.path.abspath(__file__)
for _ in range(3):
    src = os.path.dirname(src)
    if not src in sys.path:
        sys.path.insert(0, src)
from share import fs_utils
from share import light_weight_utils as lw_utils
from collections import abc


def debug_override(nested_obj, debug_config):
    if hasattr(nested_obj, 'items'):
        for k, v in nested_obj.items():
            debug_value = debug_config.get(k)
            if debug_value is not None:
                nested_obj[k] = debug_value
            else:
                debug_override(v, debug_config)
    elif isinstance(nested_obj, (list, tuple)):
        for v in nested_obj:
            debug_override(v, debug_config)


def add_feature_group_defaults(feature_config, feature_group):
    for key, vconfig in feature_group.items():
        for fname in vconfig['features']:
            fconfig = feature_config.get(fname, {})  # if not found, do nothing.
            fconfig.update({k: v for k, v in vconfig.items() if k in fconfig})


def add_sparse_feature_defaults(config, mode):
    # NOTE(jyj): This function must be called after attach_feature_stats.
    feature_stats = config['feature_stats']
    model_config = config.model_config
    feature_config = config['feature_config']
    for param_name, global_default in [('embedding_dim', config.embedding_dim)
                                       ] + list(model_config.items()):
        for fname, fconfig in feature_config.items():
            if hasattr(fconfig, param_name) and fconfig.get(
                    param_name) is None and global_default is not None:
                fconfig[param_name] = global_default
    add_feature_group_defaults(feature_config,
                               config['shared_embedding_features'])
    add_feature_group_defaults(feature_config,
                               config['concat_sequence_features'])

    if config.reader in ['TextLineDataset', 'FusedDataset']:
        if mode == 'export' and config.lookup_embedding_ids:
            config['need_tokenization'] = False
        else:
            for fname, fconfig in config.feature_config.items():
                fstat = feature_stats[fname]
                fconfig['has_tokenizer'] = fstat.get('tokenizer') not in [
                    'default_parser', None, '']
                # if export, id features are pretokenized, unlike in training.
                fconfig['tokenize_during_training'] = (mode != 'export' or
                    fstat['data_type'] == 'string') and fconfig['has_tokenizer']
                if fconfig.get('tokenize_during_training'): # any
                    config['need_tokenization'] = True
    if config.debug_override_config:
        debug_override(config, config.debug_override_config)


def attach_feature_stats(config, feature_stats_file):
    with fs_utils.fopen(feature_stats_file) as fp:
        feature_stats = json.load(fp)
        data_dir = feature_stats.get('data_dir',
                                     fs_utils.parent_dir(feature_stats_file))
        lw_utils.make_paths_absolute(feature_stats, basedir=data_dir)
    if config.composite_features:
        composite_config = config.setdefault('composite_feature_config', {})
        for cname, fcomp in config['composite_features'].items():
            for fconfig in fcomp['output_features']:
                feature_stat = fconfig['feature_stat']
                fname = feature_stat['name']
                feature_stats[fname] = feature_stat
                fconfig['export'] = fcomp.export
                composite_config[fname] = fconfig
    config['feature_stats'] = feature_stats


def mode_override(config, role, mode=None, role_idx=0):
    from tensorflow.estimator import ModeKeys  # local import for speed.
    new_config = copy.deepcopy(config)
    if mode == ModeKeys.EVAL:   # primary eval can be performed by chief.
        config_schema.deep_update(new_config, config.eval_config)
        if role_idx > 0 and role == 'evaluator':
            lw_utils.make_paths_absolute(config, basedir=config['data_dir'])
            eval_files = config.get('secondary_eval_config_files')
            if eval_files:
                #lw_utils.debuginfo('role_idx  = %s' % str(role_idx))
                config['secondary_eval_config_file'] = eval_files[role_idx - 1]
            with fs_utils.fopen(config['secondary_eval_config_file']) as f:
                source_override = json.load(f)
            config_schema.deep_update(new_config, source_override)
            config_schema.deep_update(new_config, source_override.get(
                'eval_config', {}))
            if role_idx <= len(config['secondary_eval_configs']):
                config_schema.deep_update(new_config, config[
                    'secondary_eval_configs'][role_idx - 1])
    elif mode in [ModeKeys.TRAIN, None]:
        if config.train_config:
            config_schema.deep_update(new_config, config.train_config)
    else:
        assert mode in [ModeKeys.PREDICT, 'export'], 'mode=%s' % mode
        config_schema.deep_update(new_config, config.predict_config)
    return new_config


def parse_config(config_file, override_params, mode=None, role_and_index=None):
    """This supports both training and eval config json files."""
    with fs_utils.fopen(config_file) as f:
        json_config = json.load(f)
    config = config_schema.parse(json_config)
    from tensorflow.estimator import ModeKeys  # local import for speed.
    role, index = 'chief', 0
    if role_and_index is not None:
        role, index = role_and_index.split(':')
        index = int(index)
    lw_utils.debuginfo('override_params = %s' % override_params)
    overrides  = [] if not override_params else [
        x for y in override_params for x in y]  # flatten
    config = argparse_override( # default overrides.
        config, filter_overrides(overrides, mode=None))
    lw_utils.make_paths_absolute(config, basedir=config['data_dir'])
    config = mode_override(config, role, mode, index)
    config = argparse_override(config, filter_overrides(
        overrides, mode, index))
    config = config_schema.parse(config)
    lw_utils.make_paths_absolute(config, basedir=config['data_dir'])
    attach_feature_stats(config, config.feature_stats_file)
    add_sparse_feature_defaults(config, mode)
    ret = AttrDict(config)
    return ret


def parse_value(value, param_type):
    if param_type in [tuple, list, dict]:  # dict covers Schema. TODO: rm tuple
        assert not any(t in value for t in ['true', 'null', 'false']), value
        value = eval(value)
    elif value == 'None':
        value = None
    elif param_type != str:  # this is for other primitive types: int, float.
        value = eval('%s(%s)' % (param_type.__name__, value))
    return value


def recurse_keys(config, keys, value, param_type):
    d = config
    for k in keys[:-1]:
        if d.get(k) is None:
            d.pop(k)
        d = d.setdefault(k, {})
    value = value.strip('\'"')
    if keys[-1].endswith('-'):  # remove from list/dict
        assert value[0] + value[-1] == '[]', (keys, value)
        for k in sorted(eval(value), reverse=True):
            d[keys[-1][:-1]].pop(k, None)
    # value is a list of pairs keyed by str or int; latter implies list insertion.
    elif keys[-1].endswith('+'):
        keys[-1] = keys[-1][:-1]
        assert value[:2] + value[-2:] == '[()]', (keys, value)
        assert param_type in [list, dict], (keys, value, param_type)
        if param_type is dict:
            d[keys[-1]].update(eval(value))
        else:
            insert_values = eval(value)  # e.g. [(0, obj0), (2, obj2), (-1, obj1)]
            assert isinstance(insert_values, abc.MutableMapping)
            assert all(isinstance(k[0], int) for k in insert_values)
            original_length = len(d[keys[-1]])
            # NOTE: negative indices add 1 to accommodate insertion to the end.
            # https://stackoverflow.com/questions/30212447
            insert_values = [((original_length + k + 1) if k < 0 else k, v)
                             for k, v in insert_values]
            for k, v in sorted(insert_values, reverse=True):
                d[keys[-1]].insert(k, v)
        d[keys[-1]].update(eval(value))
    else:
        d[keys[-1]] = parse_value(value, param_type)


def filter_overrides(overrides, mode=None, role_idx=0):
    ret = []
    from tensorflow.estimator import ModeKeys
    for override in overrides:
        tmp = override.split('.')
        first_key, remainder = tmp[0], '.'.join(tmp[1:])
        if first_key.startswith('eval_config'):
            if mode == ModeKeys.EVAL and role_idx == 0:
                ret.append(remainder)
        elif first_key.startswith('train_config'):
            if mode == ModeKeys.TRAIN:
                ret.append(remainder)
        elif first_key.startswith('predict_config'):
            if mode in [ModeKeys.PREDICT, 'export']:
                ret.append(remainder)
        elif first_key.startswith('secondary_eval_configs['):
            secondary_eval_idx = int(first_key.split(']')[0].split('[')[1])
            if mode == ModeKeys.EVAL and role_idx == secondary_eval_idx + 1:
                ret.append(remainder)
        elif mode is None:
            ret.append(override)
    return ret


def argparse_override(config, overrides, mode='train'):
    for override in overrides:
        try:
            tmp = override.split('=')
            key, value = tmp[0], '='.join(tmp[1:]).strip('\'"')
            keys = key.split('.')  # last key may end with +/-
            param_type = config_schema.lookup_by_keys(
                keys[:-1] + [re.split('-|\+', keys[-1])[0]], field='type')
            recurse_keys(config, keys, value, param_type)
        except Exception:
            assert False, 'Can\'t parse override: {} . {}, {}'.format(
                override, lw_utils.full_trace(), config)
    return config


def extract_field(config_path, keys, basedir_key, basedir,
                  override_value=None):
    with fs_utils.fopen(config_path) as f:
        config = config_schema.parse(AttrDict(json.load(f)))
    key_list = ''.join(['["%s"]' % k for k in keys.split('.')])
    if override_value:
        argparse_override(config, ['%s=%s' % (keys, override_value)])
    basedir = basedir or config.get(basedir_key)
    if basedir:
        lw_utils.make_paths_absolute(config, basedir)
    try:
        return eval('config%s' % key_list)
    except:
        lw_utils.debuginfo('key list not in config: %s' % str(key_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config Utilities.')
    parser.add_argument('mode', metavar='mode', type=str, help='running mode')
    parser.add_argument('--config_path', type=str)
    parser.add_argument(
        '--config_keys', type=str, help='A dot separated sequence of keys')
    parser.add_argument('--basedir_key', type=str, default=None)
    parser.add_argument('--basedir', type=str, default=None)
    parser.add_argument('--override_value', type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.mode == 'extract_field':
        sys.stdout.write(
            str(
                extract_field(
                    args.config_path,
                    args.config_keys,
                    args.basedir_key,
                    args.basedir,
                    override_value=args.override_value)))
    else:
        raise ValueError('Unsupported mode.')
