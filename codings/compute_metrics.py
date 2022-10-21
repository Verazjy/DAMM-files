"""
Compute metrics, such as auc, and ndcg based on predction results.
usage:
python model_lib/ranking/src/analysis/compute_metrics.py \
 -c cpj/config/personal_care_tune/default.json \
 -o pred/personal_care_test_default.metrics

Use the following command to compute metrics for baseline, which is based on
their posistion.
python model_lib/ranking/src/analysis/compute_metrics.py \
 -c cpj/config/personal_care_tune/default.json \
 -o pred/personal_care_test_default.metrics_baseline -b True
"""

import math
import numpy as np
import sys
import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
import uuid
import random
import pickle
import argparse
import re
import tensorflow as tf
src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)
import utils
import parser_utils
import config as conf
sys.path.insert(0, os.path.join(src, '../..'))
from share import fs_utils
from share import light_weight_utils as lw_utils
from analysis.process_handler import ProcessHandler

s_score = 'probabilities'
s_label = 'label'
s_price = 'price'

# metrics = (
#     """auc2_ord@1,auc2_ord@5,auc2_ord@10,auc2_ord@20,auc2_ord@40,auc2_ord,click_ndcg@1,click_ndcg@5,
#     click_ndcg@10,click_ndcg@20,click_ndcg@40,click_ndcg,order_ndcg@1,
#     order_ndcg@5,order_ndcg@10,order_ndcg@20,order_ndcg@40,order_ndcg,price@1,
#     price@10,price@20,price@40,price,gmv_ord@1,gmv_ord@10,gmv_ord@20,gmv_ord@40,gmv_ord,
#     click_avg_pos@1,click_avg_pos@10,click_avg_pos@20,click_avg_pos@40,
#     click_avg_pos,order_avg_pos@1,order_avg_pos@10,order_avg_pos@20,
#     order_avg_pos@40,order_avg_pos""").split(',')
metrics = (
    """auc2_ord@1,auc2_ord@5,auc2_ord@10,auc2_ord@20,auc2_ord@40,auc2_ord,
    auc2@1,auc2@5,auc2@10,auc2@20,auc2@40,auc2,
    price@1,price@10,price@20,price@40,price,
    gmv_ord@1,gmv_ord@10,gmv_ord@20,gmv_ord@40,gmv_ord,
    gmv@1,gmv@10,gmv@20,gmv@40,gmv,
    is_jx_product@1,is_jx_product@10,is_jx_product@20,is_jx_product@40,gmv,
    is_self@1,is_self@10,is_self@20,is_self@40,is_self""").split(',')

class MetricBase:
    def __init__(self):
        pass

    def __call__(self, session):
        pass

    def check_feature_meet(self, session):
        return self.get_features_required() < set(session.columns)


class mrr(MetricBase):
    def __init__(self, k=999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        mrr = 0.0
        k = 0
        for idx, (name, row) in enumerate(session[:self.k].iterrows()):
            label = row[s_label]
            if label == 1:
                mrr += 1.0 / (idx + 1)
                k += 1
        if k > 0:
            mrr /= k
        return mrr


class mrr_weighted(MetricBase):
    def __init__(self, k=999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

        return set([s_score, s_label, s_price])

    def __call__(self, session):
        mrr = 0.0
        k = 0
        for idx, (name, row) in enumerate(session[:self.k].iterrows()):
            label = row[s_label]
            price = row[s_price]
            if label > 0:
                mrr += 1.0 / (idx + 1) * price
                k += 1
        if k > 0:
            mrr /= k
        return mrr


class ap(MetricBase):
    def __init__(self, k=999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        ap = 0.0
        k0 = 0
        k1 = 0
        for _, row in session[:self.k].iterrows():
            score = row[self.s_score]
            label = row[s_label]
            if label == 1:
                k1 += 1
                ap += float(k1) / float(k0 + k1)
            else:
                k0 += 1
        if k1 > 0:
            ap /= k1
        return ap


class ap_weighted(MetricBase):
    def __init__(self, k=999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        ap = 0.0
        k0 = 0
        k1 = 0
        pos = 0

        for _, row in session[:self.k].iterrows():
            label = row[s_label]
            price = row[s_price]
            if label > 0:
                k1 += price
                pos += 1
                ap += float(k1) / float(k0 + k1)
            else:
                k0 += 1
        if pos > 0:
            ap /= pos
        return ap


class auc(MetricBase):
    def __init__(self, k=999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        auc = 0.0
        tp0 = 0.0
        fp0 = 0.0
        P = 0
        N = 0
        k0 = 0
        k1 = 0

        for _, row in session[:self.k].iterrows():
            act = row[s_label]
            if act > 1e-9:
                P += 1.0
            else:
                N += 1.0

        if P == 0:
            return 0.0
        if N == 0:
            return 1.0

        for _, row in session[:self.k].iterrows():
            act = row[s_label]
            if act > 1e-9:
                k1 += 1
            else:
                k0 += 1
            tp1 = float(k1) / P
            fp1 = float(k0) / N
            auc += (fp1 - fp0) * (tp1 + tp0) / 2
            # print("kk", (fp1-fp0)*(tp1+tp0)/2, fp1, fp0, tp1, tp0,k1, k0)
            tp0 = tp1
            fp0 = fp1

        return auc


class acc(MetricBase):
    def __init__(self, k=999999, s_score=s_score, label_threshold=1):
        self.k = k
        self.s_score = s_score
        self.label_threshold = label_threshold

    def __call__(self, session):
        acc = 0.
        P = []
        N = []
        for _, row in session[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act > self.label_threshold:
                P.append(row)
            else:
                N.append(row)

        if len(P) == 0 or len(N) == 0:
            return 0.0

        for row1 in P:
            for row2 in N:
                if row1[self.s_score] > row2[self.s_score]:
                    acc += 1.
        return acc / (len(P) * len(N))


class auc2_base(MetricBase):
    def __init__(self, k=999999, s_score=s_score, label_threshold=0):
        self.k = k
        self.s_score = s_score
        self.label_threshold = label_threshold

    def __call__(self, session):
        auc = 0.0
        tp0 = 0.0
        fp0 = 0.0
        fp1 = 1.0
        P = 0
        N = 0
        k0 = 0
        k1 = 0

        for _, row in session[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act >= self.label_threshold:
                P += 1.0
            else:
                N += 1.0

        if P == 0:
            return 0.0
        if N == 0:
            return 1.0

        for _, row in session[:self.k].iterrows():
            act = row[s_label]
            if act >= self.label_threshold:
                k1 += 1
                tp1 = float(k1) / P
                fp1 = float(k0) / N
                auc += (fp1 - fp0) * (tp1 + tp0) / 2
                # print("kk", (fp1-fp0)*(tp1+tp0)/2, fp1, fp0, tp1, tp0, k1, k0)
                tp0 = tp1
                fp0 = fp1
            else:
                k0 += 1

        auc += 1.0 - fp1
        return auc


class auc2(auc2_base):  # aka auc2_clk
    def __init__(self, k=999999, s_score=s_score):
        super(auc2, self).__init__(k, s_score, label_threshold=1)


class auc2_ord(auc2_base):
    def __init__(self, k=999999, s_score=s_score):
        super(auc2_ord, self).__init__(k, s_score, label_threshold=2)


############compute auc juchi#############
class auc_zigzag(MetricBase):
    def __init__(self, k=999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        rec = []
        for _, row in session[:self.k].iterrows():
            rec.append((row[s_label], row[self.s_score]))

        sum_pospair = 0.0
        sum_npos = 0.0
        sum_nneg = 0.0
        buf_pos = 0.0
        buf_neg = 0.0
        wt = 1
        for j in range(len(rec)):
            ctr = rec[j][0]
            # keep bucketing predictions in same bucket
            if j != 0 and rec[j][1] != rec[j - 1][1]:
                sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5)
                sum_npos += buf_pos
                sum_nneg += buf_neg
                buf_neg = 0.0
                buf_pos = 0.0

            buf_pos += ctr * wt
            buf_neg += (1.0 - ctr) * wt

        sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5)
        sum_npos += buf_pos
        sum_nneg += buf_neg
        if sum_npos * sum_nneg == 0:
            return 0.5
        sum_auc = sum_pospair / (sum_npos * sum_nneg)
        return sum_auc


class click_dcg(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        sumdcg = 0
        i = 0
        for _, row in session[:self.k].iterrows():
            act = row[s_label]
            if act > 0:
                sumdcg += ((1 << int(act)) - 1) * math.log(2) / math.log(2 + i)
            i = i + 1
        return sumdcg


class click_ndcg(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        a = click_dcg(k=self.k)
        s1 = a(session)
        session2 = session.sort_values('label', ascending=False)
        s2 = a(session2)
        if s2 <= 1e-9:
            return 0
        else:
            return s1 / s2


class order_dcg(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        sumdcg = 0
        i = 0
        for _, row in session[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act >= 2:
                sumdcg += ((1 << int(act)) - 1) * math.log(2) / math.log(2 + i)
            i = i + 1
        return sumdcg


class order_ndcg(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        a = order_dcg(k=self.k)
        s1 = a(session)
        session2 = session.sort_values('label', ascending=False)
        s2 = a(session2)
        if s2 <= 1e-9:
            return 0
        else:
            return s1 / s2


class rmse(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        return math.sqrt(
            np.mean([(row[self.s_score] - row[s_label])**2
                     for _, row in session[:self.k].iterrows()]))


class diff(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        return sum([(row[self.s_score] - row[s_label])
                    for _, row in session[:self.k].iterrows()])


class price(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        price_list = [row[s_price] for _, row in session[:self.k].iterrows()]
        return sum(price_list) * 1.0 / len(price_list)


class is_self(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        is_self_list = [row['is_self_fea'] for _, row in session[:self.k].iterrows()]
        return sum(is_self_list) * 1.0 / len(is_self_list)

class is_jx_product(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        value_list = [row['is_jx_product'] for _, row in session[:self.k].iterrows()]
        return sum(value_list) * 1.0 / len(value_list)

class gmv(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        return sum([
            row[s_price] if row[s_label] > 0 else 0
            for _, row in session[:self.k].iterrows()
        ])


class gmv_ord(MetricBase):
    def __init__(self, k=99999999, s_score=s_score, label_threshold=2):
        self.k = k
        self.s_score = s_score
        self.threshold = label_threshold

    def __call__(self, session):
        return sum([
            row[s_price] if row[s_label] >= self.threshold else 0
            for _, row in session[:self.k].iterrows()
        ])


class click_avg_pos(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        click_list = []
        index = 0
        for _, row in session[:self.k].iterrows():
            index += 1
            if row[s_label] > 0:
                click_list.append(index)
        return sum(click_list) * 1.0 / len(click_list) if len(
            click_list) != 0 else 0


class order_avg_pos(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        order_list = []
        index = 0
        for _, row in session[:self.k].iterrows():
            index += 1
            if row[s_label] >= 2:
                order_list.append(index)
        return sum(order_list) * 1.0 / len(order_list) if len(
            order_list) != 0 else 0


class click_pos_dis(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        click_list = []
        index = 0
        for _, row in session[:self.k].iterrows():
            index += 1
            if row[s_label] > 0:
                click_list.append(index)
        return click_list


class order_pos_dis(MetricBase):
    def __init__(self, k=99999999, s_score=s_score):
        self.k = k
        self.s_score = s_score

    def __call__(self, session):
        order_list = []
        index = 0
        for _, row in session[:self.k].iterrows():
            index += 1
            if row[s_label] >= 2:
                order_list.append(index)
        return order_list


def eval_metric(group, metrics_list, case_output, compute_baseline, head_cxr):
    metric_result_gp = []

    if compute_baseline:
        group[head_cxr] = group['pos'] * -1
    baseline_metric = [x[:-1] for x in metrics_list if x.endswith('*')]
    model_metric = [x for x in metrics_list if not x.endswith('*')]
    if len(baseline_metric) > 0:  # for baseline metric
        group = group.sort_values('weight', kind='mergesort', ascending=False)
        for metric in baseline_metric:
            k = 999999999
            if '@' in metric:
                metric_name, k = metric.split('@')
                k = int(k)
            else:
                metric_name = metric

            exec("m = {metric_name}({k},{s_score})".format(
                metric_name=metric_name, k=k, s_score='\'weight\''))
            metric_result_gp.append(locals()['m'](group))

    if len(model_metric) > 0:  # for model metric
        try:
            group = group.sort_values(
                'weight', kind='mergesort', ascending=False)
        except:
            pass
        group = group.sort_values(head_cxr, kind='mergesort', ascending=False)

        for metric in model_metric:
            k = 999999999
            if '@' in metric:
                metric_name, k = metric.split('@')
                k = int(k)
            else:
                metric_name = metric

            exec("m = {metric_name}({k},{s_score})".format(
                metric_name=metric_name, k=k, s_score='\''+head_cxr+'\''))
            metric_result_gp.append(locals()['m'](group))  #*len(group))
            #print('%s\t%s\t%f' % (metric, group.loc[group.index[0], 'pvid'], metric_result_gp[-1]))

    if random.random() > 0.999999:
        name = str(uuid.uuid4())
        # feature_list = ['query','uuid','pvid','price','weight']
        group = pd.DataFrame(group)
        group.to_csv(
            path_or_buf=case_output + '/' + name + '_data',
            sep='\t',
            header=True,
            index=False,
            encoding='utf-8')
        with fs_utils.fopen(case_output + '/' + name + '_metric', 'w') as file:
            # label,query,uuid,pvid,price,weight
            print(str(group) + '\n', file=file)
            print(",".join([str(x) for x in metrics_list]) + '\n', file=file)
            print(
                ",".join([str(x) for x in metric_result_gp]) + '\n', file=file)

    return metric_result_gp


def cal_metrics(data, groupby_fea, metrics_list, output_dir, compute_baseline,
                groupby_fea_2, head_cxr):
    metrics_list = [x.strip() for x in metrics_list if x.find('pos_dis') == -1]

    # divide the list into global and group
    group_metric_name_list = [
        x.strip() for x in metrics_list if not x.strip().startswith('#')
    ]
    group_metric_name_list = [
        x for x in group_metric_name_list if x.endswith('*')
    ] + [x for x in group_metric_name_list if not x.endswith('*')]

    global_metric_name_list = [
        x.strip()[1:] for x in metrics_list if x.strip().startswith('#')
    ]
    global_metric_name_list = [
        x for x in global_metric_name_list if x.endswith('*')
    ] + [x for x in global_metric_name_list if not x.endswith('*')]

    if len(group_metric_name_list) != 0:
        process_num = 25
        case_output_dir = output_dir + '/case_output'
        if not fs_utils.exists(case_output_dir):
            fs_utils.mkdir(case_output_dir)

        if not groupby_fea_2:
            group_lst = (group for name, group in data.groupby(groupby_fea)
                         if group['label'].any())
            handler = ProcessHandler(
                group_lst,
                eval_metric,
                process_num,
                metrics_list=group_metric_name_list,
                case_output=case_output_dir,
                compute_baseline=compute_baseline,
                head_cxr=head_cxr)
            handler.run()
            result = handler.get_result()
            group_metric_result = pd.DataFrame(
                result, columns=group_metric_name_list)
            # get the mean of metric
            group_mean_value = group_metric_result.mean()
        else:
            group_mean_value_all_user_df = []
            group_mean_value_mean_all_user_serise = []
            for u_name, u_group in data.groupby(groupby_fea_2.split(',')):
                if u_group['label'].any():
                    result_obj = pool.map_async(
                        partial(
                            eval_metric,
                            metrics_list=group_metric_name_list,
                            case_output=case_output_dir,
                            compute_baseline=compute_baseline),
                        (group for name, group in u_group.groupby(groupby_fea)
                         if group['label'].any()))
                    result = result_obj.get()
                    group_metric_result = pd.DataFrame(
                        result, columns=group_metric_name_list)

                    # get the mean of metric
                    if len(group_metric_result) > 1:
                        u_group_mean_value = group_metric_result.mean()
                        u_group_mean_value.name = 0
                        group_mean_value_mean_all_user_serise.append(
                            u_group_mean_value)
                    else:
                        u_group_mean_value = group_metric_result
                        group_mean_value_all_user_df.append(u_group_mean_value)
            group_mean_value_all_user_df.append(
                pd.DataFrame(group_mean_value_mean_all_user_serise))
            group_mean_value_all_user = pd.concat(group_mean_value_all_user_df)
            group_mean_value = group_mean_value_all_user.mean()

        # output information
        m_index = list(group_mean_value.index)
        print("\n")
        print("\t".join(m_index), file=sys.stdout)
        print(
            "\t".join(list(str(float(x)) for x in group_mean_value)),
            file=sys.stdout)

    if len(global_metric_name_list) != 0:
        # calculate the global list
        print("global_metric_name_list:" + str(global_metric_name_list))
        global_metric_result = eval_metric(
            data,
            global_metric_name_list,
            case_output=output_dir + '/case_output',
            compute_baseline=compute_baseline)

        print('#' + '\t#'.join(global_metric_name_list))
        print('\t'.join(str(x) for x in global_metric_result))

    fpath = fs_utils.path_join(output_dir, "metrics")
    with fs_utils.fopen(fpath, 'w') as file:
        if len(group_metric_name_list) != 0:
            print("\t".join(m_index), file=file)
            print("\t".join(list(str(x) for x in group_mean_value)), file=file)

        if len(global_metric_name_list) != 0:
            print('#' + '\t#'.join(global_metric_name_list), file=file)
            print('\t'.join(str(x) for x in global_metric_result), file=file)


def cal_pos_dis(data, groupby_fea, metrics_list, output_dir):
    # divide the list into global and group
    metrics_list = [x.strip() for x in metrics_list if x.find('pos_dis') != -1]
    group_metric_name_list = [
        x.strip() for x in metrics_list if not x.strip().startswith('#')
    ]
    group_metric_name_list = [
        x for x in group_metric_name_list if x.endswith('*')
    ] + [x for x in group_metric_name_list if not x.endswith('*')]

    global_metric_name_list = [
        x.strip()[1:] for x in metrics_list if x.strip().startswith('#')
    ]
    global_metric_name_list = [
        x for x in global_metric_name_list if x.endswith('*')
    ] + [x for x in global_metric_name_list if not x.endswith('*')]

    if len(group_metric_name_list) != 0:
        pool = Pool(processes=25)

        result_obj = pool.map_async(
            partial(eval_pos_dis, metrics_list=group_metric_name_list),
            (group for name, group in data.groupby(groupby_fea)
             if group['label'].any()))
        result = result_obj.get()

        pos_map_list = []
        for i in range(len(group_metric_name_list)):
            pos_map_list.append({})

        for group_result in result:
            for i in range(len(group_metric_name_list)):
                for pos in group_result[i]:
                    pos_map_list[i][pos] = pos_map_list[i].get(pos, 0) + 1

        print(
            "pos_dis_group:\t" + "\t".join(group_metric_name_list),
            file=sys.stdout)

    if len(global_metric_name_list) != 0:
        # calculate the global list
        print("global_metric_name_list:" + str(global_metric_name_list))
        global_metric_result = eval_pos_dis(data, global_metric_name_list)

        print(
            'pos_dis_global:\t#' + '\t#'.join(global_metric_name_list),
            file=sys.stdout)

    for i in range(len(group_metric_name_list)):
        with fs_utils.fopen(
                output_dir + '/' + group_metric_name_list[i].replace('*', '_'),
                'w') as file:
            for key in sorted(pos_map_list[i]):
                print(str(key) + "\t" + str(pos_map_list[i][key]), file=file)

    for i in range(len(global_metric_name_list)):
        with fs_utils.fopen(
                output_dir + '/_' + global_metric_name_list[i].replace(
                    '*', '_'), 'w') as file:
            print(
                '\n'.join(str(x) for x in global_metric_result[i]), file=file)


def eval_pos_dis(group, metrics_list):
    metric_result_gp = []

    baseline_metric = [x[:-1] for x in metrics_list if x.endswith('*')]
    model_metric = [x for x in metrics_list if not x.endswith('*')]
    # tricky for auc stable unstable sort
    # kind : {‘quicksort’, ‘mergesort’, ‘heapsort’}, optional
    # only mergesort is stable
    if len(baseline_metric) > 0:  # for baseline metric
        group = group.sort_values('weight', kind='mergesort', ascending=False)
        for metric in baseline_metric:
            k = 999999999
            if '@' in metric:
                metric_name, k = metric.split('@')
                k = int(k)
            else:
                metric_name = metric

            exec("m = {metric_name}({k},{s_score})".format(
                metric_name=metric_name, k=k, s_score='\'weight\''))
            metric_result_gp.append(locals()['m'](group))

    if len(model_metric) > 0:  # for model metric
        try:
            group = group.sort_values(
                'weight', kind='mergesort', ascending=False)
        except:
            pass
        group = group.sort_values(s_score, kind='mergesort', ascending=False)

        for metric in model_metric:
            k = 999999999
            if '@' in metric:
                metric_name, k = metric.split('@')
                k = int(k)
            else:
                metric_name = metric
            exec("m = {metric_name}({k},{s_score})".format(
                metric_name=metric_name, k=k, s_score='\'probabilities\''))
            metric_result_gp.append(locals()['m'](group))

    return metric_result_gp


def eval_one_model(scored_test_file, eval_out_dir, groupby_fea, \
        metrics_list, compute_baseline, groupby_fea_2, heads=[s_score]):
    assert isinstance(metrics_list, list)
    with fs_utils.fopen(scored_test_file) as table_file:
        dataset = pd.io.parsers.read_table(
            table_file, header=0, encoding='utf-8',quoting=3)
        assert all(t in dataset.columns.values for t in
            ['label', 'pvid', 'price']+list(heads))
    if len(dataset) == 0:
        print("load data failed, quit")
        exit(1)

    if not fs_utils.exists(eval_out_dir):
        fs_utils.mkdir(eval_out_dir)

    for head_cxr in heads:
        cal_metrics(dataset, groupby_fea, metrics_list,
            eval_out_dir + '/' + head_cxr,
            compute_baseline, groupby_fea_2, head_cxr)
    cal_pos_dis(dataset, groupby_fea, metrics_list, eval_out_dir)

def preprocess_dataset(dataset, task_weights):
    # 线上计算merger score的计算逻辑 searcher/kernels/algo/kernels/gmp/p13n_dl_rank_utility.cc#L1339
    dataset['multi_task'] = (np.power(dataset['ctcvr'], task_weights[0])
                            + np.power(dataset['ctr'], task_weights[1])
                            + np.power(dataset['ctavr'], task_weights[2])
                            + np.power(np.log1p(dataset['price']), task_weights[3]))
    return dataset

def eval_multi_task_model(scored_test_file, eval_out_dir, groupby_fea, \
        metrics_list, compute_baseline, groupby_fea_2, task_weights, heads=[s_score]):
    assert isinstance(metrics_list, list)
    with fs_utils.fopen(scored_test_file) as table_file:
        dataset = pd.io.parsers.read_table(
            table_file, header=0, encoding='utf-8',quoting=3)
        assert all(t in dataset.columns.values for t in
            ['label', 'pvid', 'price']+list(heads))
    if len(dataset) == 0:
        print("load data failed, quit")
        exit(1)

    if not fs_utils.exists(eval_out_dir):
        fs_utils.mkdir(eval_out_dir)
    # remove log
    dataset = preprocess_dataset(dataset, task_weights)
    for head_cxr in heads:
        cal_metrics(dataset, groupby_fea, metrics_list,
            eval_out_dir + '/' + head_cxr,
            compute_baseline, groupby_fea_2, head_cxr)
    print('ctcvr weight: %s, ctr weight: %s, ctavr weight: %s, price weight: %s'%(
            task_weights[0], task_weights[1], task_weights[2], task_weights[3]))
    cal_metrics(dataset, groupby_fea, metrics_list,
        eval_out_dir + '/' + 'multi_task',
        compute_baseline, groupby_fea_2, 'multi_task')
    cal_pos_dis(dataset, groupby_fea, metrics_list, eval_out_dir)

def get_single_example(feature_id_map,
                       example,
                       prediction_result,
                       baseline=False):
    #print(example)
    label = (
        0 if feature_id_map['label'] not in example.features.feature else
        example.features.feature[feature_id_map['label']].int64_list.value[0])
    query = ("" if feature_id_map['query'] not in example.features.feature else
             ''.join([
                 b.decode('utf-8') for b in example.features.feature[
                     feature_id_map['query']].bytes_list.value
             ]))

    uid = ("" if feature_id_map['uid'] not in example.features.feature else
           example.features.feature[
               feature_id_map['uid']].bytes_list.value[0].decode('utf-8'))
    pvid = ("" if feature_id_map['pvid'] not in example.features.feature else
            example.features.feature[
                feature_id_map['pvid']].bytes_list.value[0].decode('utf-8'))
    price = (
        0 if feature_id_map['price'] not in example.features.feature else
        example.features.feature[feature_id_map['price']].float_list.value[0])
    pos = (0 if feature_id_map['pos'] not in example.features.feature else
           example.features.feature[feature_id_map['pos']].int64_list.value[0])
    catid = (
        0 if feature_id_map['catid'] not in example.features.feature else
        example.features.feature[feature_id_map['catid']].int64_list.value[0])
    prediction_uid = prediction_result['uid'].decode('utf-8')
    if prediction_result['uid'].decode('utf-8') != uid:
        print("error" + " w " + uid + " f:" +
              prediction_result['uid'].decode('utf-8'))
        return
    if prediction_result['pvid'].decode('utf-8') != pvid:

        print("pvid is not equal error" + pvid + " " +
              prediction_result['pvid'].decode('utf-8'))
        return

    fields = [
        label, query, uid, pvid, price, pos, catid,
        pos * -1.0 if baseline else prediction_result[s_score][0]
    ]
    return fields


def convert_prediction_for_metric_computation(config, output_file, baseline):

    prediction_result = pickle.load(
        fs_utils.fopen(config.prediction_file, 'rb'))

    fields = ['label', 'query', 'uid', 'pvid', 'price', 'pos', 'catid']
    feature_id_map = {
        feature_name: lw_utils.get_feature_key(
            getattr(config.feature_stats, feature_name), feature_name)
        for feature_name in fields
    }
    with fs_utils.fopen(output_file, 'w') as fp:
        record_iterator = tf.python_io.tf_record_iterator(
            config.test_dataset_files)
        fields.append(s_score)
        line = '\t'.join([fd for fd in fields])
        fp.write(line + '\n')

        count = 0
        for record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(record)
            result = get_single_example(feature_id_map, example,
                                        prediction_result[count], baseline)
            if not result:
                continue
            line = '\t'.join([str(fd) for fd in result])
            fp.write(line + '\n')
            count += 1


def join_tsv_file(prediction_tsv,
                  metrics_file,
                  prediction_header_map,
                  config,
                  tsv_header_file=None,
                  tsv_file_to_join=None,
                  fix_siamese_header=True):
    if not tsv_file_to_join:
        assert tsv_file_to_join == ''
        tsv_file_to_join = config.session_validate_tsv_with_header
    assert tsv_file_to_join
    fields = ['label', 'query', 'uid', 'pvid', 'price', 'pos', 'is_self_fea', 'is_jx_product', 'catid', 'hc_cid1s']
    utils.debuginfo(
        'tsv_file_to_join = %s, prediction_tsv = %s, metrics_file = %s' %
        (tsv_file_to_join, prediction_tsv, metrics_file))
    with fs_utils.fopen(tsv_file_to_join) as f, fs_utils.fopen(
            prediction_tsv) as g, fs_utils.fopen(metrics_file, 'w') as h:
        if tsv_header_file:
            with fs_utils.fopen(tsv_header_file) as j:
                input_header = j.readline().strip().split('\t')
        else:
            input_header = f.readline().strip().split('\t')
        if fix_siamese_header:

            def remove_suffix_a(t):
                assert not t.endswith('_b')
                return re.sub('_a$', '', t)

            input_header = list(map(remove_suffix_a, input_header))
        if config.session_validate_header_map:
            input_header_map = dict(
                t.split(':')
                for t in config.session_validate_header_map.split(','))
        else:
            input_header_map = {}
        input_header_indices = [
            input_header.index(input_header_map.get(t, t)) for t in fields
        ]
        pred_header = g.readline().strip().split('\t')
        pred_header_map = dict(
            t.split(':') for t in prediction_header_map.split(','))
        pred_header = [pred_header_map.get(t, t) for t in pred_header]
        h.write('\t'.join(fields + pred_header))
        for test_line, pred_line in zip(f, g):
            input_tmp = test_line.strip().split('\t')
            input_columns = [input_tmp[i] for i in input_header_indices]
            h.write('\n' + '\t'.join(input_columns + [pred_line.strip()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate_feature_stats')
    parser.add_argument(
        '-c',
        '--config',
        metavar='config.json',
        type=str,
        help='config file path; not needed if join_with_test_feature=False.')
    parser.add_argument(
        '-o',
        '--metrics_file',
        metavar='metric_temp',
        type=str,
        help='The tsv file with all the columns needed to compute metrics.')
    parser.add_argument(
        '-b',
        '--compute_baseline',
        metavar='metric_temp',
        type=parser_utils.str2bool,
        help='compute_baseline')
    parser.add_argument(
        '-t',
        '--join_with_test_tfrecord',
        type=parser_utils.str2bool,
        help='join tfrecord for evaluation',
        default=False)
    parser.add_argument(
        '--tsv_file_to_join',
        type=str,
        default=None,
        help='If not None, join '
        'features from an external tsv file.')
    parser.add_argument(
        '--tsv_header_file',
        type=str,
        default=None,
        help='If not None, '
        'tsv_file_to_join is assumed to be headerless.')
    parser.add_argument(
        '--prediction_tsv',
        type=str,
        default=None,
        help='tsv output from predict mode in run.py')
    parser.add_argument(
        '--prediction_header_map',
        type=str,
        default='score:probabilities',
        help=
        'Comma separated list of substitution rules for prediction headers.')
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='A comma separated list of metrics to compute.')
    parser.add_argument('--groupby_fea', type=str, default='pvid,query,uid')
    parser.add_argument(
        '--groupby_fea_2',
        type=str,
        default=None,
        help=
        'Comma separated list of features. Group data before calculating auc.'
        'eg:if value=uid, the data used to calculate auc would be group by uid first.'
    )
    args = parser.parse_args()
    if args.config:
        config = conf.parse_config(args.config, None)
        lw_utils.make_paths_absolute(config, config.data_dir)
        args.config = config
    assert not (args.join_with_test_tfrecord and args.tsv_file_to_join)
    if args.join_with_test_tfrecord:
        convert_prediction_for_metric_computation(config, args.metrics_file,
                                                  args.compute_baseline)
    if args.tsv_file_to_join is not None:
        join_tsv_file(
            args.prediction_tsv,
            args.metrics_file,
            args.prediction_header_map,
            args.config,
            args.tsv_header_file,
            args.tsv_file_to_join,
            fix_siamese_header=True)
    eval_one_model(args.metrics_file, 'eval', args.groupby_fea.split(','),
                   args.metrics and args.metrics.split(',') or metrics,
                   args.compute_baseline, args.groupby_fea_2)
