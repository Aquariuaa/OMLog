import pandas
import random
from collections import Counter


def split_meta_task(batch_data, num_tasks):
    '''
    for split batch data to meta-task
    :param batch_data:
    :param num_tasks:
    :return:
    '''
    querys = []
    keys_to_split = list(batch_data.keys())
    assert all(isinstance(batch_data[key], list) for key in keys_to_split), "all keys must be list"
    query_len = int(len(batch_data[keys_to_split[0]]))/num_tasks
    query_num = [int(0 + i * query_len) for i in range(num_tasks+1)]
    for q in range(len(query_num)-1):
        query_dict = {}
        for key in keys_to_split:
            query_dict[key] = batch_data[key][query_num[q]:query_num[q+1]]
        querys.append(query_dict)
    return querys

def filter_normal_by_ND(input_dict, pred_key):
    '''
    for filter normal samples from batch
    :param input_dict:
    :param pred_key:
    :return:
    '''

    preds = input_dict[pred_key]
    filtered_dict = {}
    for key, value in input_dict.items():
        if key != pred_key and isinstance(value, list) and len(value) == len(preds):
            filtered_value = [v for v, p in zip(value, preds) if p]
            if filtered_value:
                filtered_dict[key] = filtered_value
    filtered_dict[pred_key] = [p for p in preds if p]
    return filtered_dict

def fix_ini_support(batch_data, support_memory):
    '''
    get support set for the first round
    :param batch_data:
    :param support_memory:
    :return:
    '''
    hc_batch = filter_normal_by_ND(batch_data, 'autoencoder_pred')
    ini_support = {}
    for key in hc_batch.keys():
        ini_support[key] = hc_batch[key][:support_memory]
    return ini_support


def merge_support(support_pre, query, support_memory, qe):
    '''
    Merging into a meta-task
    :param support_pre:
    :param query:
    :param support_memory:
    :param qe:
    :return:
    '''
    train_num = Counter(query['autoencoder_pred'])[True]
    elem_num = len(query['autoencoder_pred'])
    if train_num < support_memory:
        support_pre = filter_normal_by_ND(support_pre, 'autoencoder_pred')
        merged_data = {}
        for key in query.keys():
            if qe % 2 == 0:
                merged_data[key] = query[key] + support_pre[key][:(support_memory - train_num)]
            else:
                merged_data[key] = query[key] + support_pre[key][-(support_memory - train_num):]
    else:
        merged_data = query
    return merged_data

def offline_save(history_train_set, save_memory):
    merged = {}
    for d in history_train_set:
        for k, v in d.items():
            if k in merged:
                if isinstance(merged[k], list) and isinstance(v, list):
                    merged[k].extend(v)
                elif isinstance(merged[k], list):
                    print(f"Warning: Overwriting list with non-list value for key {k}")
                    merged[k] = v
                elif isinstance(v, list):
                    merged[k] = [merged[k]] + v
                else:
                    merged[k] = v
            else:
                merged[k] = v
            merged[k] = merged[k][-save_memory:]
    # print("merged", len(merged["session_key"]))
    return merged

