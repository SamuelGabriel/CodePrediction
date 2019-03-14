#!/usr/bin/env python
"""
Usage:
    evaluate.py [options] TRAINED_MODEL TEST_DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
    --run-analysis                   Enable running analysis to retrieve example inferences and more statistics. This will print a lot of things especially example predictions for one file in each batch and statistics for each batch, as well as global counts in the end. [default: False]
"""
import gzip
import json
import os
import pickle
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dpu_utils.mlutils.vocabulary import Vocabulary
from more_itertools import chunked

import tensorflow as tf
import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug

from model import Model

def print_tokens_and_others(encoded, others, vocab):
    l = 100
    b = 0
    i = -1
    to_print = []
    for e in encoded:
        to_print.append([vocab.get_name_for_id(t) for t in e[b:b+l,-1].flatten()])
    for o in others:
        o = o[b:b+l,-1]
        to_print.append(list(o.flatten()))
    line_size = 15
    to_print = [list(chunked(l, line_size)) for l in to_print]
    print('xxxxxxxx')
    for ls in zip(*to_print):
        print('-------')
        for l in ls:
            print('\t'.join(str(e) for e in l))

def large_lambda_1(s: tuple, tokens: np.ndarray, token_lens: np.ndarray, predictions: np.ndarray, probs: np.ndarray, vocab: Vocabulary, file_batch: int, lambdas: np.ndarray=None, num_correct_ids: int=0, num_ids: int=1):
    state, att_states, att_ids, alpha_states, att_counts, lambda_state = s
    tokens = np.transpose(tokens)
    att_states = np.squeeze(att_states, 2)
    alpha_states = np.squeeze(alpha_states, 2)
    max_atts = alpha_states.max(-1)
    unk_i = vocab.token_to_id[vocab.get_unk()]
    seq_mask = np.transpose(np.tile(np.arange(tokens.shape[0]),[tokens.shape[1],1])) < np.tile(token_lens, [tokens.shape[0],1])
    n_unk = (tokens == unk_i)[seq_mask].sum()
    n_non_pad = seq_mask.sum()
    unk_p = n_unk/n_non_pad
    print('File Batch: ', file_batch)
    print('some tokens: ', tokens[seq_mask][:10])
    if max_atts is not None:
        max_copy_prob = max_atts * lambda_state[:,1]
        average_lambda = np.ma.array(lambda_state[:,1], mask=tokens[-1] == 0).mean()
        print('average copy lambda: ', average_lambda)
        print('max copy lambda: ', lambda_state[:,1].max())
        print('max copy token prob: ',np.amax(max_copy_prob))
        attention_stats = [average_lambda]
    else:
        attention_stats = []
    print_tokens_and_others((tokens, predictions), (probs,), vocab)
    print('id acc: ', num_correct_ids/num_ids)
    am =  np.unravel_index(np.argmin(predictions), predictions.shape)
    print(predictions[am], tokens[am[0]], tokens[am])
    return [np.min(predictions), np.mean(predictions),n_unk,n_non_pad, num_correct_ids, num_ids]+attention_stats

def aggregator(outputs: List[List[np.ndarray]]):
    a_outputs = np.array(outputs)
    print('Averages: ', a_outputs.mean(0))
    print('Maxs: ', a_outputs.max(0))
    print('Mins: ', a_outputs.min(0))



def restore(path: str) -> Model:
    with gzip.open(path) as f:
        saved_data = pickle.load(f)
    model = Model(saved_data['hyperparameters'], saved_data['modelparameters'].__dict__, saved_data.get('run_name'))
    model.metadata.update(saved_data['metadata'])
    model.init()

    variables_to_initialize = []
    with model.sess.graph.as_default():
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in sorted(model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), key=lambda v: v.name):
                used_vars.add(variable.name)
                if variable.name in saved_data['weights']:
                    # print('Initializing %s from saved value.' % variable.name)
                    restore_ops.append(variable.assign(saved_data['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in sorted(saved_data['weights']):
                if var_name not in used_vars:
                    if var_name.endswith('Adam:0') or var_name.endswith('Adam_1:0') or var_name in ['beta1_power:0', 'beta2_power:0']:
                        continue
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            model.sess.run(restore_ops)
            print('number of parameters: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    return model


def run(arguments) -> None:
    model = restore(arguments['TRAINED_MODEL'])
    test_data = model.load_data_from_dir(arguments['TEST_DATA_DIR'],
                                         max_num_files=arguments.get('--max-num-files'))
    returns = model.test(test_data, analysis_fun=large_lambda_1 if arguments.get('--run-analysis') else None)
    if arguments.get('--run-analysis'):
        aggregator(returns)
    

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
