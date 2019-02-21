import os
import sys
import shutil
import copy
from glob import iglob
import pickle
import itertools
import datetime
import numpy as np

from tfmodel import *


def get_file_list(config, data_path, pattern, description):
    files = [y for x in os.walk(data_path) for y in iglob(os.path.join(x[0], pattern))]

    if len(files) == 0:
        print("No partitions found for %s data, exiting..." % description)
        sys.exit()

    print("Found %d%s partitions for %s data"
          % (len(files), " prebatched" if config.use_prebatched else "", description))
    if config.num_partitions:
        print("But only using %d due to num_partitions parameter" % config.num_partitions)
    files = files[:config.num_partitions]
    return files


def copy_temp_files(files, temp_dir):
    temp_files = []
    for file in files:
        target_file = os.path.split(file)[1]
        target_file = os.path.join(temp_dir, target_file)
        shutil.copy2(file, target_file)
        temp_files.append(target_file)
    return temp_files


def create_model(is_training, config, targets, input_data, lengths, dropout_keep_rate, masks=None, initial_state=None):
    if config.attention and config.attention_variant == "input":
        return AttentionModel(is_training=is_training, config=config, targets=targets, input_data=input_data, lengths=lengths, dropout_keep_rate=dropout_keep_rate, masks=masks, initial_state=initial_state)
    elif config.attention and config.attention_variant == "output":
        return AttentionOverOutputModel(is_training=is_training, config=config, targets=targets, input_data=input_data, lengths=lengths, dropout_keep_rate=dropout_keep_rate, masks=masks, initial_state=initial_state)
    elif config.attention and config.attention_variant == "keyvalue":
        return AttentionKeyValueModel(is_training=is_training, config=config, targets=targets, input_data=input_data, lengths=lengths, dropout_keep_rate=dropout_keep_rate, masks=masks, initial_state=initial_state)
    elif config.attention and config.attention_variant == "exlambda":
        return AttentionWithoutLambdaModel(is_training=is_training, config=config, targets=targets, input_data=input_data, lengths=lengths, dropout_keep_rate=dropout_keep_rate, masks=masks, initial_state=initial_state)
    elif config.attention and config.attention_variant == "baseline":
        return AttentionBaselineModel(is_training=is_training, config=config, targets=targets, input_data=input_data, lengths=lengths, dropout_keep_rate=dropout_keep_rate, masks=masks, initial_state=initial_state)
    else:
        return BasicModel(is_training=is_training, config=config, targets=targets, input_data=input_data, lengths=lengths, dropout_keep_rate=dropout_keep_rate, initial_state=initial_state)

def get_initial_state(model):
    state = []
    att_states = None
    att_ids = None
    att_counts = None
    for c, m in model.initial_state[0] if model.is_attention_model else model.initial_state:
        state.append((c.eval(), m.eval()))
    if model.is_attention_model:
        att_states = [s.eval() for s in list(model.initial_state[1])]
        att_ids = [s.eval() for s in list(model.initial_state[2])]
        att_counts = [s.eval() for s in list(model.initial_state[4])]

    return state, att_states, att_ids, att_counts

def construct_state_feed_dict(model, state, att_states, att_ids, att_counts):
    feed_dict = {}
    for i, (c, m) in enumerate(model.initial_state[0]) if model.is_attention_model else enumerate(model.initial_state):
        feed_dict[c], feed_dict[m] = state[i]

    if model.is_attention_model:
        for i in range(len(model.initial_state[1])):
            feed_dict[model.initial_state[1][i]] = att_states[i]
            feed_dict[model.initial_state[2][i]] = att_ids[i]
            feed_dict[model.initial_state[4][i]] = att_counts[i]

    return feed_dict

def extract_results(state_eval, model):
    state_end = len(model.final_state[0])*2 if model.is_attention_model else len(state_eval)
    state_flat = state_eval[:state_end]
    state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

    num_att_states = len(model.final_state[1]) if model.is_attention_model else 0
    att_states = state_eval[state_end:state_end + num_att_states] if model.is_attention_model else None
    att_ids = state_eval[state_end+num_att_states:state_end + num_att_states*2] if model.is_attention_model else None
    alpha_states = state_eval[state_end+num_att_states*2:state_end+num_att_states*3] if model.is_attention_model else None
    att_counts = state_eval[state_end+num_att_states*3:state_end+num_att_states*4]
    lambda_state = state_eval[-1] if model.is_attention_model else None

    return state, att_states, att_ids, alpha_states, att_counts, lambda_state

def get_evals_for_state(model):
    extra_evals = []
    for c, m in model.final_state[0] if model.is_attention_model else model.final_state:
        extra_evals.append(c)
        extra_evals.append(m)

    if model.is_attention_model:
        extra_evals.extend(model.final_state[1] + model.final_state[2] + model.final_state[3] + model.final_state[4])
        extra_evals.append(model.final_state[5])

    return extra_evals

def attention_masks(attns, masks, length):
    lst = []
    if "full" in attns:
        lst.append(np.ones([1, length]))
    if "identifiers" in attns:
        lst.append(masks[:, 0:length] if len(masks.shape) == 2 else np.reshape(masks[0:length], [1, length]))

    return np.transpose(np.concatenate(lst)) if lst else np.zeros([0, length])


class FlagWrapper:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

    def __getattr__(self, name):
        return self.__dict__['__flags'][name]


def copy_flags(flags):
    dict_copy = copy.copy(flags.__dict__)
    return FlagWrapper(dict_copy)


def identity_map(x):
    return x


def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


# from
#   http://stackoverflow.com/questions/33759623/tensorflow-how-to-restore-a-previously-saved-model-python
def save_model(saver, sess, path, model, config):
    if not os.path.exists(path):
        os.makedirs(path)

    now = datetime.now().strftime("%Y-%m-%d--%H-%M--%f")
    out_path = os.path.join(path, now + "/")

    tf.train.write_graph(model.graph.as_graph_def(), out_path, 'model.pb', as_text=False)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    saver.save(sess, os.path.join(out_path, "model.tf"))

    latest_path = os.path.join(path, "latest")
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(now, latest_path)
    return out_path


def load_model(sess, path):
    load_variables(sess, os.path.join(path, "model.tf"), tf.trainable_variables())


def load_variables(session, path, variables):
    saver = tf.train.Saver(variables)
    saver.restore(session, path)
