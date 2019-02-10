import gzip
import os
import pickle
import time
from glob import glob
from collections import defaultdict, OrderedDict, Counter, deque
from typing import List, Dict, Any, Iterable, Tuple, Optional, Union, NamedTuple
from itertools import islice

import numpy as np
import tensorflow as tf
from more_itertools import chunked
from dpu_utils.mlutils.vocabulary import Vocabulary
from graph_pb2 import Graph
from graph_pb2 import FeatureNode
import sys
import utils
print(sys.version)

from tensorflow.python import debug as tf_debug



class ModelParams(object):
    def __init__(self):
        self.lambda_type = ['state', 'att', 'input'] # Method to calculate lambda, possible values are fixed, state, att, input.
        # state, att and input can be combined or also just be 'fixed'
        self.max_grad_norm = 5 # Maximum norm for gradients
        self.num_layers = 1 # Number of LSTM layers
        self.hidden_size = 200 # Size of hidden state 
        self.num_samples = 0 # Number of samples for sampled softmax
        self.max_attention = 20 # Maximum size of attention matrix
        self.attention = ['1'] * 1 # len is how many attentions to use # eiher 'full' or 'identifiers'
        self.attention_variant = 'input' # which kind of attention to use 'output' or 'input'
        self.masked_copying = True # should the attention be given a mask?

    def set_hyperparameters(self, vocab_size, seq_length, batch_size):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size


DATA_FILE_EXTENSION = 'proto'


LoadedSamples = Dict[str, np.ndarray]


def get_data_files_from_directory(data_dir: str, max_num_files: Optional[int]=None) -> List[str]:
    files = glob(os.path.join(data_dir, '**/*.%s' % DATA_FILE_EXTENSION), recursive=True)
    if max_num_files:
        files = sorted(files)[:int(max_num_files)]
    else:
        files = list(files)
    np.random.shuffle(files)
    return files


class Model(object):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
                'optimizer': 'Adam',  # One of "SGD", "RMSProp", "Adam"
                'dropout_keep_rate': 0.9,
                'learning_rate': 0.01,
                'learning_rate_decay': 1.,
                'momentum': 0.85,
                'gradient_clip': 5,
                'max_len': 100,
                'batch_size': 30,
                'max_epochs': 500,
                'patience': 5,
               }

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 modelparameters: Dict[str, Any],
                 model_save_dir: Optional[str]=None) -> None:
        # start with default hyper-params and then override them
        self.hyperparameters = self.get_default_hyperparameters()
        self.hyperparameters.update(hyperparameters)
        self.modelparameters = ModelParams()
        for key,value in modelparameters.items():
            setattr(self.modelparameters,key,value)
        print(self.hyperparameters)
        print(self.modelparameters.__dict__)

        self.__metadata = {}  # type: Dict[str, Any]
        self.__placeholders = {}  # type: Dict[str, Union[tf.placeholder, tf.placeholder_with_default]]
        self.__ops = {}  # type: Dict[str, Any]

        self.__run_name = hyperparameters['run_id']
        self.__model_save_dir = model_save_dir or "."
        self.__sess = tf.Session(graph=tf.Graph())# tf_debug.LocalCLIDebugWrapperSession(tf.Session(graph=tf.Graph()))# tf.Session(graph=tf.Graph())#, config=tf.ConfigProto(log_device_placement=True))

    @property
    def metadata(self):
        return self.__metadata

    @property
    def placeholders(self):
        return self.__placeholders

    @property
    def ops(self):
        return self.__ops

    @property
    def sess(self):
        return self.__sess

    @property
    def run_name(self):
        return self.__run_name

    def save(self, path: str) -> None:
        variables_to_save = list(set(self.__sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        weights_to_save = self.__sess.run(variables_to_save)
        weights_to_save = {var.name: value
                           for (var, value) in zip(variables_to_save, weights_to_save)}

        data_to_save = {
                         "model_type": type(self).__name__,
                         "hyperparameters": self.hyperparameters,
                         "modelparameters": self.modelparameters,
                         "metadata": self.metadata,
                         "weights": weights_to_save,
                         "run_name": self.run_name,
                       }

        with gzip.GzipFile(path, 'wb') as outfile:
            pickle.dump(data_to_save, outfile)

    def init(self):
        """
        Initialise the actual ML model, i.e., build the TF computation graph.
        """
        with self.__sess.graph.as_default():
            self.placeholders['dropout_keep_rate'] = \
                tf.placeholder(dtype=tf.float32, shape=(), name='dropout_keep_rate')
            self.placeholders['tokens'] = \
                tf.placeholder(dtype=tf.int64, shape=[self.hyperparameters['batch_size'], self.hyperparameters['max_len']], name='tokens')
            self.placeholders['tokens_lengths'] = \
                tf.placeholder(dtype=tf.int64, shape=[self.hyperparameters['batch_size']], name='tokens_lengths')
            self.placeholders['masks'] = \
                tf.placeholder(dtype=tf.bool, shape=[self.hyperparameters['batch_size'], self.hyperparameters['max_len']], name='masks')
            self.placeholders['global_step'] = \
                tf.placeholder(dtype=tf.int64, shape=(), name='global_step')

            self.make_model()
            self.__make_training_step()

    def make_model(self) -> None:
        """
        Create the core model.

        Note: This has to create self.ops['loss'] (a scalar).
        """
        tokens = tf.transpose(self.placeholders['tokens'])
        masks = tf.transpose(self.placeholders['masks'])
        tokens_lengths = self.placeholders['tokens_lengths']
        dropout_keep_rate = self.placeholders['dropout_keep_rate']
        #TODO# Insert your model here, creating self.ops['loss']
        m_params = self.modelparameters
        m_params.set_hyperparameters(vocab_size=len(self.metadata['token_vocab']), seq_length=self.hyperparameters['max_len']-1, batch_size=self.hyperparameters['batch_size'])
        
        if m_params.attention:
            if m_params.masked_copying:
                masks = tf.stack([masks[:-1,:]]*len(m_params.attention), axis=2)
            else:
                masks = tf.ones([self.hyperparameters['max_len']-1, self.hyperparameters['batch_size'], len(m_params.attention)])
        else:
            masks = None

        model = utils.create_model(True, m_params, tokens[1:,:], tokens[:-1,:], tokens_lengths, dropout_keep_rate, masks=masks)

        mask = tf.transpose(tf.sequence_mask(self.placeholders['tokens_lengths'], self.hyperparameters['max_len'], dtype=tf.int64))
        # Mask is not active for training, but I think it is the same in the model
        self.ops['loss'] = model.cost
        predictions = tf.reshape(tf.argmax(model.predict, -1), [m_params.seq_length, m_params.batch_size])
        self.ops['num_correct_tokens'] = tf.reduce_sum(tf.cast(tf.equal(predictions, tokens[1:,:]), tf.int64) * mask[1:, :])

    def __make_training_step(self) -> None:
        """
        Constructs self.ops['train_step'] from self.ops['loss'] and hyperparameters.
        """
        optimizer_name = self.hyperparameters['optimizer'].lower()
        learning_rate = tf.train.exponential_decay(self.hyperparameters['learning_rate'],
                                                   self.placeholders['global_step'],
                                                   1, self.hyperparameters['learning_rate_decay'],
                                                   staircase=True)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.hyperparameters['learning_rate'],
                                                  decay=self.hyperparameters['learning_rate_decay'],
                                                  momentum=self.hyperparameters['momentum'])
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception('Unknown optimizer "%s".' % (self.hyperparameters['optimizer']))

        # Calculate and clip gradients
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.ops['loss'], trainable_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hyperparameters['gradient_clip'])
        pruned_clipped_gradients = []
        for (gradient, trainable_var) in zip(clipped_gradients, trainable_vars):
            if gradient is None:
                continue
            pruned_clipped_gradients.append((gradient, trainable_var))
        self.ops['train_step'] = optimizer.apply_gradients(pruned_clipped_gradients)

    def load_data_file(_, file_path: str) -> Iterable[Tuple[str, bool]]:
        """
        Load a single data file, returning token streams.

        Args:
            file_path: The path to a data file.

        Returns:
            Iterable of lists of strings, each a list of tokens observed in the data.
        """
        with open(file_path, "rb") as f:
            g = Graph()
            g.ParseFromString(f.read())
            v = [(n.contents.lower(), n.type == FeatureNode.IDENTIFIER_TOKEN) for n in g.node
                    if n.type in
                        [FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN]
                    ]
            return v


    def load_metadata_from_dir(self, data_dir: str, max_num_files: Optional[int]=None) -> None:
        """
        Compute model metadata such as a vocabulary.

        Args:
            data_dir: Directory containing data files.
            max_num_files: Maximal number of files to load.

        Note: This populates the model.metadata dictionary
        """
        data_files = get_data_files_from_directory(data_dir, max_num_files)
        tokens = Counter(t for f in data_files for t, _ in self.load_data_file(f))
                
        self.__metadata = {'token_vocab': Vocabulary.create_vocabulary(tokens, max_size=5000)}

    def load_data_from_raw_sample_sequences(self, token_seqs: Iterable[Tuple[str, bool]]) -> LoadedSamples:
        """
        Load and tensorise data.

        Args:
            token_seqs: Sequences of tokens to load samples from.

        Returns:
            The loaded data, as a dictionary mapping names to numpy arrays.
        """
        loaded_data = {
            "tokens": [],
            "tokens_lengths": [],
            "masks": []
            }  # type: Dict[str, List[Any]]
        
        vocab = self.metadata['token_vocab']
        current_chunk = []
        sub_chunk_steps = 4
        token_seqs_iterator = iter(token_seqs)
        while True:
            s = list(islice(token_seqs_iterator, self.hyperparameters['max_len']//sub_chunk_steps))
            current_chunk += s
            current_chunk = current_chunk[-self.hyperparameters['max_len']:]
            if not s:
                break
            loaded_data['tokens_lengths'].append(len(current_chunk))
            loaded_data['tokens'].append(vocab.get_id_or_unk_multiple([e for e, m in current_chunk], pad_to_size=self.hyperparameters['max_len']))
            m = [m for e, m in current_chunk]
            loaded_data['masks'].append(m + (self.hyperparameters['max_len']-len(m))*[np.nan])

        """
        n_slice = []
        for t in token_seqs:
            n_slice += [t]
            n_slice = n_slice[:self.hyperparameters['max_len']]
            ids = vocab.get_id_or_unk_multiple(n_slice, pad_to_size=self.hyperparameters['max_len'])
            loaded_data['tokens_lengths'].append(len(n_slice))
            loaded_data['tokens'].append(ids)
        """
        

        # Turn into numpy arrays for easier slicing later:
        assert (len(loaded_data['tokens']) == len(loaded_data['tokens_lengths'])) and (len(loaded_data['tokens']) == len(loaded_data['masks'])), \
            "Loaded 'tokens' and 'tokens_lengths' lists need to be aligned and of" \
            + "the same length!"
        loaded_data['tokens'] = np.array(loaded_data['tokens'])
        loaded_data['masks'] = np.array(loaded_data['masks'])
        loaded_data['tokens_lengths'] = np.array(loaded_data['tokens_lengths'])
        return loaded_data

    def load_data_from_dir(self, data_dir: str, max_num_files: Optional[int]=None) -> LoadedSamples:
        data_files = get_data_files_from_directory(data_dir, max_num_files)
        return self.load_data_from_raw_sample_sequences(em
                                                        for data_file in data_files
                                                        for em in self.load_data_file(data_file))

    def __split_data_into_minibatches(self,
                                      data: LoadedSamples,
                                      epoch_number: int,
                                      is_train: bool,
                                      drop_incomplete_final_minibatch: bool=True) \
            -> Iterable[Tuple[int, Dict[tf.Tensor, Any]]]:
        """
        Take tensorised data and chunk into feed dictionaries corresponding to minibatches.

        Args:
            data: The tensorised input data.
            is_train: Flag indicating if we are in train mode (which causes shuffling and the use of dropout)
            drop_incomplete_final_minibatch: If True, all returned minibatches will have the configured size
             and some examples from data may not be considered at all. If False, the final minibatch will
             be shorter than the configured size.

        Returns:
            Iterable sequence of pairs of
              (1) Number of samples in the batch
              (2) A feed dict mapping placeholders to values,
        """
        total_num_samples = len(data['tokens'])
        indices = np.arange(total_num_samples)
        if is_train:
            np.random.shuffle(indices)
        
        batch_size = self.hyperparameters['batch_size']
        for chunked_indices in chunked(indices, n=batch_size):
            if drop_incomplete_final_minibatch and len(chunked_indices) < batch_size:
                continue

            feed_dict = {
                self.placeholders['dropout_keep_rate']: self.hyperparameters['dropout_keep_rate'] if is_train else 1.0,
                self.placeholders['tokens']: data['tokens'][chunked_indices],
                self.placeholders['tokens_lengths']: data['tokens_lengths'][chunked_indices],
                self.placeholders['masks']: data['masks'][chunked_indices],
                self.placeholders['global_step']: epoch_number
            }

            yield len(chunked_indices), feed_dict

    def __run_epoch_in_batches(self, data: LoadedSamples, epoch_name: str, is_train: bool, epoch_number: int =0) -> Tuple[float, float]:
        """
        Args:
            data: The loaded data to run the model on.
            epoch_name: Name for the epoch, e.g., "Train Epoch 1".
            is_train: Flag indicating if we should be performing a training step.

        Returns:
            Average loss over all samples in this epoch and accuracy of predictions.
        """
        epoch_loss, epoch_correct_predictions, epoch_total_tokens = 0.0, 0.0, 0.0
        num_samples_so_far, num_total_samples = 0, len(data['tokens'])
        data_generator = self.__split_data_into_minibatches(data, epoch_number, is_train=is_train)
        epoch_start = time.time()
        for minibatch_counter, (samples_in_batch, batch_data_dict) in enumerate(data_generator):
            print("%s: Batch %5i. Processed %i/%i samples. Loss so far: %.4f.  Acc. so far: %.2f%%  "
                    % (epoch_name, minibatch_counter,
                       num_samples_so_far, num_total_samples,
                       epoch_loss / max(num_samples_so_far, 1),
                       epoch_correct_predictions / max(epoch_total_tokens, 1) * 100),
                    flush=True,
                    end="\r")
            ops_to_run = {'loss': self.__ops['loss']}
            if is_train:
                ops_to_run['train_step'] = self.__ops['train_step']
            if 'num_correct_tokens' in self.__ops:
                ops_to_run['num_correct_tokens'] = self.__ops['num_correct_tokens']
            op_results = self.__sess.run(ops_to_run, feed_dict=batch_data_dict)
            num_samples_so_far += samples_in_batch
            assert not np.isnan(op_results['loss'])
            epoch_loss += op_results['loss']
            if 'num_correct_tokens' in self.__ops:
                epoch_total_tokens += np.sum(batch_data_dict[self.placeholders['tokens_lengths']])
                epoch_correct_predictions += op_results['num_correct_tokens']
        used_time = time.time() - epoch_start
        print("\r\x1b[K  Epoch %s took %.2fs [processed %s samples/second]"
              % (epoch_name, used_time, int(num_samples_so_far/used_time)))
        return epoch_loss / num_samples_so_far, epoch_correct_predictions / max(epoch_total_tokens, 1) * 100

    def train(self, train_data: LoadedSamples, valid_data: LoadedSamples) -> str:
        """
        Train model with early stopping.

        Args:
            train_data: Tesorised data to train on.
            valid_data: Tensorised data to validate the trained model.
        
        Returns:
            Path to saved model.
        """
        model_path = os.path.join(self.__model_save_dir,
                                  "%s_model_best.pkl.gz" % (self.hyperparameters['run_id'],))
        with self.__sess.as_default():
            init_op = tf.variables_initializer(self.__sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self.__sess.run(init_op)
            self.save(model_path)
            best_val_loss = float("inf")
            epoch_number, no_improvement_counter = 0, 0
            while (epoch_number < self.hyperparameters['max_epochs']
                   and no_improvement_counter < self.hyperparameters['patience']):
                print('==== Epoch %i ====' % (epoch_number,))
                train_loss, train_acc = self.__run_epoch_in_batches(train_data,
                                                                    "%i (train)" % (epoch_number,),
                                                                    is_train=True, epoch_number=epoch_number)
                print(' Training Loss: %.6f, Accuracy: %.2f%%' % (train_loss, train_acc))
                val_loss, val_acc = self.__run_epoch_in_batches(valid_data, 
                                                                "%i (valid)" % (epoch_number,),
                                                                is_train=False, epoch_number=epoch_number)
                print(' Validation Loss: %.6f, Accuracy: %.2f%%' % (val_loss, val_acc))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_counter = 0
                    self.save(model_path)
                    print("  Best result so far -- saving model as '%s'." % (model_path,))
                else:
                    # record epochs without improvement for early stopping
                    no_improvement_counter += 1
                epoch_number += 1

        return model_path

    def test(self, test_data: LoadedSamples):
        """
        Simple test routine returning per-token accuracy, conditional on correct prediction
        of the sequence so far.

        Args:
            test_data: Tensorised data to test on.
        """
        _, test_acc = self.__run_epoch_in_batches(test_data, "Test", is_train=False)
        print('Test accuracy: %.2f%%' % test_acc)
