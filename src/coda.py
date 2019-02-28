import subprocess
import sys
from argparse import ArgumentParser

# usage: python coda.py <name> <other args>

parser = ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--hypers-override', dest='hypers_override', default='{}')
parser.add_argument('--models-override', dest='models_override', default='{}')
parser.add_argument('--max-out', action='store_true')
parser.add_argument('--typed-dataset', action='store_true')
args = parser.parse_args()

corpus_name = 'typedcorpus' if args.typed_dataset else 'corpus'

extra_requests = ['--request-cpus', '6', '--request-memory', '55g'] if args.max_out else []

subprocess.run(['cl', 'work', 'sgm-rnn-tutorial'])
cmd ="python3 -u language_model/train.py {} ".format(' '.join(
        ['--hypers-override', "\'"+args.hypers_override+"\'", '--models-override', "\'"+args.models_override+"\'"]
     )) \
     + f"trained_models/ {corpus_name}/train {corpus_name}/valid"
subprocess.run(['cl', 'run', ':' + corpus_name, ':language_model', cmd, '--request-gpus', '1', '--request-docker-image', 'samgmuller/tf1.12base:0.4', "--name", args.name] + extra_requests)