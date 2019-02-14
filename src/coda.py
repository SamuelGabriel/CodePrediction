import subprocess
import sys
from argparse import ArgumentParser

# usage: python coda.py <name> <other args>

parser = ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--hypers-override', dest='hypers_override', default='{}')
parser.add_argument('--models-override', dest='models_override', default='{}')
parser.add_argument('--max-out', dest='max_out', default=False)
args = parser.parse_args()

extra_requests = ['--request-cpus', '6', '--request-memory', '56g'] if args.max_out else []

subprocess.run(['cl', 'work', 'sgm-rnn-tutorial'])
cmd ="python3 language_model/train.py {} trained_models/ corpus/train corpus/valid".format(' '.join(
    ['--hypers-override', "\'"+args.hypers_override+"\'", '--models-override', "\'"+args.models_override+"\'"]))
subprocess.run(['cl', 'run', ':corpus', ':language_model', cmd, '--request-gpus', '1', '--request-docker-image', 'samgmuller/tf1.12base:0.4', "--name", args.name] + extra_requests)