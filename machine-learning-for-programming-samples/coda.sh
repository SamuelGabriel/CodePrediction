#!/bin/bash


cl run :corpus :language_model "ls language_model; ls corpus; python3 language_model/train.py trained_models/ corpus/org/elasticsearch/{xpack,search}" --request-gpus 1 --request-docker-image samgmuller/tf1.12base:0.3 --name $1 $2 $3
