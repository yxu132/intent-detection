#!/bin/bash

# replace DATA_DIR with your own data path

DATA_DIR=/Users/yxu132/data

python run_classifier_intent.py --do_train --data_dir=${DATA_DIR}/oos_binary/down_sampled_bin \
    --bert_config_file=${DATA_DIR}/bert/uncased_L-12_H-768_A-12/bert_config.json --task_name=intent \
    --vocab_file=${DATA_DIR}/bert/uncased_L-12_H-768_A-12/vocab.txt --output_dir=intent_output \
    --init_checkpoint=${DATA_DIR}/bert/uncased_L-12_H-768_A-12/bert_model.ckpt --do_lower_case
