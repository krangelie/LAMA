#!/bin/bash

data_path=/export/home/kraft/data
log_dir=$data_path/lama/output/results/
#log_file=$log_dir/lama_log.out

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

mkdir -p $log_dir
#touch $log_file

python run_lama.py \
      lama_data_dir=$data_path/lama/ \
      results_file=$data_path/lama/last_results.csv \
      log_dir=$log_dir \
      data_path=$data_path
      #> $log_file 2>&1 &
