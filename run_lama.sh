#!/bin/bash

declare -a dir_list=("gender/female" "gender/male" "location/AF" "location/AS" "location/EU" "location/SA" "location/OC" "location/NA")

data_path=/export/home/kraft/data/

for dir in "${dir_list[@]}"; do

  lama_data_dir=${data_path}lama-bias/${dir}
  results_dir=$lama_data_dir
  log_file=$results_dir/nohup.out

  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0

  mkdir -p results_dir
  touch $log_file

  nohup python run_lama.py \
        lama_data_dir=$lama_data_dir \
        results_file=$results_dir/last_results.csv \
        log_dir=$results_dir \
        data_path=$data_path \
        run_google=False \
        run_squad=False \
        > $log_file 2>&1 &

done