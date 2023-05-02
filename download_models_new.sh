#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

#ROOD_DIR="$(realpath $(dirname "$0"))"
DATA_DIR="/export/home/kraft/data/"
DST_DIR="$DATA_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"


echo "cased models"

echo "GPT2"
if [[ ! -f $DST_DIR/gpt/gpt2/config.json ]]; then
  rm -rf "$DST_DIR/gpt/gpt2"
  mkdir -p "$DST_DIR/gpt/gpt2"
  cd "$DST_DIR/gpt/gpt2"
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json' -O vocab.json
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt' -O merges.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json' -O 'config.json'
  cd ../..
fi

echo "GPT2-medium"
if [[ ! -f $DST_DIR/gpt/gpt2-medium/config.json ]]; then
  rm -rf "$DST_DIR/gpt/gpt2-medium"
  mkdir -p "$DST_DIR/gpt/gpt2-medium"
  cd "$DST_DIR/gpt/gpt2-medium"
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json' -O vocab.json
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt' -O merges.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-config.json' -O 'config.json'
  cd ../..
fi

echo "HuggingFace RoBERTa"
mkdir -p "$DST_DIR/roberta/roberta-base"
if [[ ! -f $DST_DIR/roberta/roberta-base/config.json ]]; then
  rm -rf "$DST_DIR/roberta/roberta-base"
  mkdir -p "$DST_DIR/roberta/roberta-base"
  cd "$DST_DIR/roberta/roberta-base"
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json' -O vocab.json
  wget 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt' -O merges.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json' -O 'config.json'
  cd ../..
fi

echo "LUKE-base"
mkdir -p "$DST_DIR/luke"
if [[ ! -f "$DST_DIR/luke/config.json" ]]; then
  rm -rf "$DST_DIR/luke"
  mkdir -p "$DST_DIR/luke"
  cd "$DST_DIR/luke"
  wget 'https://huggingface.co/studio-ousia/luke-base/raw/main/vocab.json' -O vocab.json
  wget 'https://huggingface.co/studio-ousia/luke-base/raw/main/merges.txt' -O merges.txt
  wget -c 'https://huggingface.co/studio-ousia/luke-base/resolve/main/pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://huggingface.co/studio-ousia/luke-base/raw/main/config.json' -O 'config.json'
  cd ../..
fi

echo "Colake-base"
mkdir -p "$DST_DIR/colake"
if [[ ! -f "$DST_DIR/colake/model.bin" ]]; then
  rm -rf "$DST_DIR/colake"
  mkdir -p "$DST_DIR/colake"
  cd "$DST_DIR/colake"
  wget 'https://drive.google.com/file/d/1MEGcmJUBXOyxKaK6K88fZFyj_IbH9U5b/edit' -O model.bin
  cd ../..
fi

cd "$DATA_DIR"
echo 'Building common vocab'
if [ ! -f "$DST_DIR/common_vocab_cased.txt" ]; then
  python lama/vocab_intersection.py data_path=$DATA_DIR
else
  echo 'Already exists. Run to re-build:'
  echo 'python lama/vocab_intersection.py'
fi

