# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, sys
from modules import build_model_by_name
from tqdm import tqdm
import argparse
import spacy
import modules.base_connector as base
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


CASED_MODELS = [
    {
        # "hfRoBERTa base"
        "lm": "hfroberta",
        "hfroberta_model_name": "roberta-base",
        "hfroberta_model_dir": "pre-trained_language_models/roberta/roberta-base",
        "tokenizer_dir": "roberta-base",
    },
    {
        # "hfLUKE base"
        "lm": "hfluke",
        "luke_model_name": "luke-base",
        "luke_model_dir": "pre-trained_language_models/luke/luke-base",
        "tokenizer_dir": "studio-ousia/luke-base",
    },
    {
        # "Colake base"
        "lm": "colake",
        "colake_model_name": "colake",
        "luke_model_dir": "pre-trained_language_models/colake/model.bin",
        "tokenizer_dir": "roberta-base",
    },
    {
        # "OpenAI GPT-2"
        "lm": "gpt2",
        "gpt2_model_name": "gpt2-medium",
        "gpt2_model_dir": "pre-trained_language_models/gpt/gpt2-medium",
        "tokenizer_dir": "pre-trained_language_models/gpt/gpt2-medium"
    },
    {
        # "KELM-GPT-2"
        "lm": "gpt2",
        "gpt2_model_name": "gpt2-medium",
        "gpt2_model_dir": "kelm/output/gpt2-medium/kelm_full/",
        "tokenizer_dir": "pre-trained_language_models/gpt/gpt2-medium"
    }
]

CASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_cased.txt"

LOWERCASED_MODELS = []

LOWERCASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_lowercased.txt"


def __vocab_intersection(models, filename):

    vocabularies = []

    for arg_dict in models:

        args = argparse.Namespace(**arg_dict)
        print(args)
        model = build_model_by_name(args.lm, args)

        vocabularies.append(model.vocab)
        print(type(model.vocab))

    if len(vocabularies) > 0:
        common_vocab = set(vocabularies[0])
        for vocab in vocabularies:
            common_vocab = common_vocab.intersection(set(vocab))

        # no special symbols in common_vocab
        for symbol in base.SPECIAL_SYMBOLS:
            if symbol in common_vocab:
                common_vocab.remove(symbol)

        # remove stop words
        from spacy.lang.en.stop_words import STOP_WORDS
        for stop_word in STOP_WORDS:
            if stop_word in common_vocab:
                print(stop_word)
                common_vocab.remove(stop_word)

        common_vocab = list(common_vocab)

        # remove punctuation and symbols
        nlp = spacy.load('en')
        manual_punctuation = ['(', ')', '.', ',']
        new_common_vocab = []
        for i in tqdm(range(len(common_vocab))):
            word = common_vocab[i]
            doc = nlp(word)
            try:
                token = doc[0]
            except:
                print(f"Skip vocab entry {i}.")
                continue
            if(len(doc) != 1):
                print(word)
                for idx, tok in enumerate(doc):
                    print("{} - {}".format(idx, tok))
            elif word in manual_punctuation:
                pass
            elif token.pos_ == "PUNCT":
                print("PUNCT: {}".format(word))
            elif token.pos_ == "SYM":
                print("SYM: {}".format(word))
            else:
                new_common_vocab.append(word)
            # print("{} - {}".format(word, token.pos_))
        common_vocab = new_common_vocab

        # store common_vocab on file
        with open(filename, 'w') as f:
            for item in sorted(common_vocab):
                f.write("{}\n".format(item))


@dataclass
class VocabConfig:
    data_path: str = ""  # location of pre-trained_language_models folder in which the models and tokenizers are stored
    # see scripts/run_experiments.py

cs = ConfigStore.instance()
cs.store(name="conf", node=VocabConfig())

@hydra.main(version_base=None, config_name="conf")
def main(cfg: VocabConfig) -> None:
    # cased version
    __vocab_intersection(CASED_MODELS, os.path.join(cfg.data_path, CASED_COMMON_VOCAB_FILENAME))
    # lowercased version
    #__vocab_intersection(LOWERCASED_MODELS, LOWERCASED_COMMON_VOCAB_FILENAME)


if __name__ == '__main__':
    main()
