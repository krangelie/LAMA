# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from scripts.batch_eval_KB_completion import main as run_evaluation
from scripts.batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict

_LMs = [
    # {
    #     "lm": "gpt2",
    #     "label": "gpt2",
    #     "models_names": ["gpt2"],
    #     "gpt2_model_name": "gpt2",
    #     "gpt2_model_dir": None,
    #     "tokenizer_dir": "pre-trained_language_models/gpt/gpt2",
    # },
    {
        "lm": "gpt2-medium",
        "label": "gpt2-medium",
        "models_names": ["gpt2"],
        "gpt2_model_name": "gpt2-medium",
        "gpt2_model_dir": "pre-trained_language_models/gpt/gpt2-medium",
        "tokenizer_dir": "pre-trained_language_models/gpt/gpt2-medium"
    },
    {
        "lm": "gpt2-medium",
        "label": "gpt2-medium_kelm-full",
        "models_names": ["gpt2"],
        "gpt2_model_name": "gpt2-medium",
        "gpt2_model_dir": "kelm/output/gpt2-medium/kelm_full/",
        "tokenizer_dir": "pre-trained_language_models/gpt/gpt2-medium"
    },
    {
        "lm": "luke",
        "label": "luke",
        "models_names": ["hfluke"],
        "luke_model_name": "studio-ousia/luke-base",
        "luke_model_dir": None,
        "tokenizer_dir": None
    }
    ]


LMs = [
    {
        "lm": "roberta-base",
        "label": "roberta-b-kelm",
        "models_names": ["hfroberta"],
        "hfroberta_model_name": "roberta-base",
        "hfroberta_model_dir": "/media/angelie/Samsung_T5/KELM-tuned-models/KELM-RoBERTa/roberta"
                              "-base",
        "tokenizer_dir": "roberta-base"
    }]
def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param,
    results_file,
    log_dir,
    data_path,
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open(results_file, "a+")
    results_file.write(
        "=={}==\n".format(input_param["label"])
    )
    results_file.flush()

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": None,
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": log_dir,
            "data_path": data_path,
            "full_logdir": log_dir + "{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        # fix https://github.com/facebookresearch/LAMA/issues/30
        if input_param["lm"] in ["elmo"]:
            if model is not None:
                del model
                model = None

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters(data_path_pre="data/"):
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre += "Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters, cfg):
    for ip in LMs:
        print(ip["label"])

        use_negated_probes = False  # vanilla LAMA
        # use_negated_probes = True  # Negated-LAMA
        run_experiments(*parameters,
                        input_param=ip,
                        results_file=cfg.results_file,
                        log_dir=cfg.log_dir,
                        data_path=cfg.data_path,
                        use_negated_probes=use_negated_probes)


def run_lama(cfg):
    use_negated_probes = False  # vanilla LAMA
    # use_negated_probes = True  # Negated-LAMA

    print("1. Google-RE")
    parameters = get_GoogleRE_parameters(cfg.lama_data_dir)
    #run_experiments(*parameters, results_file=cfg.results_file, log_dir=cfg.log_dir, use_negated_probes=use_negated_probes)
    run_all_LMs(parameters, cfg)

    print("2. T-REx")
    parameters = get_TREx_parameters(cfg.lama_data_dir)
    #run_experiments(*parameters, cfg.results_file, use_negated_probes=use_negated_probes)
    run_all_LMs(parameters, cfg)

    print("3. ConceptNet")
    print("NA")
    #parameters = get_ConceptNet_parameters(cfg.benchmark.lama_data_dir)
    #run_experiments(*parameters, cfg.results_file, use_negated_probes=use_negated_probes)
    #run_all_LMs(parameters, cfg)

    print("4. SQuAD")
    parameters = get_Squad_parameters(cfg.lama_data_dir)
    #run_experiments(*parameters, cfg.results_file,   use_negated_probes=use_negated_probes)
    run_all_LMs(parameters, cfg)

    #/export/home/kraft/data/

