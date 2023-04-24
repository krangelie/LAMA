import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from scripts.run_experiments import run_lama

@dataclass
class LamaConfig:
    lama_data_dir: str = "" # LAMA dataset directory
    results_file: str = "" # where to store the analysis results
    log_dir: str = "" # where to store the log
    data_path : str = "" # location of pre-trained_language_models folder in which the models and tokenizers are stored
                            # see scripts/run_experiments.py


cs = ConfigStore.instance()
cs.store(name="conf", node=LamaConfig())


@hydra.main(version_base=None, config_name="conf")
def run(cfg: LamaConfig) -> None:
    print(f"Running eval with LAMA")
    run_lama(cfg)


if __name__ == '__main__':
    run()