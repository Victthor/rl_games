import os
from hydra import compose, initialize
from omegaconf import OmegaConf


if __name__ == "__main__":
    # context initialization
    with initialize(config_path=os.path.join('..', 'configs', 'dqn_tutorial'), job_name="test_app"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))
