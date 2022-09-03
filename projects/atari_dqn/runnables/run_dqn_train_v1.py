
import os
import hydra

from algo_infra.tasks.train_agent_task import TrainAgent


@hydra.main(version_base=None, config_path=os.path.join('..', 'configs', 'dqn_v1'), config_name="cfg")
def main_train(cfg):
    tagent = TrainAgent(cfg)
    tagent.train()


if __name__ == '__main__':
    main_train()
