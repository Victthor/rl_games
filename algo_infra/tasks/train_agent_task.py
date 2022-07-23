
from hydra.utils import instantiate


class TrainAgent:

    def __init__(self, cfg) -> None:

        self.cfg = cfg

        self.agent = instantiate(cfg.agent)
        self.train_environment = instantiate(cfg.environment)
        self.eval_environment = instantiate(cfg.environment)
        self.driver = instantiate(cfg.driver)
        self.pretrain_driver = instantiate(cfg.collect_driver)
        self.collect_driver = instantiate(cfg.collect_driver)
        self.dataset = instantiate(cfg.dataset)
        self.logger = instantiate(cfg.logger)
        
    def train(self) -> None:

        # # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        # self.agent.train = common.function(self.agent.train)

        # Collect experience prior to training
        self.pretrain_driver.run(self.train_environment.reset())

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        # Reset the environment
        time_step = self.train_environment.reset()
        
        for _ in range(self.cfg.num_iterations):
            
            # Collect a few steps and save to the replay buffer
            time_step, _ = self.collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network
            experience, unused_info = next(self.dataset)
            loss_info = self.agent.train(experience)
            train_loss = loss_info.loss
