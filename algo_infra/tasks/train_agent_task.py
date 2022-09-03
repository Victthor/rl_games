
from hydra.utils import instantiate
# from tf_agents.environments import tf_py_environment
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class TrainAgent:

    def __init__(self, cfg) -> None:

        """
        driver runs environment (receives environment and policy), 
        observer watches this run and writes trajectory experiences into replay buffer, 
        replay buffer writes to tables in reverb server
        """

        self.cfg = cfg

        # 1) training and evaluation environments
        self.train_environment = instantiate(cfg.environment)
        self.eval_environment = instantiate(cfg.environment)
        self.tf_train_environment = TFPyEnvironment(self.train_environment)
        self.tf_eval_environment = TFPyEnvironment(self.eval_environment)

        # 2) agent (DQN, ...)
        # partial instantiation, agent needs specs from environment: time_step_spec and action_spec
        self.agent = instantiate(cfg.agent)
        self.agent = self.agent(
            time_step_spec=self.tf_train_environment.time_step_spec(), 
            action_spec=self.tf_train_environment.action_spec()
        )

        # 3) reverb server to store experience
        self.reverb_server = instantiate(cfg.reverb_server, self.agent, _recursive_=False)

        # 4) buffer
        # partial instantiation, needs server and data_spec
        self.experience_replay_buffer = instantiate(cfg.experience_replay_buffer)
        self.experience_replay_buffer = self.experience_replay_buffer(
            data_spec=self.agent.collect_data_spec, 
            server_address=None,  # add future support
            local_server=self.reverb_server
        )

        # 5) observer
        # partial instantiation, needs experience_replay_buffer's py_client
        self.experience_observer = instantiate(cfg.experience_observer)
        self.experience_observer = self.experience_observer(py_client=self.experience_replay_buffer.py_client)

        # 6a) init driver, collects experience prior to training to accumulate buffer
        # partial instantiation, needs env, policy, observers
        self.initial_collect_driver = instantiate(cfg.drivers.initial_collect_driver)
        self.initial_collect_driver = self.initial_collect_driver(
            env=self.train_environment,
            policy=PyTFEagerPolicy(self.agent.collect_policy, use_tf_function=True),
            observers=[self.experience_observer],
        )

        # 6b) driver
        # partial instantiation, needs env, policy, observers
        self.collect_driver = instantiate(cfg.drivers.collect_driver)
        self.collect_driver = self.collect_driver(
            env=self.train_environment,
            policy=PyTFEagerPolicy(self.agent.collect_policy, use_tf_function=True),
            observers=[self.experience_observer],
        )
        
        # 7) dataset
        self.dataset = self.experience_replay_buffer.as_dataset(
            num_parallel_calls=cfg.experience_dataset.num_parallel_calls,
            sample_batch_size=cfg.experience_dataset.sample_batch_size,
            num_steps=cfg.experience_dataset.num_steps,
        ).prefetch(cfg.experience_dataset.prefetch)

        # 8) logger
        self.logger = instantiate(cfg.logger)
        
    def train(self) -> None:

        # Initialize agent
        self.agent.initialize()

        # # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        # self.agent.train = common.function(self.agent.train)

        # Collect experience prior to training
        self.initial_collect_driver.run(self.train_environment.reset())

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        # Reset the environment
        time_step = self.train_environment.reset()

        iterator_dataset = iter(self.dataset)
        
        for _ in range(self.cfg.num_iterations):
            
            # Collect a few steps and save to the replay buffer
            time_step, _ = self.collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network
            experience, unused_info = next(iterator_dataset)
            loss_info = self.agent.train(experience)
            train_loss = loss_info.loss
            print(train_loss)
