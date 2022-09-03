
import plotly.express as px
import matplotlib.pyplot as plt
import gym
import reverb
from tf_agents.policies import random_py_policy
from tf_agents.environments import suite_gym
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.agents import TFAgent
from tf_agents.specs import tensor_spec
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment


class DummyAgent(TFAgent):
    def __init__(self, time_step_spec, action_spec, policy):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=None
        )

    def _train(self, experience, weights=None):
        pass


"""Create our environment. Basically we define what game we want to play"""
# env_name = 'ALE/Breakout-v5'
env_name = 'ALE/Phoenix-v5'
env = gym.make(env_name)

"""Reset our environment, notice it returns the first frame of the game"""
# first_frame = env.reset()

# plt.imshow(first_frame)

# fig = px.imshow(first_frame)
# fig.show()

# py environment
py_env = suite_gym.load(env_name)

print('Observation Spec:')
print(py_env.time_step_spec().observation)
print('Reward Spec:')
print(py_env.time_step_spec().reward)
print('Action Spec:')
print(py_env.action_spec())

tf_env = tf_py_environment.TFPyEnvironment(py_env)

print('TF Observation Spec:')
print(tf_env.time_step_spec().observation)
print('TF Reward Spec:')
print(tf_env.time_step_spec().reward)
print('TF Action Spec:')
print(tf_env.action_spec())

# random policy
random_policy = random_py_policy.RandomPyPolicy(py_env.time_step_spec(), py_env.action_spec())

# dummy agent
agent = DummyAgent(py_env.time_step_spec(), py_env.action_spec(), random_policy)

# reverb server
table_name = 'single_policy_evaluation'
replay_buffer_max_length = 1000

replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature
)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server
)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    py_client=replay_buffer.py_client,
    table_name=table_name,
    max_sequence_length=20000,
)

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    py_env,
    agent.collect_policy,
    [rb_observer],
    max_episodes=3,
)

# Reset the environment.
time_step = py_env.reset()

tf_time_step = tf_env.reset()

# Collect a few steps and save to the replay buffer.
time_step, _ = collect_driver.run(time_step)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=1,
    sample_batch_size=1,
    num_steps=None).prefetch(3)

for episode_traj, unused_info in dataset.as_numpy_iterator():
    # print(episode_traj)

    img = episode_traj.observation[0]
    fig = px.imshow(img, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
    fig.show()

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 5
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2

    bbb = 1

n = 1
