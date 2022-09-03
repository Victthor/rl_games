
import plotly.express as px
from skimage.transform import rescale
from skimage.color import rgb2gray
import numpy as np
from copy import deepcopy

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import suite_gym
from tf_agents.trajectories import TimeStep
from tf_agents.specs import BoundedArraySpec


class SuiteGymMod(PyEnvironment):
    '''
    This wrapper class based on preprocessing description of DQN original paper
    1) channelwise maximum between current and previous frame to avoid atari blickering
    2) extract Y channel (gray)
    3) rescale by a factor of 2
    4) concatenate m such frames together as single step
    '''

    def __init__(self, env_name, params=None) -> None:
        
        self.py_env = suite_gym.load(env_name)
        self.params = params

        # store previous observation to avoid blickering in atari frames (odd/even)
        self.prev_observation = None

        self._episode_ended = False

        obs_params = params.get('observation', {})
        self.observation_dtype = obs_params.get('dtype', np.float32)
        self.observation_minimum = obs_params.get('minimum', 0.0)
        self.observation_maximum = obs_params.get('maximum', 255.0)

    def action_spec(self):

        return self.py_env.action_spec()

    def observation_spec(self):

        orig_spec = self.py_env.observation_spec()

        shape = (
            int(orig_spec.shape[0] * self.params['rescale']['scale']), 
            int(orig_spec.shape[1] * self.params['rescale']['scale']), 
            self.params['repeats']
        )

        spec = BoundedArraySpec(
            shape, 
            self.observation_dtype, 
            minimum=self.observation_minimum, 
            maximum=self.observation_maximum
        )

        return spec

    def _reset(self):

        next_time_step = self.py_env.reset()

        # store frame
        self.prev_observation = deepcopy(next_time_step[-1])
        
        # preprocess
        observation = self.preprocess_observation(np.asarray(next_time_step[-1], dtype=self.observation_dtype))

        # repeat m times
        observation = np.repeat(observation[:, :, np.newaxis], repeats=[self.params['repeats']], axis=2)

        # TimeStep(step_type, reward, discount, observation)
        return TimeStep(next_time_step[0], next_time_step[1], next_time_step[2], observation)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode
            return self.reset()
        
        observations = []
        step_type, reward, discount = None, None, None

        for _ in range(self.params['repeats']):

            next_time_step = self.py_env.step(action)

            # anti flicker
            observation = np.maximum(next_time_step[-1], self.prev_observation).astype(dtype=self.observation_dtype)

            # preprocess
            observation = self.preprocess_observation(observation)

            observations.append(observation)

            # cache last observation to handle flickering
            self.prev_observation = deepcopy(next_time_step[-1])

            # update step_type, reward, discount (can add update rule)
            step_type, reward, discount = next_time_step[:3]

            if next_time_step.is_last():
                self._episode_ended = True
                break
        
        n_observations = len(observations)

        # replicate last observation if episode ended
        if n_observations < self.params['repeats']:
            observations += [observations[-1]] * (self.params['repeats'] - n_observations)

        assert(len(observations) == self.params['repeats'])

        observations = np.stack(observations, axis=2)

        # TimeStep(step_type, reward, discount, observation)
        return TimeStep(step_type, reward, discount, observations)

    def preprocess_observation(self, observation):

        # extract Y channel
        observation = rgb2gray(observation)

        # rescale
        observation = rescale(observation, scale=self.params['rescale']['scale'])

        return observation


if __name__ == '__main__':

    from tf_agents.policies import random_py_policy
    from tf_agents.environments import tf_py_environment
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def get_obs_fig(observation):

        fig = make_subplots(rows=2, cols=2)

        fig.add_trace(go.Heatmap(z=observation[:, :, 0], colorscale='gray'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=observation[:, :, 1], colorscale='gray'), row=1, col=2)
        fig.add_trace(go.Heatmap(z=observation[:, :, 2], colorscale='gray'), row=2, col=1)
        fig.add_trace(go.Heatmap(z=observation[:, :, 3], colorscale='gray'), row=2, col=2)

        fig.update_yaxes(autorange='reversed', row=1, col=1)
        # fig.update_xaxes(constrain='domain', row=1, col=1)
        fig.update_yaxes(autorange='reversed', row=1, col=2)
        # fig.update_xaxes(constrain='domain', row=1, col=2)
        fig.update_yaxes(autorange='reversed', row=2, col=1)
        # fig.update_xaxes(constrain='domain', row=2, col=1)
        fig.update_yaxes(autorange='reversed', row=2, col=2)
        # fig.update_xaxes(constrain='domain', row=2, col=2)

        return fig

    env_name = 'ALE/Phoenix-v5'
    params = {
        'repeats': 4,
        'rescale': {
            'scale': 0.5,
        }
    }

    env = SuiteGymMod(env_name=env_name, params=params)

    print('Observation Spec:')
    print(env.time_step_spec().observation)
    print('Reward Spec:')
    print(env.time_step_spec().reward)
    print('Discount Spec:')
    print(env.time_step_spec().discount)
    print('Reward Spec:')
    print(env.time_step_spec().reward)

    print('Action Spec:')
    print(env.action_spec())

    tf_env = tf_py_environment.TFPyEnvironment(env)

    print('TF Observation Spec:')
    print(tf_env.time_step_spec().observation)
    print('TF Reward Spec:')
    print(tf_env.time_step_spec().reward)
    print('TF Action Spec:')
    print(tf_env.action_spec())

    random_policy = random_py_policy.RandomPyPolicy(env.time_step_spec(), env.action_spec())

    # TimeStep(step_type, reward, discount, observation)
    time_step = env.reset()

    tf_time_step = tf_env.reset()

    step = 1

    fig = get_obs_fig(time_step[-1])
    fig.show()

    while not time_step.is_last():
        print(f'step: {step}')
        step += 1
        policy_step = random_policy.action(time_step)
        time_step = env.step(policy_step.action)
        
        fig = get_obs_fig(time_step[-1])
        fig.show()

        nn = 1

    b = 1
