
from tf_agents.environments import tf_py_environment
from hydra.utils import instantiate


def tf_py_env_factory(params):

    env = instantiate(params)
    return tf_py_environment.TFPyEnvironment(env)
