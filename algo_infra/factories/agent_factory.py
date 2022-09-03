
from hydra.utils import instantiate


def agent_factory(train_env, params=None):

    agent = instantiate(
        params, 
        time_step_spec=train_env.time_step_spec(), 
        action_spec=train_env.action_spec()
    )
    
    return agent
