
import reverb
from hydra.utils import instantiate
from tf_agents.specs import tensor_spec


def rb_server_factory(agent, tables):

    """
    agent needs to get signature spec
    """

    signature = tensor_spec.from_spec(agent.collect_data_spec)
    signature = tensor_spec.add_outer_dim(signature)

    tables_ = []
    for table in tables:
        table_ = instantiate(table)  # _partial_, returns functools.partial, see https://hydra.cc/docs/advanced/instantiate_objects/overview/ v1.2
        table_ = table_(signature=signature)
        tables_.append(table_)
    
    return reverb.Server(tables_)
