import json

from .agent import DDPG
import os

REF_PATH = os.path.dirname(__file__)

def load_agent(task, folder, uuid):
    """Load the weights from an agent previously trained"""
    path = REF_PATH + f'/../../{folder}/{uuid}/'

#    import ipdb; ipdb.set_trace()

    with open(path + 'params.json', 'r') as f:
        params = json.load(f)

    # fix the type of num_nodes stored on json file
    params['num_nodes'] = int(params['num_nodes'])

    # loading the agent
    agent = DDPG(task, **params)
    agent.load_weights(path + 'weights')

    return agent
