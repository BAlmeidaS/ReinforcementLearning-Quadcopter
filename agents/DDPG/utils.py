import json

from .agent import DDPG


def load_agent(task, folder, uuid):
    """Load the weights from an agent previously trained"""
    path = f'{folder}/{uuid}/'

    with open(path + 'params.json', 'r') as f:
        params = json.load(f)

    # fix the type of num_nodes stored on json file
    params['num_nodes'] = int(params['num_nodes'])

    # loading the agent
    agent = DDPG(task, **params)
    agent.load_weights(folder + 'weights')

    return agent
