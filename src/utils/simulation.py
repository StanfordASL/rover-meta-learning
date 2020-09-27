import numpy as np
from tqdm import tqdm as tqdm

def rollout(T, agent, dynamics, cost=None, verbose=False, randomize=True, plot_functions=None):
    """
    T: int > 0
    agent: Agent
    dynamics: Dynamics
    cost: CostFunction (if not passed, costs aren't evaluated)

    carries out a rollout of length T with agent interacting with dynamics,
    collecting costs according to cost

    returns data, a dictionary with various data on the rollout
    """
    states = []
    actions = []
    costs = []
    Linv_eig = []
    
    agent.reset()
    x = dynamics.reset()

    states.append(x)

    for t in tqdm(range(T), disable=(not verbose)):
        # choose action
        u = agent.act(x)
        actions.append(u)
        # incur cost
        if cost:
            j = cost.stage_cost(x,u)
            costs.append(j)
        # observe transition
        xp = dynamics.step(u)
        agent.incorporate_transition(x,u,xp)

        # step forward
        x = xp
        states.append(x)
        
        if agent.model is not None:
            Linv_eig.append(np.max(np.linalg.eig(agent.model.params[1].detach())[0]))
        
        if plot_functions is not None:
            for p in plot_functions:
                p(agent,dynamics,states,actions)

    if cost:
        costs.append(cost.terminal_cost(x))
        costs = np.stack(costs)
        
    data = {
        'states': np.stack(states),
        'actions': np.stack(actions),
        'costs': costs,
        'params': dynamics.parameters.values(),
        'Linv_eig': Linv_eig
    }
    
    data.update(agent.episode_log())

    
    return data


def rollout_comparison(T, agents, dynamics, cost=None, verbose=False):
    """
    T: int > 0
    agents: list of Agent objects
    dynamics: Dynamics
    cost: CostFunction (if not passed, costs aren't evaluated)

    carries out a rollout of length T with agent interacting with dynamics,
    collecting costs according to cost

    returns list of data, a dictionary with various data on the rollout for each Agent in agents
    """
    # shuffle parameters of dynamics
    dynamics.reset()
    stats_list = []
    for agent in agents:
        data = rollout(T, agent, dynamics, cost, verbose, randomize=False)
        stats_list.append(data)

    return stats_list

def open_loop_rollout(dynamics, actions):
    """
    returns sequence of states obtained by applying actions open loop
    """
    x = dynamics.reset()
    states = [x]
    for u in actions:
        x = dynamics.step(u)
        states.append(x)
    return np.stack(states, axis=0)


def evaluate_trajs(cost, states, controls):
    """
    Evaluates states, controls under CostFunction cost
    states: (N, T+1, dim)
    controls: (N, T, dim)
    """
    N = states.shape[0]
    T = controls.shape[1]
    costs = np.zeros(N)
    for i, (x, u) in enumerate(zip(states, controls)):
        for t in range(T):
            costs[i] += cost.stage_cost(x[t],u[t])
        costs[i] += cost.terminal_cost(x[T])
    
    return costs