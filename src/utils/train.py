from ..models.dataset import DynamicsRandomDataset, DynamicsRandomActionDataset, DynamicsTrajDataset, PresampledDataset, TorchDatasetWrapper
from ..agents.random import RandomAgent
from .simulation import evaluate_trajs
from torch.utils.data import DataLoader

import numpy as np
import os


def get_folder_name(filename):
    return '/'.join(filename.split('/')[:-1])

def maybe_makedir(filename):
    path_to_create = get_folder_name(filename)
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def save_transitions_to_file(dynamics, state_space, action_space, N, T, filename, verbose=True, sample_trajectories=False):
    """
    samples transition data from (dynamics, state_space, action_space)
    with N total datasets with T samples each

    saves to filename in the npz format
    
    sample_trajectories: if false, samples independent transitions
    """
    if sample_trajectories:
        agent = RandomAgent(action_space)
        dataset = DynamicsTrajDataset(dynamics, agent)
    else:
        dataset = DynamicsRandomDataset(dynamics, state_space, action_space)
        
    X,Y = dataset.sample(N,T, verbose=verbose)
    
    maybe_makedir(filename)

    np.savez(filename, X=X, Y=Y)
    if verbose:
        print("Saved transitions to ", filename)


def load_dataset_from_file(filename, whiten=False):
    data = np.load(filename)
    X = data["X"]
    Y = data["Y"]

    dataset = PresampledDataset(X,Y,whiten)

    return dataset



# framework for on-policy training

class ModelBasedRL:
    def __init__(self, dynamics, cost, agent, initial_agent, T, batch_size=25):
        self.dynamics = dynamics
        self.cost = cost
        self.agent = agent
        self.initial_agent = initial_agent
        self.Ns = []
        self.avg_costs = []
        self.batchsize = batch_size
        self.T = T
        
        self.dataset = None
        
        
    def initial_phase(self, N, verbose=True):
        dataset = DynamicsTrajDataset(self.dynamics, self.initial_agent)
        X,Y = dataset.sample(N, self.T, verbose=verbose)
        costs = self.compute_batch_costs(X,Y)
        self.dataset = PresampledDataset(X, Y)
#         self.dataset.prune()
        
    def load_data(self, filename):
        self.dataset = PresampledDataset(filename=filename)
        
    def save_data(self, filename):
        self.dataset.save(filename)
        
    def save_model(self, filename):
        self.agent.model.save(filename)
        
    def load_model(self, filename):
        self.agent.model.load(filename)

    def train_model(self, n_iter, verbose=True):
        if not self.dataset:
            print("Call initial_phase first")
            raise Exception
        self.dataloader = DataLoader(TorchDatasetWrapper(self.dataset, traj_len=self.T),batch_size=self.batchsize)
        self.agent.model.train(self.dataloader, n_iter, verbose=verbose)

    def gather_on_policy_trajs(self, N, verbose=True):
        if not self.dataset:
            print("Call initial_phase first")
            raise Exception
        dataset = DynamicsTrajDataset(self.dynamics, self.agent)
        X,Y = dataset.sample(N, self.T, verbose=verbose)        
        costs = self.compute_batch_costs(X,Y)

        self.dataset.append(X, Y)
#         self.dataset.prune()
        return costs

    def compute_batch_costs(self, X, Y):
        states = np.concatenate([X[:,:,:self.dynamics.ob_dim], Y[:,-1:,:]], axis=1)
        controls = X[:,:,self.dynamics.ob_dim:]
        
        costs = evaluate_trajs(self.cost, states, controls)
        avg_cost = np.mean(costs)
        
        if self.dataset is not None:
            self.Ns.append(self.dataset.N)
        else:
            self.Ns.append(0)
        self.avg_costs.append(np.mean(costs))
        print("avg_cost:", np.mean(costs))
        
        return costs