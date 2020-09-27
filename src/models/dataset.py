import numpy as np
from tqdm import tqdm
from ..utils.simulation import rollout
from torch.utils.data import Dataset, DataLoader
import torch

class Camelid_Dataset:
    def __init__(self):
        pass

    # draw n_sample (x,y) pairs drawn from n_func functions
    # returns (x,y) where each has size [n_func, n_samples, x/y_dim]
    def sample(self, n_funcs, n_samples):
        raise NotImplementedError

        
class TorchDatasetWrapper(Dataset):
    """
    This provides a torch wrapper on the existing camelid dataset classes.
    It is used for dataloaders in training of torch models.
    
    We are doing our own batching (since this is how camelid datasets were implemented)
    Therefore we can't grab data according to indices and it is sampled randomly
    Thus, epochs do not make sense.
    """
    
    def __init__(self, camelid_dataset, traj_len=50):
        self.dataset = camelid_dataset
        self.batch_size = 1
        self.traj_len = traj_len
        
    def __len__(self):
        # returns the number of trajectories
        if hasattr(self.dataset, 'N'):
            return self.dataset.N
        else:
            return int(1e5)
    
    def __getitem__(self,idx):
        # will return a trajectory (corresponding to a particular task)
        # return a batch
        
        xu,xp = self.dataset.sample(self.batch_size, self.traj_len)
        
        # split x/u
        o_dim = xp.shape[-1]
        
        x = xu[0,:,:o_dim]
        u = xu[0,:,o_dim:]
        xp = xp[0,...]
        # map to tensors
        x_torch = torch.from_numpy(x)
        u_torch = torch.from_numpy(u)
        xp_torch = torch.from_numpy(xp)
        
        sample = {
            'x': x_torch,
            'u': u_torch,
            'xp': xp_torch
        }
        
#         print(xp)
        
        return sample

class PresampledTrajectoryDataset(Camelid_Dataset):
    def __init__(self, trajs, controls):
        self.trajs = trajs
        self.controls = controls
        self.o_dim = trajs[0].shape[-1]
        self.u_dim = controls[0].shape[-1]
        self.N = len(trajs)

    def sample(self, n_funcs, n_samples):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        for i in range(n_funcs):
            j = np.random.randint(self.N)
            T = self.controls[j].shape[0]
            if n_samples > T:
                raise ValueError('You are requesting more samples than are in this trajectory.')
            start_ind = 0
            if T > n_samples:
                start_ind = np.random.randint(T-n_samples)
            inds_to_keep = np.arange(start_ind, start_ind+n_samples)
            x[i,:,:self.o_dim] = self.trajs[j][inds_to_keep]
            x[i,:,self.o_dim:] = self.controls[j][inds_to_keep]
            y[i,:,:] = self.trajs[j][inds_to_keep+1]

        return x,y

class PresampledDataset(Camelid_Dataset):
    def __init__(self, X=None, Y=None, whiten=False, shuffle=False, filename=None, x_check=None, y_check=None):
        if (X is not None) and (Y is not None):
            # TODO: implement load from file functionality
            self.X = X
            self.Y = Y
        elif filename is not None:
            data = np.load(filename)
            self.X = data["X"]
            self.Y = data["Y"]
        else:
            raise Exception
            
        self.shuffle = shuffle
        
        self.x_check = x_check
        self.y_check = y_check
            
        self.x_dim = self.X.shape[-1]
        self.y_dim = self.Y.shape[-1]
        self.N = self.X.shape[0]
        self.T = self.X.shape[1]
        
        self.input_scaling = np.ones([1,1,self.X.shape[-1]])
        self.output_scaling = np.ones([1,1,self.Y.shape[-1]])
        
        deltas = (self.Y.T - self.X[:,:,:self.y_dim].T).reshape([self.y_dim,-1]).T
        #to filter episodes with 3 sigma events
        self.means = np.mean(deltas, axis=0)
        self.stds = np.std(deltas, axis=0)
        
        if whiten:
            self.input_scaling = np.std(self.X, axis=(0,1), keepdims=True)
            self.output_scaling = np.std(self.Y, axis=(0,1), keepdims=True)

    def sample(self, n_funcs, n_samples):
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))

        for i in range(n_funcs):
            j = np.random.randint(self.N)
            if n_samples > self.T:
                raise ValueError('You are requesting %d samples but there are only %d in the dataset.'%(n_samples, self.T))

            inds_to_keep = np.random.choice(self.T, n_samples)
            x[i,:,:] = self.X[j,inds_to_keep,:] / self.input_scaling
            y[i,:,:] = self.Y[j,inds_to_keep,:] / self.output_scaling
            
            if self.shuffle:
                inds_to_keep = np.random.choice(self.T, n_samples)
                x[i,:,:] = self.X[j,inds_to_keep,:] / self.input_scaling
                y[i,:,:] = self.Y[j,inds_to_keep,:] / self.output_scaling
            else:
                start_idx = 0 if (self.T == n_samples) else np.random.randint(self.T - n_samples)
                x[i,:,:] = self.X[j,start_idx:start_idx+n_samples,:]
                y[i,:,:] = self.Y[j,start_idx:start_idx+n_samples,:]

        return x,y
    
    def append(self,X,Y):
        self.X = np.concatenate([self.X, X], axis=0)
        self.Y = np.concatenate([self.Y, Y], axis=0)
        self.N = self.X.shape[0]
        
    def prune(self):
        """
        removes functions with values Y - X[:,:,:ydim] exceeding k*sigma to remove crazy data
        """
        X = self.X
        Y = self.Y
        
        good_eps = np.ones_like(self.X[:,0,0])
        if self.x_check is not None:
            good_eps = np.logical_and(good_eps, self.x_check(X))
        if self.y_check is not None:
            good_eps = np.logical_and(good_eps, self.y_check(X))
        
        self.X = X[good_eps,:,:]
        self.Y = Y[good_eps,:,:]
        self.N = self.X.shape[0]
        
    def save(self, filename):
        np.savez(filename, X=self.X, Y=self.Y)
    

class SinusoidDataset(Camelid_Dataset):
    def __init__(self, x_range, amp_range, phase_range, freq_range, noise_var=0.01):
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.freq_range = freq_range
        self.x_range = x_range
        self.noise_std = np.sqrt(noise_var)

    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 1
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        amp_list = self.amp_range[0] + np.random.rand(n_funcs)*(self.amp_range[1] - self.amp_range[0])
        phase_list = self.phase_range[0] + np.random.rand(n_funcs)*(self.phase_range[1] - self.phase_range[0])
        freq_list = self.freq_range[0] + np.random.rand(n_funcs)*(self.freq_range[1] - self.freq_range[0])
        for i in range(n_funcs):
            x_samp = self.x_range[0] + np.random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = amp_list[i]*np.sin(freq_list[i]*x_samp + phase_list[i]) + self.noise_std*np.random.randn(n_samples)

            x[i,:,0] = x_samp
            y[i,:,0] = y_samp

        if return_lists:
            return x,y,amp_list,phase_list

        return x,y


class TwoSinusoidDataset(Camelid_Dataset):
    def __init__(self, x_range, amp_range, phase_range, freq_range, noise_var=0.01):
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.freq_range = freq_range
        self.x_range = x_range
        self.noise_std = np.sqrt(noise_var)

    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 2
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        amp_list = self.amp_range[0] + np.random.rand(n_funcs,2)*(self.amp_range[1] - self.amp_range[0])
        phase_list = self.phase_range[0] + np.random.rand(n_funcs,2)*(self.phase_range[1] - self.phase_range[0])
        freq_list = self.freq_range[0] + np.random.rand(n_funcs,2)*(self.freq_range[1] - self.freq_range[0])
        for i in range(n_funcs):
            x_samp = self.x_range[0] + np.random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = amp_list[i:i+1]*np.sin(freq_list[i:i+1,:]*x_samp[:,None] + phase_list[i:i+1]) + self.noise_std*np.random.randn(n_samples,2)

            x[i,:,0] = x_samp
            y[i,:,:] = y_samp

        if return_lists:
            return x,y,amp_list,phase_list

        return x,y


class DynamicsTrajDataset(Camelid_Dataset):
    def __init__(self, dynamics, agent):
        self.dynamics = dynamics
        self.agent = agent
        self.o_dim = self.dynamics.ob_dim
        self.u_dim = self.dynamics.u_dim

    def sample(self, n_funcs, n_samples, verbose=False):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))


        for i in tqdm(range(n_funcs), disable=(not verbose)):
            # sim a trajectory
            data = rollout(n_samples, self.agent, self.dynamics)

            x[i,:,:o_dim] = data["states"][:-1,:]
            x[i,:,o_dim:] = data["actions"][:,:]
            y[i,:,:] = data["states"][1:,:]

        return x,y

class DynamicsRandomActionDataset(Camelid_Dataset):
    def __init__(self, dynamics, state_space, action_space, action_scale=1.):
        self.dynamics = dynamics
        self.state_space = state_space
        self.action_space = action_space
        self.action_scale = action_scale
        self.o_dim = self.dynamics.ob_dim
        self.u_dim = self.dynamics.u_dim
        
        
    # TODO: properly handle cases where obs != state
    def sample(self, n_funcs, n_samples, verbose=False):
        o_dim = self.dynamics.ob_dim
        u_dim = self.dynamics.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))


        
        for i in tqdm(range(n_funcs), disable=not verbose):
            self.dynamics.reset()
            s = self.state_space.sample()
            for j in range(n_samples):
                a = self.action_space.sample() * self.action_scale
                sp = self.dynamics.transition(s,a)
                o = self.dynamics.observation(s)
                op = self.dynamics.observation(sp)
                
                x[i,j,:o_dim] = o
                x[i,j,o_dim:] = a
                y[i,j,:] = op
                
                s = sp

        return x,y



class DynamicsRandomDataset(Camelid_Dataset):
    def __init__(self, dynamics, state_space, action_space):
        self.dynamics = dynamics
        self.state_space = state_space
        self.action_space = action_space
        self.o_dim = self.dynamics.ob_dim
        self.u_dim = self.dynamics.u_dim
        
        
    # TODO: properly handle cases where obs != state
    def sample(self, n_funcs, n_samples, verbose=False):
        o_dim = self.o_dim
        u_dim = self.action_space.dim()
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))


        for i in tqdm(range(n_funcs), disable=not verbose):
            self.dynamics.reset()
            for j in range(n_samples):
                s = self.state_space.sample()
                a = self.action_space.sample()
                sp = self.dynamics.transition(s,a)
                o = self.dynamics.observation(s)
                op = self.dynamics.observation(sp)
                
                x[i,j,:o_dim] = o
                x[i,j,o_dim:] = a
                y[i,j,:] = op

        return x,y


# ----
# Sections below are for OpenAI gym integration, which is not supported in
# favor of the camelid specific environment class structure
# ----

# Assumes env has a forward_dynamics(x,u) function
class GymUniformSampleDataset(Camelid_Dataset):
    def __init__(self, env):
        self.env = env
        self.o_dim = env.observation_space.shape[-1]
        self.u_dim = env.action_space.shape[-1]

    def sample(self, n_funcs, n_samples, verbose=False):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        for i in tqdm(range(n_funcs), disable=not verbose):
            self.env.reset()
            for j in range(n_samples):
                s = self.env.get_ob_sample()
                a = self.env.get_ac_sample()
                sp = self.env.forward_dynamics(s,a)

                # TODO make all of the sampling in terms of observations
#                 sp = self.env._get_obs(sp)

                x[i,j,:o_dim] = s
                x[i,j,o_dim:] = a
                y[i,j,:] = sp

        return x,y


# wraps a gym env + policy as a dataset
# assumes that the gym env samples parameters from the prior upon reset
class GymDataset(Camelid_Dataset):
    def __init__(self, env, policy):
        import gym
        self.env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        self.policy = policy
        self.o_dim = env.observation_space.shape[-1]
        self.u_dim = env.action_space.shape[-1]

    def sample(self, n_funcs, n_samples, shuffle=False, verbose=False):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))


        for i in tqdm(range(n_funcs), disable=(not verbose)):
            # sim a trajectory
            x_traj = []
            u_traj = []
            xp_traj = []

            ob = self.env.reset()
            done = False
            while not done:
                ac = self.policy(ob)
                obp, _, done, _ = self.env.step(ac)
                x_traj.append(ob)
                u_traj.append(ac)
                xp_traj.append(obp)

                ob = obp

            T = len(x_traj)
            if T < n_samples:
                print('episode did not last long enough')
                n_samples = T-1

            if shuffle:
                inds_to_keep = np.random.choice(T, n_samples)
            else:
                start_ind = 0 #np.random.randint(T-n_samples)
                inds_to_keep = range(start_ind, start_ind+n_samples)
            x[i,:,:o_dim] = np.array(x_traj)[inds_to_keep,:]
            x[i,:,o_dim:] = np.array(u_traj)[inds_to_keep,:]
            y[i,:,:] = np.array(xp_traj)[inds_to_keep,:]

        return x,y
