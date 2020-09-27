# this file implements the core components as abstract classes

# defines an environment (dynamics, cost function)
import collections
from .utils import seeding
from abc import ABC, abstractmethod
from enum import Enum

class Space:
    """
    object that represents either a state space or action space
    """
    def __init__(self):
        pass

    def sample(self):
        """
        returns a sample point from the space
        """
        raise NotImplementedError

    def dim(self):
        return self.dimension

class Dynamics:
    """
    Dynamics implements a forward simulator of a dynamical system.
    This represents the _true_ system dynamics that are used for simulation.
    """
    def __init__(self, parameters):
        """
        parameters: utils.ParameterDistribution
        """
        self.parameters = parameters
        self.ob_dim = None
        self.u_dim = None

    def seed(self, seed=None):
        """
        creates internal np_random rng, seeded at seed
        seeds parameters at the same value
        """
        self.np_random, seed = seeding.np_random(seed)
        self.parameters.seed(seed)
        return [seed]

    def reset(self, randomize=True):
        """
        resets internal state, if randomize==True, randomizes parameters
        returns initial observation
        """
        if randomize:
            self.parameters.sample()

        self.reset_state()
        return self.observation(self.state)

    def reset_state(self):
        """
        sets self.state to the init state
        """
        raise NotImplementedError

    def transition(self, x, u):
        """
        returns the noise-free next state if the current state is x, and the
        current action is u
        """
        raise NotImplementedError

    def observation(self, x):
        """
        returns the observation is the internal state is x
        default is to just use obs = x
        """
        return x

    def step(self, u):
        """
        sets self.state to the next state after taking action u from state self.state
        returns next observation
        """
        self.state = self.transition(self.state, u)

        return self.observation(self.state)


class CostFunction:
    QuadCostXU = collections.namedtuple('QuadCostXU', 'C Cx Cxx Cu Cuu Cux')
    QuadCostX = collections.namedtuple('QuadCostXU', 'C Cx Cxx')

    """
    CostFunction implements a cost function for as environment.
    """
    def stage_cost(self,x,u):
        """
        returns the stage cost when at state x and taking action u
        """
        raise NotImplementedError

    def quadratized_stage_cost(self,x_,u_):
        """
        input: quadratization point x_,u_
        returns: quadratic approximation of stage cost function
                             [ 1]^T[ 2*C Cx^T Cu^T ][ 1]
                 cost ~= 1/2 [dx]  [ Cx  Cxx  Cux^T][dx]
                             [du]  [ Cu  Cux  Cuu  ][du]

        where dx = (x - x_), du = (u - u_)

        returns terms as a QuadCostXU named tuple
        """
        raise NotImplementedError

    def terminal_cost(self,x):
        """
        returns the terminal cost at state x
        """
        raise NotImplementedError

    def quadratized_terminal_cost(self,x_):
        """
        input: quadratization point x_
        returns: quadratic approximation of the terminal function

                          1/ [ 1]^T[ 2*C Cx^T][ 1]
                 cost ~=  /2 [dx]  [ Cx  Cxx ][dx]

        where dx = (x - x_)

        returns terms as a QuadCostX namedtuple
        """
        raise NotImplementedError

    
class DynamicsModel(ABC):
    """
    DynamicsModel represents a model of the environment dynamics that is used by
    Controller to select actions. Adaptive models must implemennt the
    incorporate_transition function, which may be called by Agent as transitions
    are observed on the true system.

    Each controller type call get_model with different ModelType.
    """
    class ModelType(Enum):
        POSTERIOR_PREDICTIVE = 1
        POST_PRED = 1
        MEAN_VAR = 5
        MAP = 2
        SAMPLE = 3
        BATCH_SAMPLE = 4

    @abstractmethod
    def reset(self):
        """
        resets stored statistics to prior statistics
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self, with_grad=False, modeltype=ModelType.POSTERIOR_PREDICTIVE, **kwargs):
        """
        Outputs a function f(x,u) which generally
            - Takes in np.arrays x, u: (..., x_dim), (..., u_dim)
            - Outputs np.arrays mu, sig, corresponding to the model's prediction
              of the next state (mean and variance)
                mu: (..., x_dim)
                sig: (..., x_dim, x_dim)
                
                (if with_grad == True)
                dmu_dx: (..., x_dim, x_dim)
                dmu_du: (..., x_dim, u_dim)
                
            for ModelType.POSTERIOR_PREDICTIVE
                mu, sig should represent the mean and standard deviance of the next state
                according to the current posterior predictive density
            
            for ModelType.MEAN_VAR
                mu is the mean prediction
                var is the covariance of the last layer (as opposed to predictive covariance)
            
            for ModelType.MAP
                mu should represent the MAP prediction of the next state, 
                sig is the process noise
                
            for ModelType.SAMPLE
                samples a model from posterior
                mu should represent the prediction according sample
                sig is the process noise
                
            for ModelType.BATCH_SAMPLE
                samples num_samples models from posterior 
                inputs to f are expected to be (..., num_samples, x_dim/u_dim) shape
                mu should represent (..., num_samples, x_dim) means of prediction 
                    according to samples.
                sig should represent (..., num_samples, x_dim, x_dim) copies of 
                    posterior predictive
        """
        raise NotImplementedError

    def incorporate_transition(self, x, u, xp):
        """
        updates internal state using in the information from the observed state
        transition (x, u ,xp).
        
        if not implemented, this does nothing (model is non adaptive)
        """
        pass
        
class Controller:
    def __init__(self, model, cost):
        """
        the Controller is intialized with a
            model: DynamicsModel
            cost: CostFunction
        """
        self.model = model
        self.cost = cost

    def reset(self):
        """
        resets internal state, including internal timestep
        """
        raise NotImplementedError

    def optimize(self,x):
        """
        optimizes a sequence of actions starting from state x
        stores the optimal sequence internally
        """
        raise NotImplementedError

    def query(self,x,t=0):
        """
        returns action to take at state x and time t

        assumes that optimize has already been called
        """
        raise NotImplementedError

    def step(self):
        """
        called once per timestep of simulation, can be used to update internal counter.
        """
        pass

class Agent:
    """
    an Agent interacts with an environment, taking actions and observing transitions.
    """
    def reset(self):
        """
        resets internal state
        """
        raise NotImplementedError

    def act(self,x):
        """
        returns the action to take at state x
        """
        raise NotImplementedError

    def incorporate_transition(self,x,u,xp):
        """
        updates internal state after observing a transition (x,u,xp)

        unless overridden, this function does nothing.
        """
        pass

    def episode_log(self):
        """
        returns a dictionary of various stats that might be useful for logging

        typically called at the end of each episode

        self.reset should clear the data at the start of a new episode

        unless overridden, this function returns an empty dictionary
        """
        return {}
