from ..core import Space, Dynamics, DynamicsModel, CostFunction
from ..utils import ParameterDistribution
from ..utils.differentiable_cost_funcs import quadratic
import numpy as np
from ..rover_sim.rover_updated import Rover
from ..rover_sim.terrain import FlatTerrain, SinusoidTerrain


class StateSpace(Space):
    #dimension = 1
    def __init__(self, low=None, high=None):
        self.dimension = 1 # velocity
        self.low = [0] if low is None else low
        self.high = [100] if high is None else high
    def sample(self):
        v = np.random.uniform(low=0,high=100)
        return v

state_space = StateSpace()

class ActionSpace(Space):
    #dimension = 6
    def __init__(self, low=None, high=None):
        self.dimension = 6 # slips, sinkages
        self.low = [-0.0016, -0.016, -0.016, 0.0040, 0.0040, 0.0040] if low is None else low
        self.high = [0.240, 0.240, 0.240, 0.0105, 0.0105, 0.0105] if high is None else high
    def sample(self, n_samples=()):
        #return 2*np.random.rand(self.dimension) - 1
        n_samples = ((n_samples,) if isinstance(n_samples, int) else
            tuple(n_samples))
        return np.random.uniform(low=self.low, high=self.high, size=n_samples +
            (self.dimension,))

action_space = ActionSpace()

def makeDefaultRover():
    alpha = np.deg2rad(110)
    beta = np.deg2rad(90)

    # Pick the angles at horizontal level motion
    th1 = np.deg2rad(45)
    th2 = np.deg2rad(60)

    # Find all the arm lengths
    l1 = 2.0
    l2 = 1.0
    rem_ht = l1*np.sin(th1) - l2 * np.cos(alpha + th1 - np.pi/2)
    l3 = rem_ht/np.sin(th2)
    l4 = rem_ht/np.cos(beta + th2 - np.pi/2)

    wheel_rad = 0.4
    wheel_width = 0.1

    body_len = 2.0
    body_wid = 0.2

    rover = Rover(l1, l2, l3, l4, alpha, beta, gamma = th1, wheel_rad = wheel_rad, wheel_width = wheel_width, body_len = body_len, body_wid = body_wid)

    rover.set_terrain(FlatTerrain())

    mass = 10.0 # kg
    g = 9.81 # m/s2
    wheel_MOI = 1.0*(wheel_rad**2.0)/4.0 # kg m^2
    rover.set_inertias(mass = mass, g = g, wheel_MOI = wheel_MOI)

    return rover

class RockerBogieDynamics(Dynamics):
    def __init__(self, randomization={"c", "phi"}, **kwargs):
        self.rover = makeDefaultRover()
        params = ParameterDistribution()

        if "c" in randomization:
            print("c rand")
            params.register("c", 1e3,[0.7e3, 5e3])
        else:
            params.register("c", 1e3)

        if "phi" in randomization:
            print("phi rand")
            params.register("phi", np.deg2rad(30),[np.deg2rad(15),np.deg2rad(45)])
        else:
            params.register("phi", np.deg2rad(30))
            
            
        params.register("x0", np.array([0.]))
        params.register("dt", 0.1)

        # rmse is 0.283
        params.register("noise", np.array([0.00]))  

        # overwrite parameter values with those from the config file
        for key in kwargs:
            value = kwargs[key]
            if value is None: continue
            value = value if "__iter__" in dir(value) else [value]

            # [parameter, [low, high]]
            if len(value) == 2 and "__iter__" in dir(value[1]):
                params.register(key, value[0], [value[1][0], value[1][1]])
            else:
                params.register(key, value)

        super().__init__(parameters=params)

        self.ob_dim = 1
        self.u_dim = 6

    def reset_state(self):
        self.state = self.parameters["x0"]

    # def resample_dynamics(self):
    #     # Print mu and crr values
    #     print("mu = ",self.parameters["mu"])
    #     print("crr = ",self.parameters["crr"])
    #     self.parameters.sample()
    #     print("new mu = ",self.parameters["mu"])
    #     print("new crr = ",self.parameters["crr"])
    
    def observation(self,s):
        # v = s
        return s

    def transition(self,s, u):
        # print("transition in roc_with_terr_sim")
        # print("s looks like",s)
        # print("u looks like",u)
        # These params shouldn't change
        # Loose sand
        # n = 1.1; k_c = 0.9e3; k_phi = 1523.4e3; k = 0.025; c1 = 0.18; c2 = 0.32 
        # Compact sand
        n = 0.47; k_c = 0.9e3; k_phi = 1523.4e3; k = 0.038; c1 = 0.43; c2 = 0.32

        terr_params = [self.parameters["c"], self.parameters["phi"], n, k, k_c, k_phi, n, c1, c2]
        dt = 0.1
        vp = self.rover.transition_vel_only(s, u, terr_params, dt)
        
        if np.isnan(s):
            print("s has nans")
        if np.isnan(vp):
            print("vp has nans")
        return vp

    # def step(self, u):
    #     """
    #     sets self.state to the next state after taking action u from state self.state
    #     returns next observation
    #     """
    #     self.state = self.transition(self.state, u)

    #     return self.observation(self.state)

class SwitchingRockerBogieDynamics(RockerBogieDynamics):
    def __init__(self, dt, randomization={"mu","crr"}, **kwargs):
        super().__init__(dt, randomization={"mu","crr"}, **kwargs)
        self.hazard = 0.05

    def step(self, u):
        self.state = self.transition(self.state, u)

        self.resampling = False
        if np.random.rand() < self.hazard:
            print('resampling dynamics')
            self.resample_dynamics() # dynamics randomization
            self.resampling = True
                  
        return self.observation(self.state)
                  
    def resampled(self):
        return self.resampling

class OracleDynamicsModel(DynamicsModel):
    """
    Wraps RockerBogieDynamics object to act as a dynamics model for a controller.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.ob_dim = 2
        self.u_dim = 3

    def reset(self):
        pass

    def predict_mean(self, x, u):
        return self.dynamics.transition(x,u)

    def batch_linearize(self, x, u):
        # TODO: Not sure if necessary right now
        raise NotImplementedError

class GoalCostFunction(CostFunction):
    def __init__(self, x_g=0., vx_g=0.):
        self.x_g = x_g
        self.vx_g = vx_g


        self.x_cost = 1.
        self.vx_cost = 0.5

        self.control_cost = 0.1
        self.control_cost_offset = 0.0

        self.x_cost_T = 10.
        self.vx_cost_T = 10.

    def stage_cost(self, state, action, time):
        x,vx = state
        cost = 0
        cost += self.x_cost*quadratic(x - self.x_g[time])
        cost += self.vx_cost*quadratic(vx - self.vx_g[time])

        # tau1, tau2, tau3 = action 
        cost += self.control_cost*quadratic(action).sum()

        return cost

    def quadratized_stage_cost(self, state, action, time):
        x,vx = state

        C = self.stage_cost(state, action, time)
        Cx = np.array([
            self.x_cost*quadratic(x - self.x_g[time], d=1),
            self.vx_cost*quadratic(vx - self.vx_g[time], d=1)
        ])
        Cxx = np.diag([
            self.x_cost*quadratic(x - self.x_g[time], d=2),
            self.vx_cost*quadratic(vx - self.vx_g[time], d=2)
        ])
        Cu = self.control_cost*quadratic(action - self.control_cost_offset, d=1)
        Cuu = np.diag(self.control_cost*quadratic(action - self.control_cost_offset, d=2))
        
        # action_dim x state_dim
        Cux = np.zeros((3,2))

        return self.QuadCostXU(C=C, Cx=Cx, Cxx=Cxx, Cu=Cu, Cuu=Cuu, Cux=Cux)

    def terminal_cost(self, state):
        x,vx = state

        cost = 0
        cost += self.x_cost_T*quadratic(x - self.x_g)
        cost += self.vx_cost_T*quadratic(vx - self.vx_g)

        return cost

    def quadratized_terminal_cost(self, state):
        x,vx = state

        C = self.terminal_cost(state)
        Cx = np.array([
            self.x_cost_T*quadratic(x - self.x_g, d=1),
            self.vx_cost_T*quadratic(vx - self.vx_g, d=1)
        ])
        Cxx = np.diag([
            self.x_cost_T*quadratic(x - self.x_g, d=2),
            self.vx_cost_T*quadratic(vx - self.vx_g, d=2)
        ])

        return self.QuadCostX(C=C, Cx=Cx, Cxx=Cxx)


