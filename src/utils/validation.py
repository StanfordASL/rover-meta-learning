# utilities for validating a learned model
import numpy as np

def action_sweep(dyn_fn, dynamics, x0, u0, sweepdim=0, sweep_range=[-1,1]):
    """
    simulates taking action u from state x0
    u = u0, but u[sweepdim] is varied across sweep_range
    for each action, reports mean and var of each dim of xp
                     both for true dynamics and from model
    input: dyn_fn: (x,u) -> mu, sig
           dynamics: Dynamics class with transition and observation functions
           x0: nominal state
           u0: nominal action
           sweepdim: dim of nominal action to vary
           sweep_range: range over which to vary that input
    output: u_range, range of varied input
            xp_mean_true, xp_std_true: mean and std of each dim of xp from dynamics
            xp_mean_pred, xp_std_pred: mean and std of each dim of xp from dyn_fn
    """
    N_pts = 50
    u_range = np.float64(np.tile(u0[None,:],[N_pts, 1]))
    u_range[:,sweepdim] = np.linspace(sweep_range[0],sweep_range[1],N_pts)

    N_samples = 50
    xp_true = np.zeros([N_samples, N_pts, dynamics.ob_dim])
    for i in range(N_samples):
        dynamics.reset()
        for j in range(N_pts):
            xp_true[i,j,:] = dynamics.observation(dynamics.transition(x0,u_range[j,:]))

    xp_mean_true = np.mean(xp_true, axis=0)
    xp_var_true = np.stack( [np.ones((dynamics.ob_dim,dynamics.ob_dim))*np.cov(xp_true[:,j,:], rowvar=False) for j in range(N_pts)])
    xp_std_true = np.sqrt( np.diagonal(xp_var_true, axis1=1, axis2=2) )
    xp_mean_pred, xp_var_pred = dyn_fn(np.tile(dynamics.observation(x0)[None,:], [N_pts,1]), u_range)
    xp_std_pred = np.sqrt( np.diagonal(xp_var_pred[:,:,:], axis1=1, axis2=2) )

    return u_range[:,sweepdim], xp_mean_true, xp_std_true, xp_mean_pred, xp_std_pred

def state_sweep(dyn_fn, dynamics, x0, u0, sweepdim=0, sweep_range=[-1,1]):
    """
    simulates taking action u0 from state x
    x = x0, but x[sweepdim] is varied across sweep_range
    for each state, reports mean and var of each dim of xp
                     both for true dynamics and from model
    input: dyn_fn: (x,u) -> mu, sig
           dynamics: Dynamics class with transition and observation functions
           x0: nominal state
           u0: nominal action
           sweepdim: dim of nominal action to vary
           sweep_range: range over which to vary that input
    output: u_range, range of varied input
            xp_mean_true, xp_std_true: mean and std of each dim of xp from dynamics
            xp_mean_pred, xp_std_pred: mean and std of each dim of xp from dyn_fn
    """
    N_pts = 50
    x_range = np.float64(np.tile(x0[None,:],[N_pts, 1]))
    x_range[:,sweepdim] = np.linspace(sweep_range[0],sweep_range[1],N_pts)

    o_range = np.vstack([dynamics.observation(x) for x in x_range])
    # print(x_range.shape)
    # print(o_range.shape)
    
    N_samples = 50
    op_true = np.zeros([N_samples, N_pts, dynamics.ob_dim])
    for i in range(N_samples):
        dynamics.reset()
        for j in range(N_pts):
            op_true[i,j,:] = dynamics.observation( dynamics.transition(x_range[j,:], u0) )

    xp_mean_true = np.mean(op_true, axis=0)
    xp_var_true = np.stack( [np.ones((dynamics.ob_dim,dynamics.ob_dim))*np.cov(op_true[:,j,:], rowvar=False) for j in range(N_pts)])
    xp_std_true = np.sqrt( np.diagonal(xp_var_true, axis1=1, axis2=2) )

    xp_mean_pred, xp_var_pred = dyn_fn(o_range, np.tile(u0[None,:], [N_pts,1]))
    # print(xp_var_pred.shape)
    xp_std_pred = np.sqrt( np.diagonal(xp_var_pred[:,:,:], axis1=1, axis2=2) )

    return x_range[:,sweepdim], xp_mean_true, xp_std_true, xp_mean_pred, xp_std_pred


def single_action(dyn_fn, dynamics, x0, u):
    N_samples = 50
    # print("x0 has shape",x0.shape)
    # print("u has shape", u.shape)
    xp_true = np.zeros([N_samples, dynamics.ob_dim])
    for i in range(N_samples):
        dynamics.reset()
        xp_true[i,:] = dynamics.observation(dynamics.transition(x0,u))

    xp_mean_true = np.mean(xp_true, axis=0)
    # print("mean is calculated as", xp_mean_true)
    xp_var_true = np.cov(xp_true, rowvar=False)
    # print("xp_var_true shape",xp_var_true.shape)
    if xp_var_true.ndim <2:
        xp_std_true = np.nan_to_num(np.sqrt(xp_var_true))
    else:
        xp_std_true = np.nan_to_num(np.sqrt(np.diagonal(xp_var_true)))

    # print("x0 is",x0)
    xp_mean_pred, xp_var_pred = dyn_fn(x0, u)
    # print("xp_mean_pred is ",xp_mean_pred)
    if xp_var_pred.ndim <2:
        xp_std_pred = np.sqrt(xp_var_pred)
    else:
        xp_std_pred = np.sqrt(np.diagonal(xp_var_pred))
    # print("xp_std_pred is", xp_std_pred)
    return xp_mean_true, xp_std_true, xp_mean_pred, xp_std_pred

# process saved statistics


def return_stats(data, data_oracle=None):
    total_costs = np.array( [np.sum(d) for d in data["costs"]] )
    if data_oracle is not None:
        total_costs -= np.array( [np.sum(d) for d in data_oracle["costs"]] )
    n = len(total_costs)
    mean = np.mean(total_costs)
    median = np.median(total_costs)
    sample_var = np.mean([(c-mean)**2 for c in total_costs])
    stderr = np.sqrt(sample_var/n)
    return mean, stderr

def return_stats_2(data, data_oracle=None):
    total_costs = np.array( [np.sum(d) for d in data["costs"]] )
    if data_oracle is not None:
        total_costs -= np.array( [np.sum(d) for d in data_oracle["costs"]] )
    n = len(total_costs)
    mean = np.mean(total_costs)
    median = np.median(total_costs)
    fr = calc_failure_rates(data)
    sample_var = np.mean([(c-mean)**2 for c in total_costs])
    stderr = np.sqrt(sample_var/n)
    return mean, stderr, median

def calc_failure_rates(data):
    final_states = [ s[-1,:] for s in data["states"]]
    failures = [ 1.*(np.abs(s[2]-np.pi) > 0.3) for s in final_states ]
    return np.mean(failures)
