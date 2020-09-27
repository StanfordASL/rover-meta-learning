import matplotlib.pyplot as plt
import numpy as np

def plot_admm_error(agent,dynamics,states,actions):
    controller = agent.controller
    primal_err, dual_err = controller.get_consensus_error()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(primal_err,color='r')
    plt.plot(dual_err, color='b')
    plt.legend(['Primal residual', 'Dual residual'])
    ax.set_yscale('log')
    plt.show()


def plot_actions(agent,dynamics,states,actions):
    controller = agent.controller

    u = controller.get_action_predictions()
    
    N = u.shape[1]
    T = u.shape[0]
    for n in range(N):
        plt.plot(range(T),u.cpu()[0:,n,0,0],color='b', alpha=0.3)
        plt.plot(range(T),u.cpu()[0:,n,1,0],color='r', alpha=0.3)
        
    plt.show()
        

def plot_linear_predictions(agent, dynamics, states, actions):
    model = agent.model
    controller = agent.controller
    
    # grab predictions from controller
    s0 = [s[0,0] for s in states]
    s2 = [s[2,0] for s in states]
    
    x = controller.get_particle_predictions()
    
    N = x.shape[1]
    for n in range(N):
        # TODO(james): this should take the current time index for plotting
        plt.plot(x[0:,n,0,0,],x[0:,n,2,0],color='b', alpha=0.3)
    plt.plot(s0[:-1],s2[:-1],color='r')
    plt.scatter(s0[-2],s2[-2],color='r', marker='o')
#     plt.axis([-2,8,-4,4])
    plt.axis([-8,8,-8,8])
    plt.show()

def plot_traj(axes, rollout_data, color='C0', linestyle='-', alpha=1.0, t0=0):
    """
    Plots each componennt of a trajectory and the actions taken
    
    axes is a list of obdim + udim matplotlib axes to plot on 
        (created, e.g from a plt.subplots call)
        
    rollout_data is a dictionary with keys:
        'states' -> array shape (H, obdim)
        'actions' -> shape (H-1, udim)
    
    t0: int of initial timestep of trajectory
    """        
    traj = np.array(rollout_data["states"])
    u_list = np.array(rollout_data["actions"])

    H,ob_dim = traj.shape
    u_dim = u_list.shape[-1]

    for i in range(ob_dim+u_dim):
        ax = axes[i]
        if i < ob_dim:
            ax.plot(np.arange(t0,H+t0),traj[:,i], color=color, linestyle=linestyle, alpha=alpha)
        else:
            ax.plot(np.arange(t0,t0+H-1),u_list[:,i-ob_dim], color=color, linestyle=linestyle, alpha=alpha)

def plot_mean_var(axes, mean_states, var_states, actions, color='C0'):
    """
    Plots each componennt of a trajectory with variance and the actions taken
    
    axes is a list of obdim + udim matplotlib axes to plot on 
        (created, e.g from a plt.subplots call)
        
    mean_states: np.array shape (H, obdim)
    var_states: np.array shape (H, obdim, obdim)
    actions: np.array shape (H-1, udim)

    """
    std_states = np.sqrt(np.diagonal(var_states, axis1=1, axis2=2))

    H,ob_dim = mean_states.shape
    u_dim = actions.shape[-1]

    for i in range(ob_dim+u_dim):
        ax = axes[i]
        if i < ob_dim:
            ax.fill_between(np.arange(0,H), mean_states[:,i] - 1.96*std_states[:,i], mean_states[:,i] + 1.96*std_states[:,i], color=color, alpha=0.2)
            ax.plot(np.arange(0,H),mean_states[:,i], color=color)
        else:
            ax.plot(np.arange(0,H-1),actions[:,i-ob_dim], color=color)

def plot_mean_var_from_rollouts(axes, rollouts, actions, color='C0'):
    """
    Plots each componennt of a trajectory with mean variance and the actions taken
    from a collection of trajectories with the same actions
    
    axes is a list of obdim + udim matplotlib axes to plot on 
        (created, e.g from a plt.subplots call)
        
    rollouts: np.array shape (K, H, obdim)
    actions: np.array shape (H-1, udim)

    """
    xdim = rollouts.shape[-1]
    rollouts = np.stack(rollouts)
    mean_states = np.mean(rollouts, axis=0)
    var_states = np.stack([np.ones((xdim,xdim))*np.cov(rollouts[:,t,:], rowvar=False) for t in range(rollouts[0].shape[0])])

    plot_mean_var(axes, mean_states, var_states, actions, color=color)


def plot_mean_conf_sweeps(axes, x_pts, y, y_std, color='C0', label=""):
    """
    plots mean and 95% conf intervals given
        x_pts: np.array (N)
        y_pts: np.array (N, ydim)
        y_std: np.array (N, ydim)
        
    plots on axes, list of ydim matplotlib Axes
    """
    y_dim = y.shape[-1]
    for i in range(y_dim):
        ax = axes[i]
        ax.fill_between(x_pts, y[:,i] - 1.96*y_std[:,i], y[:,i] + 1.96*y_std[:,i], color=color, alpha=0.2)
        ax.plot(x_pts, y[:,i], color=color, label=label)


# The following functions need to be updated to take in axes objects
        
        
def plot_trajs_from_data(data, color='C0', linestyle='-', alpha=1.0):
    """
    assumes data is a dict with the following keys:
    data = {
    "states": (N_sim x H x x_dim)
    "actions": (N_sim x H x u_dim)
    }
    """    
    traj = np.array(data["states"])
    u_list = np.array(data["actions"])

    N,H,ob_dim = traj.shape
    u_dim = u_list.shape[-1]

    alpha = 1.0/N

    for i in range(ob_dim+u_dim):
        plt.subplot(ob_dim+u_dim,1,i+1)
        for j in range(N):
            if i < ob_dim:
                plt.plot(np.arange(0,H),traj[j,:,i], color=color, linestyle=linestyle, alpha=alpha)
                plt.ylim([-2,2])
            else:
                plt.plot(np.arange(0,H-1),u_list[j,:,i-ob_dim], color=color, linestyle=linestyle, alpha=alpha)
                plt.ylim([-2,2])

def plot_all_trajs_from_data(data1, data2, data3, st_color='C0', ac_color='C2', linestyle='-', alpha=1.0, ylimits=None):
    """
    assumes data is a dict with the following keys:
    data = {
    "states": (N_sim x H x x_dim)
    "actions": (N_sim x H x u_dim)
    }
    """
    for k in range(3):

        if k == 0:
            data = data1
        elif k==1:
            data = data2
        elif k==2:
            data = data3


        traj = np.array(data["states"])
        u_list = np.array(data["actions"])

        N,H,ob_dim = traj.shape
        u_dim = u_list.shape[-1]

        alp = alpha/N

        for i in range(ob_dim+u_dim):
            ax = plt.subplot(ob_dim+u_dim,3,3*i + k+1)
            if ylimits is not None:
                ax.set_ylim(ylimits[i])

            for j in range(N):
                if i != ob_dim+u_dim-1:
                    ax.set_xticklabels([])

                if k > 0:
                    ax.set_yticklabels([])
                elif i < ob_dim:
                    plt.ylabel(r'$x_{%d}$' % (i+1))
                else:
                    plt.ylabel(r'$u_{&d}$' % (i-ob_dim+1))

                if ob_dim == 6 and i == 2:
                    plt.plot([0,50],[0.5,0.5],color='k',linestyle=':')


                if i < ob_dim:
                    plt.plot(np.arange(0,H),traj[j,:,i], color=st_color, linestyle=linestyle, alpha=alp)
                else:

                    plt.plot(np.arange(0,H-1),u_list[j,:,i-ob_dim], color=ac_color, linestyle=linestyle, alpha=alp)

        plt.xlabel(r'$t$')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        
# PAPER FIGURES


def plot_mean_conf(x, y_samps, color='C0'):
    """
    computes mean and std of y_samps across dim 0
    """
    y = np.mean(y_samps, axis=0)
    y_std = np.sqrt( np.var(y_samps, axis=0) )

    plt.fill_between(x, y - 1.96*y_std, y + 1.96*y_std, color=color, alpha=0.2)
    h, = plt.plot(x, y, color=color)
    return h

def gen_RMSE_curve_fig(data, noise_floor, filename=None, ylims=None):
    T = data["ce_alpaca"]["rmses"].shape[-1]

    h1 = plot_mean_conf(np.arange(T),data["ce_alpaca"]["rmses"], color="C0")
    h2 = plot_mean_conf(np.arange(T),data["ce_maml"]["rmses"], color="C1")
    h3, = plt.plot([0,T-1],[noise_floor,noise_floor], color='k', linestyle=':')
    plt.xlim(0,T-1)
    plt.ylabel("RMSE", fontsize=16)
    plt.xlabel("$t$", fontsize=16)
    plt.legend([h1,h2,h3],[r"\textbf{ALPaCA (Ours)}", "MAML", "Process Noise Floor"], fontsize=14)

    if ylims is not None:
        plt.ylim(ylims)

    if filename is not None:
        plt.savefig(filename)
