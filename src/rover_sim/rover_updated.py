import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import lsq_linear
from scipy.optimize import root
from numpy import cos, sin
# from .terrain import *

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

    # rover.set_terrain(FlatTerrain())

    mass = 10.0 # kg
    g = 9.81 # m/s2
    wheel_MOI = 1.0*(wheel_rad**2.0)/4.0 # kg m^2
    rover.set_inertias(mass = mass, g = g, wheel_MOI = wheel_MOI)

    return rover

class Rover():
    def __init__(self,l1, l2, l3, l4, alpha, beta, gamma, wheel_rad = 0.4, wheel_width = 0.1, body_len = None, body_wid = None):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3 
        self.l4 = l4 
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma
        self.wheel_rad = wheel_rad
        self.wheel_width = wheel_width
        self.body_len = body_len
        self.body_wid = body_wid
    
    def set_terrain(self, terr):
        self.terrain = terr
    
    def set_inertias(self, mass, g, wheel_MOI):
        self.mass = mass
        self.g = g
        self.wheel_MOI = wheel_MOI

    def __z_center_wheel(self, x):
        if not hasattr(self, 'terrain'):
            print("No terrain specified")
            z_gnd = 0.0
            grad = 0.0
        else:
            z_gnd = self.terrain.heightAt(x)
            grad = self.terrain.gradient(x)
        z_center = z_gnd + self.wheel_rad * np.cos(np.arctan(grad))
        return z_center

    def __func_th2(self, th2, x2, z2):
        l3 = self.l3 
        l4 = self.l4 
        beta = self.beta            
        x3 = x2 + l3*np.cos(th2) + l4*np.cos(np.pi - beta - th2)
        z3_gnd = self.__z_center_wheel(x3)
        z3_kin = z2 + l3*np.sin(th2) - l4*np.sin(np.pi - beta - th2)
        return z3_gnd - z3_kin
    
    def __func_th1(self, th1, xb, zb):
        l1 = self.l1
        l2 = self.l2
        alpha = self.alpha
        x1 = xb - l2*np.cos(np.pi - alpha - th1) - l1*np.cos(th1)
        z1_gnd = self.__z_center_wheel(x1)
        z1_kin = zb + l2*np.sin(np.pi - alpha - th1) - l1*np.sin(th1)
        return z1_gnd - z1_kin

    def __find_angles(self, x2):
        if type(x2)==torch.Tensor:
            z2 = self.__z_center_wheel(x2)
            th2_guess = np.deg2rad(50) # guess
            th2_guess = torch.full_like(x2,th2_guess)
            th2_soln = fsolve(self.__func_th2, th2_guess, args=(x2.flatten(), z2.flatten()))
            th2 = th2_soln.reshape(x2.shape)
            xb = x2 + self.l3*np.cos(th2)
            zb = z2 + self.l3*np.sin(th2)
            th1_guess = np.deg2rad(50) # guess
            th1_guess = torch.full_like(x2,th1_guess)
            th1_soln = fsolve(self.__func_th1, th1_guess, args=(xb.flatten(), zb.flatten()))
            th1 = th1_soln.reshape(x2.shape)
            return th1, th2
        else:
            z2 = self.__z_center_wheel(x2)
            th2_guess = np.deg2rad(50) # guess
            th2 = fsolve(self.__func_th2, th2_guess, args=(x2, z2))[0]
            xb = x2 + self.l3*np.cos(th2)
            zb = z2 + self.l3*np.sin(th2)
            th1_guess = np.deg2rad(50) # guess
            th1 = fsolve(self.__func_th1, th1_guess, args=(xb, zb))[0]
            return th1, th2
        

    def find_geom(self, x2):
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3 
        l4 = self.l4 
        alpha = self.alpha
        beta = self.beta
        th1, th2 = self.__find_angles(x2)
        z2 = self.__z_center_wheel(x2)
        xb = x2 + l3*np.cos(th2)
        zb = z2 + l3*np.sin(th2)
        x3 = x2 + l3*np.cos(th2) + l4*np.cos(np.pi - beta - th2)
        z3 = z2 + l3*np.sin(th2) - l4*np.sin(np.pi - beta - th2)
        z3_gnd = self.__z_center_wheel(x3)
        if type(z3)==torch.Tensor:
            assert (abs(z3-z3_gnd) <= 0.01).byte().all(), "z3 not compatible with terrain"
        else:
            assert abs(z3-z3_gnd) <= 0.01, "z3 not compatible with terrain"
        x0 = xb - l2*np.cos(np.pi - alpha - th1)
        z0 = zb + l2*np.sin(np.pi - alpha - th1)
        x1 = xb - l2*np.cos(np.pi - alpha - th1) - l1*np.cos(th1)
        z1 = zb + l2*np.sin(np.pi - alpha - th1) - l1*np.sin(th1)
        z1_gnd = self.__z_center_wheel(x1)
        if type(z1)==torch.Tensor:
            assert (abs(z1-z1_gnd) <= 0.01).byte().all(), "z1 not compatible with terrain"
        else:
            assert abs(z1-z1_gnd) <= 0.01, "z1 not compatible with terrain"
        r0 = (x0,z0)
        r1 = (x1,z1)
        r2 = (x2,z2)
        r3 = (x3,z3)
        rb = (xb,zb)
        return r0, r1, rb, r2, r3

    
    def plot_rover(self, x2, wheel_rad = None, body_wid = None, body_len = None):
        [r0, r1, rb, r2, r3] = self.find_geom(x2)
        if wheel_rad is None:
            wheel_rad = self.wheel_rad
        if body_len is None:
            body_len = self.body_len
        if body_wid is None:
            body_wid = self.body_wid
        fig, ax = plt.subplots(1)

        col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        col1 = col_list[0]

        if body_len is not None and body_wid is not None:
            # Plot body
            body_rect = plt.Rectangle((r0[0] + body_len/2, r0[1] ), width = body_wid, height = body_len, angle = 90, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(body_rect)

        # Plot linkages
        ax.plot((r0[0],r1[0]), (r0[1],r1[1]), linewidth = 4.0, color = col1)
        ax.plot((r0[0],rb[0]), (r0[1],rb[1]), linewidth = 4.0, color = col1)
        ax.plot((rb[0], r2[0]), (rb[1],r2[1]), linewidth = 4.0, color = col1)
        ax.plot((rb[0], r3[0]), (rb[1],r3[1]), linewidth = 4.0, color = col1)

        if wheel_rad is not None:
            wheel_rad_1 = wheel_rad
            wheel_rad_2 = wheel_rad
            wheel_rad_3 = wheel_rad
            # Plot wheels
            wheel_circle_1 = plt.Circle((r1[0],r1[1]), wheel_rad_1, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(wheel_circle_1)
            wheel_circle_2 = plt.Circle((r2[0],r2[1]), wheel_rad_2, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(wheel_circle_2)
            wheel_circle_3 = plt.Circle((r3[0],r3[1]), wheel_rad_3, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(wheel_circle_3)

        if hasattr(self, 'terrain'):
            xs = np.arange(-5,5)
            level_gnd = [self.terrain.heightAt(x) for x in xs]
            ax.plot(xs,level_gnd, linewidth = 4.0, color = 'brown')

        ax.axis('equal')        
        return ax
    
    def __find_slope_alphas(self, r1, r2, r3):
        alpha1 = np.arctan(self.terrain.gradient(r1[0]))
        alpha2 = np.arctan(self.terrain.gradient(r2[0]))
        alpha3 = np.arctan(self.terrain.gradient(r3[0]))
        return alpha1, alpha2, alpha3
    
    def apply_torques(self, x2, Taus):
        l1 = self.l1
        l2 = self.l2 
        l3 = self.l3
        l4 = self.l4
        rad = self.wheel_rad
        alpha = self.alpha
        beta = self.beta 

        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.__find_slope_alphas(r1, r2, r3)
        th1, th2 = self.__find_angles(x2)
        mass = self.mass
        g = self.g
        if not self.mass>0:
            print("ERROR: Rover mass not specified.")

        [tau1, tau2, tau3] = Taus
        T1 = tau1/rad
        T2 = tau2/rad
        T3 = tau3/rad

        ux = -rad*sin(alpha1) + l1*cos(th1) - l2*cos(th1+self.alpha)
        uy = rad*cos(alpha1) + l1*sin(th1) - l2*sin(th1+self.alpha)
        vx = -rad*sin(alpha2) + l3*cos(th2)
        vy = -rad*cos(alpha2) + l3*cos(th2)
        wx = -rad*sin(alpha3) + l4*cos(th2+beta)
        wy = rad*cos(alpha3) + l4*sin(th2+beta)
        zx = -l2*cos(th1+alpha)
        zy = -l2*sin(th1+alpha)

        M = np.array([[-1, -sin(alpha1), -sin(alpha2), -sin(alpha3)],
                    [0, cos(alpha1), cos(alpha2), cos(alpha3)],
                    [-zy, -sin(alpha1)*uy -cos(alpha1)*ux, 0, 0],
                    [0, 0, -sin(alpha2)*vy -cos(alpha2)*vx, -sin(alpha3)*wy -cos(alpha3)*wx]])
        X = np.array([[-T1*cos(alpha1) -T2*cos(alpha2) -T3*cos(alpha3)],
                    [-T1*sin(alpha1) -T2*sin(alpha2) -T3*sin(alpha3) + mass*g],
                    [-mass*g*zx - (T1*cos(alpha1)*uy - T1*sin(alpha1)*ux)],
                    [-(T2*cos(alpha2)*vy -T2*sin(alpha2)*vx +T3*cos(alpha3)*wy -T3*sin(alpha3)*wx)]])
        
        # f is the 4x1 vector: f[0]=rover body force Fxnet, f[1:]=normal forces on the wheels N1, N2, N3
        f = np.matmul(np.linalg.inv(M),X)
    
        [Fxnet, N1, N2, N3] = np.squeeze(f)
        Ns = np.array([N1, N2, N3])
        Ts = np.array([T1, T2, T3])

        return Fxnet, Ns, Ts
    
    def find_NTTau_given_slips_sinkages(self, i, th1, terr_params):
        [c, phi, n, k, k_c, k_phi, n, c1, c2] = terr_params
        r = self.wheel_rad
        b = self.wheel_width
        thm = (c1+i*c2)*th1 # eqn 5
        # thm = th1/2 # approximation
        th2 = 0.0   # eqn 6 
        sigma_m = (k_c/b + k_phi)*(r*(np.cos(thm) - np.cos(th1)))**n # eqn 9, simp
        A = 1- np.exp(-(r/k)*(th1 - thm - (1-i)*(np.sin(th1) - np.sin(thm)))) # eqn 11
        tau_m = (c+sigma_m*np.tan(phi))*A # eqn 10 simp
        term0 = r*b/(thm * (th1 - thm))
        term1 = -thm * np.cos(th1) + th1 * np.cos(thm) - th1 + thm
        term2 = thm * np.sin(th1) - th1*np.sin(thm)
        term3 = th1*np.sin(thm) - thm*np.sin(thm) - thm*th1 + thm**2
        term4 = th1*np.cos(thm) - thm*np.cos(thm) +thm - th1
        N_out = term0*(sigma_m * term1 - tau_m * term2 - c*term3)
        T_out = term0*(sigma_m * term2 + tau_m * term1 - c * term4)
        Tau_out = r**2 * b/2.0 * (tau_m * th1 + c * thm)
        return [N_out, T_out, Tau_out]
    
    def sim_fn_vector(self, inp, N, T, Tau, terr_params, verbose = False):
        i = inp[0]
        th1 = inp[1]
        [N_out, T_out, Tau_out] = self.find_NTTau_given_slips_sinkages(i, th1, terr_params)
        # Compute errors
        N_error = N_out - N
        T_error = T_out - T
        Tau_error = Tau_out - Tau
        N_percent_error = 100*(N_out - N)/N
        T_percent_error = 100*(T_out - T)/T
        Tau_percent_error = 100*(Tau_out - Tau)/Tau
        if verbose:
            print("N_out", N_out)
            print("T_out", T_out)
            print("Tau_out", Tau_out)
            print("N_percent_error",N_percent_error)
            print("T_percent_error",T_percent_error)
            print("Tau_percent_error",Tau_percent_error)
        return [N_error, T_error, Tau_error]
    
    def sim_fn(self, inp, N, T, Tau, terr_params, verbose = False):
        i = inp[0]
        th1 = inp[1]
        [N_out, T_out, Tau_out] = self.find_NTTau_given_slips_sinkages(i, th1, terr_params)
        # Compute errors
        N_error = N_out - N
        T_error = T_out - T
        Tau_error = Tau_out - Tau
        N_percent_error = 100*(N_out - N)/N
        T_percent_error = 100*(T_out - T)/T
        Tau_percent_error = 100*(Tau_out - Tau)/Tau
        if verbose:
            print("N_out", N_out)
            print("T_out", T_out)
            print("Tau_out", Tau_out)
            print("N_percent_error",N_percent_error)
            print("T_percent_error",T_percent_error)
            print("Tau_percent_error",Tau_percent_error)
        # residual = np.linalg.norm([N_percent_error, T_percent_error, Tau_percent_error], ord = np.inf)
        residual = N_error**2 + T_error**2 + Tau_error**2
        return residual

    def find_slip_sinkage_given_NTTau(self, N, T, Tau, i_prev, th1_prev, terr_params, verbose = False):
        ''' For single wheel, compute slip, sinkage (and th1), given N T Tau '''
        
        i_soln = None
        th1_soln = None 
        z_m = None  

        guess_1 = [i_prev, th1_prev] # i, th1
        i_nom_guess = 0.02
        th1_nom_guess = 0.2
        guess_2 = [i_prev, th1_nom_guess]
        guess_3 = [i_nom_guess, th1_nom_guess]
        guess_4 = [0.008, 0.16]
        init_guess_list = [guess_1, guess_2, guess_3, guess_4]
        num_guesses = np.shape(init_guess_list)[0]

        pot_solns = np.zeros((num_guesses, 2))
        pot_values = np.zeros(num_guesses)

        # bounds on i and th1
        bnds = ((0.0, 1.0), (0.0, np.pi/2.0))
        
        for i in range(num_guesses):
            init_guess = init_guess_list[i]
            res = root(self.sim_fn_vector, init_guess, args=(N, T, Tau, terr_params), method='lm')
            # res = minimize(self.sim_fn, init_guess, args=(N, T, Tau, terr_params), bounds=bnds)
            pot_solns[i,:] = res.x
            pot_values[i] = np.max(res.fun)
        if np.isnan(pot_values).all():
            print("ERROR: all were nan")
            print("N",N,"T",T, "Tau", Tau)
            ind = np.nanargmin(pot_values) # Warning: might throw ValueError if all NaNs encountered
        else:
            ind = np.nanargmin(pot_values) 
        [i_soln, th1_soln] = pot_solns[ind, :]
        z_m = self.find_sinkage_given_th1(self.wheel_rad, th1_soln)
        return [i_soln, th1_soln, z_m]

    def find_sinkage_given_th1(self, r, th1):
        z_m = r*(1-np.cos(th1))
        return z_m

    def find_th1_given_sinkage(self, r, z_m):
        th1 = np.arccos(1- z_m/r)
        return th1

    # EXAMPLE INTERFACE
    ### State: x2 (position of middle wheel? of rover), velocity
    # state = [x2, vel]
    ### Action: torques applied (3), slips, sinkages
    # torques = [Tau1, Tau2, Tau3]
    # slips = [i1, i2, i3]
    # sinkages = [z1, z2, z3]
    # action = np.hstack([torques, slips, sinkages])
    ### Params: terrain params
    # Loose Sand parameters
    # n = 1.1; c = 1.0e3; phi = np.deg2rad(30); k_c = 0.9e3; k_phi = 1523.4e3; k = 0.025; c1 = 0.18; c2 = 0.32 
    # Compact sand parameters
    # n = 0.47; k_c = 0.9e3; k_phi = 1523.4e3; k = 0.038; c1 = 0.43; c2 = 0.32 c = 0.69e3 phi = np.deg2rad(33)
    # terr_params=[c, phi, n, k, k_c, k_phi, n, c1, c2]
    ### Constants: time diff
    # constants = {
    #     "dt": 0.1
    # }

    def find_consistent_slips_sinkages(self, state, torques, terr_params, prev_slips=None, prev_sinkages=None):
        # Finds the slips and sinkages that are consistent with this state and torques
        x2 = state[0]
        vel = state[1]
        Taus = torques
        Fxnet, Ns, Ts = self.apply_torques(x2, Taus)
        slips = np.zeros(3)
        sinkages = np.zeros(3)
        if prev_slips is None:
            prev_slips = np.array([0.001,0.001,0.001])
        if prev_sinkages is None:
            prev_sinkages = np.array([0.02,0.02,0.02])
        for wheel in range(3):
            N = Ns[wheel]
            T = Ts[wheel]
            Tau = Taus[wheel]
            i_prev = prev_slips[wheel]
            th1_prev = self.find_th1_given_sinkage(self.wheel_rad, prev_sinkages[wheel])
            [i_soln, th1_soln, z_m] = self.find_slip_sinkage_given_NTTau(N, T, Tau, i_prev, th1_prev, terr_params)
            slips[wheel] = i_soln
            sinkages[wheel] = z_m
        return [slips, sinkages]
    
    def find_consistent_slips_sinkages_vel_only(self, v, torques, terr_params, prev_slips=None, prev_sinkages=None):
        # Finds the slips and sinkages that are consistent with this state and torques
        x2 = 0.0
        Taus = torques
        Fxnet, Ns, Ts = self.apply_torques(x2, Taus)
        slips = np.zeros(3)
        sinkages = np.zeros(3)
        if prev_slips is None:
            prev_slips = np.array([0.001,0.001,0.001])
        if prev_sinkages is None:
            prev_sinkages = np.array([0.02,0.02,0.02])
        for wheel in range(3):
            N = Ns[wheel]
            T = Ts[wheel]
            Tau = Taus[wheel]
            i_prev = prev_slips[wheel]
            th1_prev = self.find_th1_given_sinkage(self.wheel_rad, prev_sinkages[wheel])
            [i_soln, th1_soln, z_m] = self.find_slip_sinkage_given_NTTau(N, T, Tau, i_prev, th1_prev, terr_params)
            slips[wheel] = i_soln
            sinkages[wheel] = z_m
        return [slips, sinkages]
    
    def get_full_action(self, state, torques, terr_params, prev_slips=None, prev_sinkages=None):
        [slips, sinkages] = self.find_consistent_slips_sinkages(state, torques, terr_params, prev_slips, prev_sinkages)
        full_action = np.hstack([torques, slips, sinkages])
        return full_action
    
    def get_full_action_vel_only(self, v, torques, terr_params, prev_slips=None, prev_sinkages=None):
        [slips, sinkages] = self.find_consistent_slips_sinkages_vel_only(v, torques, terr_params, prev_slips, prev_sinkages)
        full_action = np.hstack([torques, slips, sinkages])
        return full_action
    
    def transition(self, state, full_action, terr_params, dt):
        # print("state looks like", state)
        # print("full_action looks like", full_action)
        if np.isnan(state).any():
            print("state has nans")
        if np.isnan(full_action).any():
            print("full_action has nans")
        if np.isnan(terr_params).any():
            print("terr_params has nans")
        x2 = state[0]
        vel = state[1]
        slips = full_action[3:6]
        sinkages = full_action[6:9]
        Ns = np.zeros(3)
        Ts = np.zeros(3)
        Taus = np.zeros(3)
        for wheel in range(3):
            i = slips[wheel]
            th1 = self.find_th1_given_sinkage(self.wheel_rad, sinkages[wheel])
            if np.isnan(i):
                print("i is nans")
            if np.isnan(th1):
                print("th1 is nans")
            [Ns[wheel], Ts[wheel], Taus[wheel]] = self.find_NTTau_given_slips_sinkages(i, th1, terr_params)
            if np.isnan(Ns[wheel]):
                print("Ns for wheel", wheel,"is nans")
                print("i=", i)
                print("th1=", th1)
                print("terr_params=", terr_params)
        if np.isnan(Ns).any():
            print("Ns has nans")
            print("x=", state)
            print("u=", full_action)
            print("terr_params=", terr_params)
        if np.isnan(Ts).any():
            print("Ts has nans")
        if np.isnan(Taus).any():
            print("Taus has nans")
        # print("True Ns ", Ns)
        # print("True Ts ", Ts)
        # print("True Taus ", Taus)
        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.__find_slope_alphas(r1, r2, r3)
        S = np.array([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
        Fxnet = S @ np.hstack([Ns, Ts])
        acc_rover = Fxnet/self.mass
        vel_new = vel + acc_rover*dt
        x2_new = x2 + vel*dt
        state_new = np.array([x2_new, vel_new])
        # noise = np.array([.01, .01])
        # return state_new + noise*np.random.randn(*np.shape(state_new))
        return state_new
    
    def transition_vel_only(self, x, u, terr_params, dt):
        # print("trans vel")
        # print("x looks like", x)
        # print("u looks like", u)
        if np.isnan(x):
            print("x has nans")
        if np.isnan(u).any():
            print("u has nans")
        if np.isnan(terr_params).any():
            print("terr_params has nans")
        vel = x
        slips = u[0:3]
        sinkages = u[3:6]
        Ns = np.zeros(3)
        Ts = np.zeros(3)
        Taus = np.zeros(3)
        for wheel in range(3):
            i = slips[wheel]
            th1 = self.find_th1_given_sinkage(self.wheel_rad, sinkages[wheel])
            if np.isnan(i):
                print("i is nans")
            if np.isnan(th1):
                print("th1 is nans")
            [Ns[wheel], Ts[wheel], Taus[wheel]] = self.find_NTTau_given_slips_sinkages(i, th1, terr_params)
            if np.isnan(Ns[wheel]):
                print("Ns for wheel", wheel,"is nans")
                print("i=", i)
                print("th1=", th1)
                print("terr_params=", terr_params)
        if np.isnan(Ns).any():
            print("Ns has nans")
            print("x=", state)
            print("u=", full_action)
            print("terr_params=", terr_params)
        if np.isnan(Ts).any():
            print("Ts has nans")
        if np.isnan(Taus).any():
            print("Taus has nans")
        # print("True Ns ", Ns)
        # print("True Ts ", Ts)
        # print("True Taus ", Taus)
        x2 = 0.0
        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.__find_slope_alphas(r1, r2, r3)
        S = np.array([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
        Fxnet = S @ np.hstack([Ns, Ts])
        acc_rover = Fxnet/self.mass
        vel_new = vel + acc_rover*dt
        # print("returning vel", vel_new)
        return vel_new
    
    def get_full_action_and_transition(self, state, torques, terr_params, dt, prev_slips=None, prev_sinkages=None):
        full_action = self.get_full_action(state, torques, terr_params, prev_slips, prev_sinkages)
        state_new = self.transition(state, full_action, terr_params, dt)
        return full_action, state_new
    
    def get_fnom_function(self, x, u, terr_params, dt):
        phi_nom_matrix = self.get_lin_model_features_matrix(x, u, terr_params, dt)
        # print("x shape", x.shape, "sliced shape", phi_nom_matrix[...,2].shape)
        fnom = x + phi_nom_matrix[...,2] # both rows, last column : shape should be (2,)
        return fnom
    
    def get_fnom_function_vel_only(self, v, u, terr_params, dt):
        phi_nom_matrix = self.get_lin_model_features_matrix_vel_only(v,u, terr_params, dt)
        # print("phi_nom_matrix shape",phi_nom_matrix.shape)
        # print("v shape", v.shape, "sliced shape", phi_nom_matrix[...,0,2].shape)
        if v.ndim != (phi_nom_matrix[...,0,2]).ndim:
            fnom = v + phi_nom_matrix[...,0:1,2]
            return fnom
        else:
            fnom = v + phi_nom_matrix[...,0,2] # 1st row, 3rd column, should be (1,)
            return fnom.squeeze()
   
    def get_c_tanphi_features_matrix(self, x, u, terr_params, dt):
        phi_nom_matrix = self.get_lin_model_features_matrix(x, u, terr_params, dt)
        c_tanphi_features_matrix = phi_nom_matrix[...,0:2] # both rows, 1st and 2nd column, should be (2,2))
        return c_tanphi_features_matrix
    
    def get_c_tanphi_features_matrix_vel_only(self, v, u, terr_params, dt):
        phi_nom_matrix = self.get_lin_model_features_matrix_vel_only(v,u, terr_params, dt)
        c_tanphi_features_matrix = phi_nom_matrix[...,0:1,0:2] # 1st (and only) row, 1st and 2nd column, should be (1,2))
        # multiply the first column entry by 1000 to normalize c value
        c_tanphi_features_matrix[...,0:1,0] = 1000* c_tanphi_features_matrix[...,0:1,0]
        return c_tanphi_features_matrix

    def get_lin_model_features_matrix(self, x, u, terr_params, dt):
        [_, _, n, k, k_c, k_phi, n, c1, c2] = terr_params

        m = self.mass
        r = self.wheel_rad
        b = self.wheel_width

        if type(x) == torch.Tensor:
            if x.ndim == 1:
                x2 = x[0]
                vel = x[1]
                slips = u[3:6]
                sinkages = u[6:9]
                x_dim = x.shape[-1]
                Q = torch.zeros((3,3))
                R = torch.zeros((3,3))
            elif x.ndim == 2:
                x2 = x[:,0]
                vel = x[:,1]
                slips = u[:,3:6]
                sinkages = u[:,6:9]
                x_dim_0 = x.shape[0]
                x_dim = x.shape[-1]
                Q = torch.zeros((x_dim_0, 3,3))
                R = torch.zeros((x_dim_0, 3,3))
            elif x.ndim == 3:
                x2 = x[:,:,0]
                vel = x[:,:,1]
                slips = u[:,:,3:6]
                sinkages = u[:,:,6:9]
                x_dim_0 = x.shape[0]
                x_dim_1 = x.shape[1]
                x_dim = x.shape[-1]
                Q = torch.zeros((x_dim_0, x_dim_1, 3,3))
                R = torch.zeros((x_dim_0, x_dim_1, 3,3))
            else:
                print("ERROR: expected tensor dimension is 1, 2 or 3, not ", x.ndim)
                raise ValueError("ERROR: expected tensor dimension is 2 or 3, not {xdim}".format(xdim = x.ndim))
        else:
            x2 = x[0]
            vel = x[1]
            slips = u[3:6]
            sinkages = u[6:9]
            x_dim = x.shape[0]
            Q = np.zeros((3,3))
            R = np.zeros((3,3))

        for wheel in range(3):
            if type(x) == torch.Tensor:
                if x.ndim == 1:
                    i = slips[wheel]
                    z = sinkages[wheel]
                elif x.ndim == 2:
                    i = slips[:,wheel]
                    z = sinkages[:,wheel]
                elif x.ndim == 3:
                    i = slips[:,:,wheel]
                    z = sinkages[:,:,wheel]
            else:
                i = slips[wheel]
                z = sinkages[wheel]
            th1 = self.find_th1_given_sinkage(r, z)
            thm = (c1+i*c2)*th1 # eqn 5
            sigma_m = (k_c/b + k_phi)*(r*(np.cos(thm) - np.cos(th1)))**n # eqn 9, simp
            A = 1- np.exp(-(r/k)*(th1 - thm - (1-i)*(np.sin(th1) - np.sin(thm)))) # eqn 11

            # Define f0-f4
            f0 = r*b/(thm*(th1-thm))
            f1 = th1*cos(thm) - thm*cos(th1) + thm - th1 
            f2 = thm*sin(th1) - th1*sin(thm)
            f3 = th1*sin(thm) - thm*sin(thm) - thm*th1 +thm**2
            f4 = th1*cos(thm) - thm*cos(thm) + thm - th1

            if type(x) == torch.Tensor:
                if x.ndim == 1:
                    # N = Q * [c; tan(phi); 1]
                    Q[wheel,:] = torch.stack([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1], dim=-1)
                    # T = R * [c; tan(phi); 1]
                    R[wheel,:] = torch.stack([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2], dim=-1)
                elif x.ndim == 2:
                    # N = Q * [c; tan(phi); 1]
                    Q[:,wheel,:] = torch.stack([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1], dim=-1)
                    # T = R * [c; tan(phi); 1]
                    R[:,wheel,:] = torch.stack([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2], dim=-1)
                elif x.ndim == 3:
                    # N = Q * [c; tan(phi); 1]
                    Q[:,:,wheel,:] = torch.stack([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1], dim=-1)
                    # T = R * [c; tan(phi); 1]
                    R[:,:,wheel,:] = torch.stack([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2], dim=-1)
            else:
                # N = Q * [c; tan(phi); 1]
                Q[wheel,:] = np.array([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1])
                # T = R * [c; tan(phi); 1]
                R[wheel,:] = np.array([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2])

        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.__find_slope_alphas(r1, r2, r3)
        

        # Fx, N, T values
        # Fx = np.array([c, np.tan(phi), 1.0]) @ np.hstack([Q.T,R.T]) @ S
        # Ns = np.array([c, np.tan(phi), 1.0]) @ Q.T
        # Ts = np.array([c, np.tan(phi), 1.0]) @ R.T

        if type(x) == torch.Tensor:
            if x.ndim == 1:
                if type(alpha1) == np.float64:
                    S = torch.tensor([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
                elif type(alpha1) == torch.Tensor:
                    S = torch.stack([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)], dim=-1)
                phi_nom_matrix = torch.zeros((x_dim, 3))
                Qtrans = Q.permute(1,0)
                Rtrans = R.permute(1,0)
                QTRTcat = torch.cat([Qtrans,Rtrans],dim = -1)
                part2b = (vel*dt).unsqueeze(-1).unsqueeze(-1)
                phi_nom_matrix[0,:] = part2b.long().squeeze(-1)
                part1 = dt/m * QTRTcat @ S.unsqueeze(-1).float()
                phi_nom_matrix[1,:] = part1.squeeze(-1)
            elif x.ndim == 2:
                S = torch.stack([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)], dim=-1)
                phi_nom_matrix = torch.zeros((x_dim_0, x_dim, 3))
                Qtrans = Q.permute(0,2,1)
                Rtrans = R.permute(0,2,1)
                QTRTcat = torch.cat([Qtrans,Rtrans],dim = -1)
                part2b = (vel*dt).unsqueeze(-1).unsqueeze(-1)
                phi_nom_matrix[:,0,:] = part2b.long().squeeze(-1)
                part1 = dt/m * QTRTcat @ S.unsqueeze(-1).float()
                phi_nom_matrix[:,1,:] = part1.squeeze(-1)
            elif x.ndim ==3:
                S = torch.stack([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)], dim=-1)
                phi_nom_matrix = torch.zeros((x_dim_0, x_dim_1, x_dim, 3))
                Qtrans = Q.permute(0,1,3,2)
                Rtrans = R.permute(0,1,3,2)
                QTRTcat = torch.cat([Qtrans,Rtrans],dim = -1)
                part2b = (vel*dt).unsqueeze(-1).unsqueeze(-1)
                phi_nom_matrix[:,:,0,:] = part2b.long().squeeze(-1)
                part1 = dt/m * QTRTcat @ S.unsqueeze(-1).float()
                phi_nom_matrix[:,:,1,:] = part1.squeeze(-1)
        else:
            S = np.array([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
            phi_nom_matrix = np.zeros((x_dim, 3))
            phi_nom_matrix[0,:] = (vel*dt) * np.array([0,0,1])
            phi_nom_matrix[1,:] = dt/m * np.hstack([Q.T,R.T]) @ S 
        # shape = [..., x_dim, 3]
        return phi_nom_matrix
    
    def get_lin_model_features_matrix_vel_only(self, x, u, terr_params, dt):
        [_, _, n, k, k_c, k_phi, n, c1, c2] = terr_params

        m = self.mass
        r = self.wheel_rad
        b = self.wheel_width

        if type(x) == torch.Tensor:
            if x.ndim == 1:
                vel = x[0]
                slips = u[0:3]
                sinkages = u[3:6]
                x_dim = x.shape[-1]
                Q = torch.zeros((3,3))
                R = torch.zeros((3,3))
            elif x.ndim == 2:
                vel = x[:,0]
                slips = u[:,0:3]
                sinkages = u[:,3:6]
                x_dim_0 = x.shape[0]
                x_dim = x.shape[-1]
                Q = torch.zeros((x_dim_0, 3,3))
                R = torch.zeros((x_dim_0, 3,3))
            elif x.ndim == 3:
                vel = x[:,:,0]
                slips = u[:,:,0:3]
                sinkages = u[:,:,3:6]
                x_dim_0 = x.shape[0]
                x_dim_1 = x.shape[1]
                x_dim = x.shape[-1]
                Q = torch.zeros((x_dim_0, x_dim_1, 3,3))
                R = torch.zeros((x_dim_0, x_dim_1, 3,3))
            else:
                print("ERROR: expected tensor dimension is 1, 2 or 3, not ", x.ndim)
                raise ValueError("ERROR: expected tensor dimension is 2 or 3, not {xdim}".format(xdim = x.ndim))
        else:
            vel = x[0]
            slips = u[0:3]
            sinkages = u[3:6]
            x_dim = x.shape[0]
            Q = np.zeros((3,3))
            R = np.zeros((3,3))

        for wheel in range(3):
            if type(x) == torch.Tensor:
                if x.ndim == 1:
                    i = slips[wheel]
                    z = sinkages[wheel]
                elif x.ndim == 2:
                    i = slips[:,wheel]
                    z = sinkages[:,wheel]
                elif x.ndim == 3:
                    i = slips[:,:,wheel]
                    z = sinkages[:,:,wheel]
            else:
                i = slips[wheel]
                z = sinkages[wheel]
            th1 = self.find_th1_given_sinkage(r, z)
            thm = (c1+i*c2)*th1 # eqn 5
            sigma_m = (k_c/b + k_phi)*(r*(np.cos(thm) - np.cos(th1)))**n # eqn 9, simp
            A = 1- np.exp(-(r/k)*(th1 - thm - (1-i)*(np.sin(th1) - np.sin(thm)))) # eqn 11

            # Define f0-f4
            f0 = r*b/(thm*(th1-thm))
            f1 = th1*cos(thm) - thm*cos(th1) + thm - th1 
            f2 = thm*sin(th1) - th1*sin(thm)
            f3 = th1*sin(thm) - thm*sin(thm) - thm*th1 +thm**2
            f4 = th1*cos(thm) - thm*cos(thm) + thm - th1

            if type(x) == torch.Tensor:
                if x.ndim == 1:
                    # N = Q * [c; tan(phi); 1]
                    Q[wheel,:] = torch.stack([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1], dim=-1)
                    # T = R * [c; tan(phi); 1]
                    R[wheel,:] = torch.stack([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2], dim=-1)
                elif x.ndim == 2:
                    # N = Q * [c; tan(phi); 1]
                    Q[:,wheel,:] = torch.stack([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1], dim=-1)
                    # T = R * [c; tan(phi); 1]
                    R[:,wheel,:] = torch.stack([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2], dim=-1)
                elif x.ndim == 3:
                    # N = Q * [c; tan(phi); 1]
                    Q[:,:,wheel,:] = torch.stack([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1], dim=-1)
                    # T = R * [c; tan(phi); 1]
                    R[:,:,wheel,:] = torch.stack([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2], dim=-1)
            else:
                # N = Q * [c; tan(phi); 1]
                Q[wheel,:] = np.array([-f0*(f3+A*f2),   -f0*sigma_m*A*f2,   f0*sigma_m*f1])
                # T = R * [c; tan(phi); 1]
                R[wheel,:] = np.array([f0*(-f4+A*f1),   f0*sigma_m*A*f1,   f0*sigma_m*f2])

        x2 = 0.0
        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.__find_slope_alphas(r1, r2, r3)
        

        # Fx, N, T values
        # Fx = np.array([c, np.tan(phi), 1.0]) @ np.hstack([Q.T,R.T]) @ S
        # Ns = np.array([c, np.tan(phi), 1.0]) @ Q.T
        # Ts = np.array([c, np.tan(phi), 1.0]) @ R.T

        if type(x) == torch.Tensor:
            if x.ndim == 1:
                if type(alpha1) == np.float64:
                    S = torch.tensor([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
                elif type(alpha1) == torch.Tensor:
                    S = torch.stack([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)], dim=-1)
                phi_nom_matrix = torch.zeros((x_dim, 3))
                Qtrans = Q.permute(1,0)
                Rtrans = R.permute(1,0)
                QTRTcat = torch.cat([Qtrans,Rtrans],dim = -1)
                part1 = dt/m * QTRTcat @ S.unsqueeze(-1).float()
                phi_nom_matrix[0,:] = part1.squeeze(-1)
            elif x.ndim == 2:
                if type(alpha1) == np.float64:
                    S = torch.tensor([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
                elif type(alpha1) == torch.Tensor:
                    S = torch.stack([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)], dim=-1)
                phi_nom_matrix = torch.zeros((x_dim_0, x_dim, 3))
                Qtrans = Q.permute(0,2,1)
                Rtrans = R.permute(0,2,1)
                QTRTcat = torch.cat([Qtrans,Rtrans],dim = -1)
                part1 = dt/m * QTRTcat @ S.unsqueeze(-1).float()
                phi_nom_matrix[:,0,:] = part1.squeeze(-1)
            elif x.ndim ==3:
                if type(alpha1) == np.float64:
                    S = torch.tensor([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
                elif type(alpha1) == torch.Tensor:
                    S = torch.stack([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)], dim=-1)
                phi_nom_matrix = torch.zeros((x_dim_0, x_dim_1, x_dim, 3))
                Qtrans = Q.permute(0,1,3,2)
                Rtrans = R.permute(0,1,3,2)
                QTRTcat = torch.cat([Qtrans,Rtrans],dim = -1)
                part1 = dt/m * QTRTcat @ S.unsqueeze(-1).float()
                phi_nom_matrix[:,:,0,:] = part1.squeeze(-1)
        else:
            S = np.array([-sin(alpha1), -sin(alpha2), -sin(alpha3), cos(alpha1), cos(alpha2),  cos(alpha3)])
            phi_nom_matrix = np.zeros((x_dim, 3))
            phi_nom_matrix[0,:] = dt/m * np.hstack([Q.T,R.T]) @ S 
        # shape = [..., x_dim, 3]
        return phi_nom_matrix
    
    def estimate_params_lin_model_vel_only(self, x_list, u_list, terr_params, dt):
        x_dim = 1
        param_dim = 3
        N = u_list.shape[0] # equivalently x_list.shape[0]-1
        A = np.zeros((N,x_dim, param_dim)) # A is all matrices M_i of shape x_dim x param_dim 1x3 stacked up to shape (N,3)
        for i in range(N):
            x = x_list[i,:]
            u = u_list[i,:]
            phi_nom_matrix = self.get_lin_model_features_matrix_vel_only(x, u, terr_params, dt)
            A[i,:,:] = phi_nom_matrix
        A = A.reshape((N*x_dim, param_dim)) # A is of shape (N,3)
        b = x_list[1:,:]- x_list[0:-1,:] # difference xp - x list
        b = b.flatten() # should be of shape (N,)
        ## numpy unbounded lstsq
        lstsq_vec = np.linalg.lstsq(A, b)[0] # est_params is of shape (3,)
        ## scipy bounded lstsq
        # lb = [0.8e3, -np.Inf, 0.9]
        # ub = [1.3e3, np.Inf, 1.1]
        # lstsq_vec = lsq_linear(A, b, bounds = (lb, ub)).x # est_params is of shape (3,)
        # change tan(phi) to phi
        est_params_vec = lstsq_vec
        est_params_vec[1] = np.arctan(lstsq_vec[1])
        # Fx, N, T values
        # Fx = np.array([c, np.tan(phi), 1.0]) @ np.hstack([Q.T,R.T]) @ S
        # Ns = np.array([c, np.tan(phi), 1.0]) @ Q.T
        # Ts = np.array([c, np.tan(phi), 1.0]) @ R.T
        return est_params_vec

    def estimate_params_lin_model(self, x_list, u_list, terr_params, dt):
        x_dim = x_list.shape[-1]
        param_dim = 3
        N = u_list.shape[0] # equivalently x_list.shape[0]-1
        A = np.zeros((N,x_dim, param_dim)) # A is all matrices M_i of shape x_dim x param_dim 2x3 stacked up to shape (2N,3)
        for i in range(N):
            x = x_list[i,:]
            u = u_list[i,:]
            phi_nom_matrix = self.get_lin_model_features_matrix(x, u, terr_params, dt)
            A[i,:,:] = phi_nom_matrix
        A = A.reshape((N*x_dim, param_dim)) # A is of shape (2N,3)
        b = x_list[1:,:]- x_list[0:-1,:] # difference xp - x list
        b = b.flatten() # should be of shape (2N,)
        ## numpy unbounded lstsq
        lstsq_vec = np.linalg.lstsq(A, b)[0] # est_params is of shape (3,)
        ## scipy bounded lstsq
        # lb = [0.8e3, -np.Inf, 0.9]
        # ub = [1.3e3, np.Inf, 1.1]
        # lstsq_vec = lsq_linear(A, b, bounds = (lb, ub)).x # est_params is of shape (3,)
        # change tan(phi) to phi
        est_params_vec = lstsq_vec
        est_params_vec[1] = np.arctan(lstsq_vec[1])
        # Fx, N, T values
        # Fx = np.array([c, np.tan(phi), 1.0]) @ np.hstack([Q.T,R.T]) @ S
        # Ns = np.array([c, np.tan(phi), 1.0]) @ Q.T
        # Ts = np.array([c, np.tan(phi), 1.0]) @ R.T
        return est_params_vec


    def predict_dynamics_lin_model(self, x, u, terr_params, dt):
        [c, phi, _, _, _, _, _, _, _] = terr_params
        phi_nom_matrix = self.get_lin_model_features_matrix(x, u, terr_params, dt)
        xp = x +  phi_nom_matrix @ np.array([c, np.tan(phi), 1.0])
        return xp  

    def predict_dynamics_lin_model_vel_only(self, x, u, terr_params, dt):
        [c, phi, _, _, _, _, _, _, _] = terr_params
        phi_nom_matrix = self.get_lin_model_features_matrix_vel_only(x, u, terr_params, dt)
        xp = x +  phi_nom_matrix @ np.array([c, np.tan(phi), 1.0])
        return xp     

    def estimate_params_Iag(self, x, xp, u, dt, terr_params):
        ''' Estimates terrain parameters c phi '''
        x2_init = x[0]
        vel_init = x[1]

        x2_new = xp[0]
        vel_new = xp[1]

        Taus = u[0:3]
        slips = u[3:6]
        sinkages = u[6:9]
        
        [_, _, n, k, k_c, k_phi, n, c1, c2] = terr_params

        # Get true Ns and Ts
        Ns = np.zeros(3)
        Ts = np.zeros(3)
        Taus = np.zeros(3)
        for wheel in range(3):
            i = slips[wheel]
            th1 = self.find_th1_given_sinkage(self.wheel_rad, sinkages[wheel])
            [Ns[wheel], Ts[wheel], Taus[wheel]] = self.find_NTTau_given_slips_sinkages(i, th1, terr_params)
        # print("Iag thinks Ns ", Ns)
        # print("Iag thinks Ts ", Ts)
        # print("Iag thinks Taus ", Taus)

        # Use sinkage, slip, Ns, and Taus to estimate c and phi
        z = np.array(sinkages) 
        i = np.array(slips) 
        r = self.wheel_rad
        b = self.wheel_width
        th1 = np.arccos(1 - z/r)
        A = 1 - np.exp(-(r/k)*(th1/2 + (1-i)*(-np.sin(th1) + np.sin(th1/2))))
        # Iagnemma eqn 23
        kappa_1 = A*(th1**2*Ns*r + 4*Taus*np.sin(th1) - 8*Taus*np.sin(th1/2))
        kappa_2 = 4*Taus*(np.cos(th1) - 2*np.cos(th1/2) +1)
        kappa_3 = A*th1*r**2*b*(np.sin(th1) - 4*np.sin(th1/2) + th1)
        kappa_4 = th1*r**2*b*(np.cos(th1) - 2*np.cos(th1/2) + 2*A*np.cos(th1) - 4*A*np.cos(th1/2) + 2*A + 1)
        K1 = kappa_2/kappa_4
        K2 = np.transpose(np.vstack([np.ones(np.size(K1)),-kappa_1/kappa_4]))
        x, _, _, _ = np.linalg.lstsq(K2, K1)
        c_est = x[0]
        phi_est = np.arctan(x[1])
        return np.array([c_est, phi_est])

    def estimate_params_Iag_vel_only(self, x, xp, u, dt, terr_params):
        ''' Estimates terrain parameters c phi '''
        vel_init = x
        vel_new = xp

        slips = u[0:3]
        sinkages = u[3:6]
        
        [_, _, n, k, k_c, k_phi, n, c1, c2] = terr_params

        # Get true Ns and Ts
        Ns = np.zeros(3)
        Ts = np.zeros(3)
        Taus = np.zeros(3)
        for wheel in range(3):
            i = slips[wheel]
            th1 = self.find_th1_given_sinkage(self.wheel_rad, sinkages[wheel])
            [Ns[wheel], Ts[wheel], Taus[wheel]] = self.find_NTTau_given_slips_sinkages(i, th1, terr_params)
        # print("Iag thinks Ns ", Ns)
        # print("Iag thinks Ts ", Ts)
        # print("Iag thinks Taus ", Taus)

        # Use sinkage, slip, Ns, and Taus to estimate c and phi
        z = np.array(sinkages) 
        i = np.array(slips) 
        r = self.wheel_rad
        b = self.wheel_width
        th1 = np.arccos(1 - z/r)
        A = 1 - np.exp(-(r/k)*(th1/2 + (1-i)*(-np.sin(th1) + np.sin(th1/2))))
        # Iagnemma eqn 23
        kappa_1 = A*(th1**2*Ns*r + 4*Taus*np.sin(th1) - 8*Taus*np.sin(th1/2))
        kappa_2 = 4*Taus*(np.cos(th1) - 2*np.cos(th1/2) +1)
        kappa_3 = A*th1*r**2*b*(np.sin(th1) - 4*np.sin(th1/2) + th1)
        kappa_4 = th1*r**2*b*(np.cos(th1) - 2*np.cos(th1/2) + 2*A*np.cos(th1) - 4*A*np.cos(th1/2) + 2*A + 1)
        K1 = kappa_2/kappa_4
        K2 = np.transpose(np.vstack([np.ones(np.size(K1)),-kappa_1/kappa_4]))
        x, _, _, _ = np.linalg.lstsq(K2, K1)
        c_est = x[0]
        phi_est = np.arctan(x[1])
        return np.array([c_est, phi_est])
     
    # Helper functions 
    def get_random_torques(self, num, two_phase = False):
        if two_phase:
            mid = np.int64(np.floor(num/2))
            Taus = np.vstack([np.random.uniform(low = 0.9, high = 1.2, size = (mid,3)),np.random.uniform(low = 0.3, high = 0.6, size = (num-mid,3))])
        else:
            Taus = np.random.uniform(low = 0.6, high = 0.9, size = (num,3))
        return Taus

    def get_random_terrain_params(self, num, terr_params):
        c_true = np.random.uniform(low = 1e3, high = 1.5e3, size = num)
        phi_true = np.random.uniform(low = np.deg2rad(10), high = np.deg2rad(45), size = num)
        [_, _, n, k, k_c, k_phi, n, c1, c2] = terr_params
        terr_params_list = np.zeros((num, 9))
        for i in range(num):
            terr_params_list[i,:] = [c_true[i], phi_true[i], n, k, k_c, k_phi, n, c1, c2]
        return terr_params_list, c_true, phi_true
