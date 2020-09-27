import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from .torch_dynamics import TorchAdaptiveDynamics
from torch.utils.tensorboard import SummaryWriter
import datetime

from copy import deepcopy

def get_encoder(config,inp_dim):
    activation = nn.Tanh()
    hid_dim = config['hidden_dim']
    phi_dim = config['phi_dim']
    x_dim = config['x_dim']
    
    encoder = nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        activation,
        nn.Linear(hid_dim, hid_dim),
        activation,
        nn.Linear(hid_dim, x_dim*phi_dim),
#         activation
    )
    return encoder

base_config = {
    # model parameters
    'x_dim': None,
    'u_dim': None,
    'hidden_dim': 128,
    'phi_dim': 128,
    'sigma_eps': [0.1]*4,
    'L_asym_corr_rank': 32, # lambda_inv is parameterized as full rank diagonal + lower rank correction
    
    # training parameters
    'learnable_noise': False,
    'max_context': 80,
    'data_horizon': 100,
    'learning_rate': 1e-3,
    'tqdm': True,
    'learning_rate_decay': False,
    'lr_decay_rate': 1e-3,
    'sigma_eps_scale_init': 1.0,
    'sigma_eps_annealing': 0, # rate by which to decrease sig eps, 0 means just use min
    # multistep training params
    'multistep_training': True, 
    'num_mmd_samples': 25,
    'kernel_bandwidth': 0.01,
    'learnable_bandwidth': False,
    'compute_posterior': True,
    'max_evaluation_length': 10,
    'multistep_curriculum': False,
    'curriculum_step': 250,
    'grad_clip_value':1,
    'enforce_feature_orthogonality':False,
    'bound_params':False,
    'ortho_weight':3.33e-2
}


class Alpaca(nn.Module):
    """
    Not set up to run as an independent model within camelid
    For an independent standalone model, use adaptiveDynamicsTorch
    """
    def __init__(self, config={}, prior_belief=None, model_path=None, phi_nom_fn=None, num_aug_features=0):
        super().__init__()

        self.config = deepcopy(base_config)
        if model_path is not None:
            data = torch.load(model_path)
            config = data["config"]        
            
        self.config.update(config)

        self.x_dim = self.config['x_dim']
        self.u_dim = self.config['u_dim']

        self.phi_dim = self.config['phi_dim'] #TODO(james): move to a more expressive config param for layers
        if phi_nom_fn!=None:
            self.phi_aug_dim = self.phi_dim + num_aug_features #TODO(somrita): pass this in
            self.phi_nom_fn = phi_nom_fn
            self.add_nom_features = True
        else:
            self.phi_aug_dim = self.phi_dim
            self.add_nom_features = False

        self.y_dim = self.x_dim

        self.sigma_scale = 1. # used to artificially scale sigma eps when computing loss in train mode
        self.sigma_eps = self.config['sigma_eps']
        self.logSigEps = nn.Parameter(torch.from_numpy(np.log(self.sigma_eps)).float(), requires_grad=self.config['learnable_noise'])

        self.length_scale = nn.Parameter(torch.tensor(self.config['kernel_bandwidth']), requires_grad=self.config['learnable_bandwidth'])
        
        if prior_belief is not None:
            Q_init, Linv_init = prior_belief
            self.Q = nn.Parameter(torch.tensor(Q_init, dtype=torch.float32),requires_grad=False)

            L_asym_init = np.linalg.cholesky(Linv_init)
            self.L_asym = nn.Parameter(torch.tensor(L_asym_init, dtype=torch.float32), requires_grad=False)

        else:
            self.Q = nn.Parameter(torch.randn(self.phi_aug_dim, 1)*4/(np.sqrt(self.phi_aug_dim)+ np.sqrt(self.y_dim)))
            L_asym_corr_rank = self.phi_aug_dim #self.config['L_asym_corr_rank']
            self.L_asym = nn.Parameter(torch.randn(self.phi_aug_dim, L_asym_corr_rank)/self.phi_aug_dim**2)
            self.L_base = nn.Parameter( torch.linspace(-5,0, self.phi_aug_dim), requires_grad=True)

        self.normal_nll_const = self.y_dim*np.log(2*np.pi)
    
        self.backbone = get_encoder(self.config, self.x_dim + self.u_dim)
        
        if model_path is not None:
            print("loading state dict")
            self.load_state_dict(data['state_dict'])
        
    @property
    def L(self):
        return self.L_asym @ self.L_asym.transpose(-2,-1) + torch.diag_embed( torch.exp( self.L_base ) )

    @property
    def logdetSigEps(self):
        return torch.sum(self.logSigEps)

    @property
    def invSigEps(self):
        return torch.diag(torch.exp(-self.logSigEps))

    @property
    def invSigEpsVec(self):
        return torch.exp(-self.logSigEps)

    @property
    def SigEpsVec(self):
        return torch.exp(self.logSigEps)

    @property
    def SigEps(self):
        return torch.diag(torch.exp(self.logSigEps))

    def encoder(self, x):
        # shapes last output of phi to be ydim x phidim x 1
        
        phi = self.backbone(x)
        phi_reshape = phi.view(*(phi.size()[:-1]),self.y_dim,self.phi_dim)
        return phi_reshape

    
    def prior_params(self):
        
        Q0 = self.Q
        L0 = self.L

        return (Q0, L0)

    def recursive_update(self, phi, y, params):
        """
            inputs: phi: shape (..., y_dim, phi_aug_dim, 1)
                    y:   shape (..., y_dim )
                    params: tuple of Q, Linv
                        Q: shape (..., y_dim, 1, phi_aug_dim)
                        Linv: shape (..., y_dim, phi_aug_dim, phi_aug_dim)
        """
        Q, L = params
        invSigEps = self.invSigEps
        L = torch.transpose(phi,-1,-2) @ invSigEps @ phi + L
        Q = Q + torch.transpose(phi,-1,-2) @ invSigEps @ y.unsqueeze(-1)

        return (Q, L)

    def log_predictive_prob(self, phi, y, posterior_params, update_params=False):
        """
            input:  phi: shape (..., phi_aug_dim)
                    y: shape (..., y_dim)     (note: y ~= K.T phi, not xp)
                    posterior_params: tuple of Q, Linv:
                        Q: shape (..., y_dim, 1, phi_aug_dim)
                        Linv: shape (..., y_dim, phi_aug_dim, phi_aug_dim)
                    update_params: bool, whether to perform recursive update on
                                   posterior params and return updated params
            output: logp: log p(y | x, posterior_parms)
                    updated_params: updated posterior params after factoring in (x,y) pair
        """
        Q, L = posterior_params
        SigEps = self.SigEps

        Linv = torch.inverse(L)
        
        K = Linv @ Q #(..., y_dim, 1, phi_aug_dim)
        predictive_cov = phi @ Linv @ torch.transpose(phi,-1,-2) + SigEps

        err = y.unsqueeze(-1)  - (phi @ K)

        inv_sig = torch.inverse(predictive_cov)
        nll_quadform = (torch.transpose(err,-1,-2) @ inv_sig @ err).squeeze(-1).squeeze(-1)
        nll_logdet = - torch.logdet(inv_sig)

        logp = -0.5*(self.normal_nll_const + nll_quadform + nll_logdet)
        
        if update_params:
            updated_params = self.recursive_update(phi,y,posterior_params)
            return logp, updated_params

        return logp


    def forward(self, x, u, posterior_params):
        """
            input: x, u, posterior params
            output: xp
        """
        z = torch.cat([x,u], dim=-1)
        phi = self.encoder(z)
        if self.add_nom_features:
            phi_nom = self.phi_nom_fn(x,u)
            phi = self.augment_phi(phi, phi_nom)

        Q, L = posterior_params

        Linv = torch.inverse(L)

        K = Linv @ Q
        SigEps = self.SigEps
        predictive_cov = phi @ Linv @ torch.transpose(phi,-1,-2) + SigEps
        mu = ( phi @ K).squeeze(-1)
                
        return mu, predictive_cov
    
    def augment_phi(self, phi, phi_nom):
        phi_aug = torch.cat([phi,phi_nom],-1)
        return phi_aug

    

class AlpacaDynamics(TorchAdaptiveDynamics):
    """
    Wrapper class that maps torch alpaca with training functions and
    online prediction and adaptation functions.
    """
    def __init__(self, model, f_nom=None, cuda=-1):
        """
        Inputs:
        model: alpacaTorch object
        f_nom: function mapping tensors x,u -> to tensor xp
        
        Sets up SummaryWriter to log to for Tensorboard visualization.
        """
        super().__init__()
        self.f_nom = f_nom
        if f_nom == None:
            self.f_nom = lambda x,u: x

        self.model = model
        self.reset()

        self.ob_dim = self.model.x_dim
        self.u_dim = self.model.u_dim
        
        # used for annealing during training
        self.train_step = 0
        self.model.sigma_scale = self.model.config['sigma_eps_scale_init']
        
        path = 'alpacatorchMH'
        self.writer = SummaryWriter('./runs/' + path + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s'))

        # set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=1 - self.model.config['lr_decay_rate'] )

        # used to control curriculum for multistep training
        self.total_train_itrs = 0
        self.cuda = cuda
        
    def reset(self):
        self.params = self.model.prior_params()

    def to_cuda(self,x):
        if self.cuda<0:
            return x.cpu()
        else:
            return x.cuda(self.cuda)
        
    def get_K_cov(self):
        sig = self.model.SigEpsVec.reshape(-1,1,1).repeat(1,self.model.phi_aug_dim, self.model.phi_aug_dim)
        return sig * Linv
    
    def get_model_torch(self, modeltype=TorchAdaptiveDynamics.ModelType.POSTERIOR_PREDICTIVE, n_samples=10, with_opt=False):
        """
        returns function mapping x,u to mu, sig of next_state
        (all torch tensors)
        """
        if modeltype == self.ModelType.POSTERIOR_PREDICTIVE:
            def f(x,u):
                mu, sig = self.model(x,u, self.params)
                mu = mu.squeeze(-1)
                if self.f_nom(x,u).ndim != mu.ndim:
                    mu += self.f_nom(x,u).squeeze()
                else:
                    mu += self.f_nom(x,u)
                return mu, sig
            
        elif modeltype == self.ModelType.MAP:
            raise NotImplementedError
                
        elif modeltype  == self.ModelType.SAMPLE:
            raise NotImplementedError

        elif modeltype == self.ModelType.BATCH_SAMPLE:
            raise NotImplementedError
        
        else:
            print(type, "model sampling not implemented.")
            raise NotImplementedError

        return f

    def incorporate_transition_torch_(self, x, u, xp):
        """
        updates self.params after conditioning on transition (x,u,xp)
        """
        self.params = self.incorporate_transition_torch(self.params,x,u,xp)
        return self.params

    def incorporate_transition_torch(self, params, x, u, xp):
        """
        returns posterior params after updating params with transition x, u, xp
        """
        z = torch.cat([x,u], dim=-1)
        phi = self.model.encoder(z)
        if self.model.add_nom_features:
            phi_nom = self.model.phi_nom_fn(x,u)
            phi = self.model.augment_phi(phi, phi_nom)

        y = xp - self.f_nom(x,u)

        # TODO should write this using recursive_update, avoid computing predictive prob
        _, params = self.model.log_predictive_prob(phi, y, params, update_params=True)
        return params
        

    # TRAINING FUNCTIONS
    def evaluate_sample_singlestep(self, sample):
        """
        uses model to evaluate a sample from the dataloader
        conditions on some number of data points before evaluating the rest with the posterior
        mean over time horizon (dim 1), mean over batch (dim 0). returns a scalar
        """
        horizon = self.model.config['data_horizon']
        max_context = self.model.config['max_context']
        
        x = self.to_cuda(sample['x'].float())
        u = self.to_cuda(sample['u'].float())
        xp = self.to_cuda(sample['xp'].float())
        
        z = torch.cat([x,u], dim=-1)

        # batch compute features and targets for BLR
        phi = self.model.encoder(z)
        if self.model.add_nom_features:
            phi_nom = self.model.phi_nom_fn(x,u)
            phi = self.model.augment_phi(phi, phi_nom)            

        y = xp - self.f_nom(x,u)

        # get prior statistics
        stats = self.model.prior_params()
        stats = [p.unsqueeze(0) for p in stats] # add dim for batch eval
        
        # evaluate everything under prior
        prior_stats = [p.unsqueeze(1) for p in stats] 
        prior_logp = self.model.log_predictive_prob(phi, y, prior_stats, update_params=False).mean(1)

        # compute log probs after conditioning
        logps = []
        
        # loop over context data, and condition BLR
        for j in range(max_context):
            phi_ = phi[:,j,...]
            y_ = y[:,j,:]

            # get posterior likelihood
            logp, stats = self.model.log_predictive_prob(phi_, y_, stats, update_params=True)
            logps.append(logp)

        # batch eval remaining data under posterior
        stats = [p.unsqueeze(1) for p in stats] # add dim for batch over time
        if max_context < horizon:
            phi_ = phi[:,max_context:,...]
            y_ = y[:,max_context:,:]

            logp = self.model.log_predictive_prob(phi_, y_, stats, update_params=False).sum(1)
            logps.append(logp)

        total_logp = torch.stack(logps, dim=1).sum(1) / horizon

        enforce_feature_orthogonality = self.model.config['enforce_feature_orthogonality']
        if enforce_feature_orthogonality:
            # Calculate orthogonality loss
            if phi.ndim == 4:
                N, T, x_dim, phi_aug_dim = phi.shape
                ortho_loss = torch.tensor(0.0)
                for n in range(N):
                    sum_over_x_dim = 0
                    for i in range(x_dim):
                        outer = torch.zeros((phi_aug_dim,phi_aug_dim))
                        for t in range(T):
                            outer += torch.ger(phi[n,t,i,:].squeeze(), phi[n,t,i,:].squeeze())
                        outer = outer/T
                        mat = torch.eye(phi_aug_dim) - outer
                        mat_fro_sq = torch.norm(mat,p='fro')**2
                        sum_over_x_dim += mat_fro_sq
                    ortho_loss += sum_over_x_dim/x_dim
                ortho_loss = ortho_loss/N
            else:
                print("ERROR: not implemented for phi.ndim = ",phi.ndim)  
                raise NotImplementedError
        else:
            ortho_loss = torch.tensor(0.0)
        ortho_weight = self.model.config['ortho_weight'] 
        ortho_loss_weighted = ortho_weight*ortho_loss
        
        bound_params = self.model.config['bound_params']
        if bound_params:
            Q, L = self.params
            Kbar = (torch.inverse(L) @ Q)
            theta_c = Kbar.squeeze()[-2]
            theta_phi = Kbar.squeeze()[-1]
            # theta_c between 0.7 and 5
            th_c_low = 0.7
            th_c_high = 5.0
            # theta_phi between tan(15 deg) and tan(60 deg), i.e. 0.268 and 1.732
            th_phi_low = 0.268
            th_phi_high = 1.732
            bounds_loss = torch.tensor(0.0)
            if theta_c > th_c_high:
                bounds_loss += (theta_c - th_c_high)**2
            elif theta_c < th_c_low:
                bounds_loss += (theta_c - th_c_low)**2
            if theta_phi > th_phi_high:
                bounds_loss += (theta_phi - th_phi_high)**2
            elif theta_phi < th_phi_low:
                bounds_loss += (theta_phi - th_phi_low)**2
        else:
            bounds_loss = torch.tensor(0.0)
        bounds_loss_weighted = 10* bounds_loss
        print("")
        print("Other loss = ", -(total_logp.mean() + prior_logp.mean()).detach().numpy(), " Ortho loss weighted=",ortho_loss_weighted.detach().numpy(), "Bounds loss weighted=", bounds_loss_weighted.detach().numpy())

        return -(total_logp.mean() + prior_logp.mean()) + ortho_loss_weighted + bounds_loss_weighted
    
    def sample_K(self, stats, N):
        raise NotImplementedError
        
    def prop_particles(self,xi,u,K):
        num_K_samples = self.model.config['num_mmd_samples']
        u_ext = u.unsqueeze(-2).repeat(1, num_K_samples, 1)
        
        # pass through network
        z = torch.cat([xi,u_ext], dim=-1)
        phi = self.model.encoder(z)
        if self.model.add_nom_features:
            phi_nom = self.model.phi_nom_fn(x,u)
            phi = self.model.augment_phi(phi, phi_nom)

        #grab sig eps
        Sig = torch.sqrt(self.model.SigEps.unsqueeze(0))

        # compute K phi(x) + eps (adding fnom!)
        fnom = self.f_nom(xi,u_ext)
        
        xp = (K @ phi).squeeze(-1) + Sig.unsqueeze(1) @ torch.randn_like(xi).unsqueeze(-1)

        return xp.squeeze(-1) + fnom
        
    def kernel(self,sq_dist):
        #evaluates elementwise squared exponential kernel
        l = self.model.length_scale
        return torch.exp(-sq_dist/(2*l))/((l*2*np.pi)**(self.ob_dim/2))    
        
    def mmd_loss(self,xp,x):
        # not currently used
        
        num_K_samples = self.model.config['num_mmd_samples']
        
        # compute squared distance
        d1 = torch.sum((xp.unsqueeze(-3) - x)**2,dim=-1)
        d2 = torch.sum((x.unsqueeze(-3) - x.unsqueeze(-4))**2,dim=-1)
        
        k1 = self.kernel(d1)
        k2 = self.kernel(d2)
        
        dist = 2.*torch.sum(k1,dim=-2)/num_K_samples - (1./(num_K_samples*(num_K_samples-1)))*torch.sum(k2,dim=[-2,-3])
        return -dist
    
    def log_kde(self,xp,x):
        # uses gaussian kernel
        l = self.model.length_scale
        num_K_samples = self.model.config['num_mmd_samples']
        
        if not self.model.config['multistep_curriculum']:
            dist = (-torch.sum((xp.unsqueeze(-3) - x)**2,dim=-1)**2)
            max_eval = self.model.config['max_evaluation_length']
            lenscale = l*self.to_cuda(torch.linspace(1, max_eval, max_eval).unsqueeze(0).unsqueeze(0))
            d1 = dist/(lenscale) + torch.log(self.ob_dim*lenscale)
        else:
            d1 = (-torch.sum((xp.unsqueeze(-3) - x)**2,dim=-1)**2)/(2*l)  

        return -torch.logsumexp(d1,dim=-2)
    
    def evaluate_sample_multistep(self, sample):
        # TODO(james): add incremental horizon increase for multistep training
        
        data_horizon = self.model.config['data_horizon']
        max_context = self.model.config['max_context']
        
        num_K_samples = self.model.config['num_mmd_samples']
        eval_len = self.model.config['max_evaluation_length']
        
        x = self.to_cuda(sample['x'].float())
        u = self.to_cuda(sample['u'].float())
        xp = self.to_cuda(sample['xp'].float())
        
        z = torch.cat([x,u], dim=-1)
        
        # batch compute features and targets for BLR
        phi = self.model.encoder(z)
        if self.model.add_nom_features:
            phi_nom = self.model.phi_nom_fn(x,u)
            phi = self.model.augment_phi(phi, phi_nom)

        y = xp - self.f_nom(x,u)

        # get prior statistics
        stats = self.model.prior_params()
        stats = [p.unsqueeze(0) for p in stats] # add dim for batch eval
        
        # uniformly sample a conditioning horizon up to max_context
        
        cond_len = max_context
#         cond_len = 0 if max_context==0 else np.random.randint(max_context)
#         start_time = 0 if cond_len==0 else np.random.randint(cond_len)
        
        loss_list = []
    
        for j in range(cond_len):
            phi_ = phi[:,j,...]
            y_ = y[:,j,:]

            # do posterior conditioning, skip loss computation
            stats = self.model.recursive_update(phi_, y_, stats)

            # ----- forward propagation + loss computation

            # sample dynamics models from the posterior         
            K_samples = self.sample_K(stats, num_K_samples)


            # extend particles to (batch size, num particles, xdim)
            # initial state
            x_ = x[:,j,:].unsqueeze(-2).repeat(1,num_K_samples,1)

            horizon = min(data_horizon, j + eval_len)

            if self.model.config['multistep_curriculum']:
                curr_step = self.model.config['curriculum_step']
                horizon = min(horizon, j + int(np.floor(self.total_train_itrs/curr_step)) + 1)

            # propagate particles
            xpred_list = []
            for t in range(j,horizon):
                x_pred = self.prop_particles(x_,u[:,t,:],K_samples)
                xpred_list.append(x_pred)
                x_ = x_pred
                
            xpred_mat = torch.stack(xpred_list, dim=-2)

            # get targets
            xp_t = xp[:,j:horizon,:]        

            # batch eval loss
            loss = self.log_kde(xp_t,xpred_mat)
            loss_list.append(loss.mean())
            
        return torch.stack(loss_list).mean()
            
        
    def train(self, dataloader, num_train_updates, val_dataloader=None, verbose=False):
        """
        Trains the dynamics model on data.
        Inputs: dataloader: torch DataLoader that returns batches of samples that can 
                            be indexed by 'x', 'u', and 'xp'
                num_train_updates: int specifying the number of gradient steps to take
                
                val_dataloader: torch DataLoader, a dataloader used for validation (optional)
                verbose: bool, whether to print training progress.
                
        Progress is logged to a tensorboard summary writer.
        
        Outputs: None. self.model is modified.
        """
        self.reset()
        config = self.model.config
        

        validation_freq = 100
        val_iters = 5

        print_freq = validation_freq if verbose else num_train_updates+1
        
        eval_method = self.evaluate_sample_singlestep
        if config['multistep_training']:
            eval_method = self.evaluate_sample_multistep
        
        data_iter = iter(dataloader)
        
        with trange(num_train_updates, disable=(not verbose or not config['tqdm'])) as pbar:
            for idx in pbar:
                try:
                    sample = next(data_iter)
                except StopIteration:
                    # reset data iter
                    data_iter = iter(dataloader)
                    sample = next(data_iter)

                self.total_train_itrs += 1

                self.optimizer.zero_grad()
                self.model.train()
                total_loss = eval_method(sample)

                # compute validation loss
                if idx % validation_freq == 0 and val_dataloader is not None:
                    total_loss_val = []

                    self.model.eval()
                    for k, val_sample in enumerate(val_dataloader):
                        total_loss_val.append( eval_method(val_sample) )

                        if k == val_iters-1:
                            total_nll_val = torch.stack(total_loss_val).mean().detach().numpy()
                            self.writer.add_scalar('NLL/Val', total_nll_val, self.train_step)
                            break

                # grad update on logp
                total_nll = total_loss
                total_nll.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(),config['grad_clip_value'])
                self.optimizer.step()
                if config['learning_rate_decay']:
                    self.scheduler.step()

                self.model.sigma_scale = max(1., self.model.sigma_scale*config['sigma_eps_annealing'])

                # ---- logging / summaries ------
                self.train_step += 1
                step = self.train_step

                # tensorboard logging
                self.writer.add_scalar('NLL/Train', total_nll.item(), step)
                
                if self.model.config['multistep_curriculum']:
                                curr_step = self.model.config['curriculum_step']
                                horizon = min(self.model.config['data_horizon'], int(np.floor(self.total_train_itrs/curr_step)) + 1)

                                self.writer.add_scalar('Evaluation_horizon', horizon, step)
                                
                Q, L = self.model.prior_params()
#                 K = (Linv @ Q.unsqueeze(-1)).squeeze(-1)
#                 _, Linv_sig, _ = torch.svd(Linv)
#                 _, Q_sig, _ = torch.svd(Q)
#                 _, K_sig, _ = torch.svd(K)

#                 for dim in range(config['x_dim']):
#                     self.writer.add_histogram('log_Linv'+str(dim)+'_sig', torch.log(Linv_sig[dim,:]), step)
#                 self.writer.add_histogram('Q_sig', Q_sig, step)
#                 self.writer.add_histogram('K_sig', K_sig, step)
#                 self.writer.add_histogram('sigma_eps_val', self.model.logSigEps, step)
                if config['learning_rate_decay']:
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], step)
                self.writer.add_scalar('sigma_eps_scale', self.model.sigma_scale, step)
                self.writer.add_scalar('kernel_bandwidth', self.model.length_scale, step)
                
                # tqdm logging
                logdict = {}
                logdict["tr_loss"] = total_nll.cpu().detach().numpy()
                if val_dataloader is not None:
                    logdict["val_loss"] = total_nll_val
                if config['learnable_bandwidth']:
                    logdict["bandwidth"] = self.model.length_scale.item()
                    
                pbar.set_postfix(logdict)

                self.reset()
