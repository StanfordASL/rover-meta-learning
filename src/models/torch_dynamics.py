from abc import ABC, abstractmethod
from enum import Enum
import torch
import os
from ..core import DynamicsModel
    
class TorchAdaptiveDynamics(DynamicsModel):
    """
    Convenience subclass for pyTorch based Adpative DynamicsModels, implementing
    saving functionality and numpy wrapping
    """
    def to_torch(self, *args):
        """
        for every argument (assumed to be a numpy array), this function
        puts the arguments into a float32 torch.Tensor and pushes it to the same device
        as self.model
        """
        device = next(self.model.parameters()).device
        return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in args]
    
    def save(self, path):
        """
        Saves self.model to path.
        """
        torch.save({
            "config":self.model.config, 
            "state_dict":self.model.state_dict() }, path)
    
    def get_model(self, with_grad=False, with_opt=False, modeltype=DynamicsModel.ModelType.POSTERIOR_PREDICTIVE, **kwargs):
        """
        uses self.get_model to get torch_fn, then wraps it with a numpy wrapper
        takes in torch_fn of the form (mu, sig) = torch_fn(x,u) which operates on torch tensors
        outputs version of the function that works on numpy inputs and gives numpy outputs
        if requested, will augment the output with dmu_dx and dmu_du
        """
        torch_fn = self.get_model_torch(modeltype=modeltype, with_opt=with_opt, **kwargs)

        def f(x_np, u_np):
            batch_shape = x_np.shape[:-1]
            x_dim = x_np.shape[-1]
            u_dim = u_np.shape[-1]

            x_np = x_np.reshape([-1,x_dim])
            u_np = u_np.reshape([-1,u_dim])

            x, u = self.to_torch(x_np, u_np)

            if with_grad:
                x.requires_grad = True
                u.requires_grad = True

            if not with_opt:
                mu, sig = torch_fn(x,u)
            else:
                mu, sig, Lp = torch_fn(x,u)
                Lp_np = Lp.squeeze(1).detach().cpu().numpy()
            mu_np = mu.detach().cpu().numpy().reshape(batch_shape + (x_dim,))
            sig_np = sig.detach().cpu().numpy().reshape(batch_shape + (x_dim, x_dim))

                
            if with_grad:
                dmu_dx = torch.cat([ torch.autograd.grad(mu[..., i].sum(), x,
                                    retain_graph=True)[0][..., None, :]
                                        for i in range(x_dim) ], -2)
                dmu_dx_np = dmu_dx.detach().cpu().numpy().reshape(batch_shape + (x_dim, x_dim))

                dmu_du = torch.cat([ torch.autograd.grad(mu[..., i].sum(), u,
                                    retain_graph=(i + 1 < x_dim))[0][..., None, :]
                                        for i in range(x_dim) ], -2)

                dmu_du_np = dmu_du.detach().cpu().numpy().reshape(batch_shape + (x_dim, u_dim))

                if not with_opt:
                    return mu_np, sig_np, dmu_dx_np, dmu_du_np
                else:
                    return mu_np, sig_np, dmu_dx_np, dmu_du_np, Lp_np

            if not with_opt:
                return mu_np, sig_np
            else:
                return mu_np, sig_np, Lp_np

        return f
    
    def incorporate_transition(self, x, u, xp):
        device = next(self.model.parameters()).device
        x, u, xp = self.to_torch(x,u,xp)

        return self.incorporate_transition_torch_(x,u,xp)
    
    @abstractmethod
    def get_model_torch(self, with_grad=False, modeltype=DynamicsModel.ModelType.POSTERIOR_PREDICTIVE, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def incorporate_transition_torch_(self, x, u, xp):
        """
        updates internal parameters using inputs
        x, u, xp: torch tensors
        
        returns updated parameters
        """
        raise NotImplementedError
    