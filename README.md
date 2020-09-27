# rover-meta-learning

This is code for [Adaptive Meta-Learning for Identification of Rover-Terrain Dynamics](https://arxiv.org/abs/2009.10191), in which we present a meta-learning based approach to adapt probabilistic predictions of rover dynamics and estimates of terrain parameters by augmenting a nominal model affine in parameters with a Bayesian regression algorithm (P-ALPaCA).

P-ALPaCA is an alternate formulation of [ALPaCA](https://github.com/StanfordASL/ALPaCA), which is a framework for online learning that can be imbued with rich, informative priors offline to enable few-shot learning with bayesian uncertainty estimates.

## installation

To use this codebase, first install the requirements by running the following line (ideally within a virtual environment)

```
pip install -r requirements.txt
```

Having done this, run the notebooks in the `demos` directory to train models, predict rover dynamics, estimate terrain parameters, and visualize the effect of orthogonality regularization. 