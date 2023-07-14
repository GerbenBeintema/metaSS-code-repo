# metaSS-code-repo

This repository contains the code used to create the numerical examples shown in the pre-print paper: 

* Gerben I Beintema, Maarten Schoukens and Roland Tóth "Meta-State-Space Learning: An Identification Approach for Stochastic Dynamical Systems", https://arxiv.org/abs/2307.06675

## Requirements

`torch, numpy, matplotlib, tqdm, scipy` or see `conda-environment.yml` for the entire conda environment list.

## Repository Structure

the meta-state-space implementation is divided into four files

* `fancy_distributions.py` implements Gaussian mixtures 
* `static_distribution_modeling.py` implements Mixture Density Network
* `fitting_tools.py` a utility file with the fitting function
* `meta_SS_models.py` implements the meta-state-space model itself.

This implementation of the meta-state-space model is than applied on a numerical example in 

* `alpha-system.ipynb` with generates the data, fits the meta-state-space model and analysis the result.

## Alternative implementation

There is an alternative implementation of the meta-state-space model available with more features in the metaSI toolbox. 

* toolbox: https://github.com/GerbenBeintema/metaSI

## Questions

Feel free to contact: g.i.beintema@tue.nl for any questions you might have. 
