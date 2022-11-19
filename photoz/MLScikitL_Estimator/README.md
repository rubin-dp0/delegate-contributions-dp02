# README.md

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- Creation date : November 2022
- Adapted for DP02 from the original DP01 version

## Goal

The goal is to perform simple PhotoZ estimation of DC2 data in three notebooks.
The first notebook extract the data from the TAP tables.
The second notebook apply PhotoZ to 3 standard scikit-learn estimators.
The third notebook adress the question of hyperparameters optimisation



## Notebooks to perform PhotoZ estimation with scikit learn


- **01_MLscikitL_PhotoZSimple_GetData.ipynb** : Extract required DC2 data from tables an write them in a picle file for later use. The purpose is to speed up the interactive demo. 
- **02_MLscikitL_PhotoZSimple_models.ipynb** : Apply three estimator models to PhotoZ 
- **03_MLscikitL_PhotoZSimple_hyperpopt.ipynb** : Optimisation of hyperparameters.
