# KSI-25: Probabilistic Estimation of Photometric Redshifts

This directory is for code and notebooks written in relation to the Kickstarte project KSI-25 "Probabilistic Estimation of Photometric Redshifts". The directory contains source code and tutorials for the three models implemented during the project, trained on SDSS-DR17 + ALLWISE and DP0.2, respectively. The three models used in this project are:

1. Infinite Gaussian Mixture Model with a Mixture Density Network Regressor
2. Variational Autoencoder with a Mixture Density Network
3. Semi-Supervised Regression Variational Autoencoder with a Mixture Density Network Regressor

The first two models are composite models, with each component trained separately, while the third model trains classification and regression at the same time.

| File / Sub-Directory | Description | Author |
|---|---|---|
| tutorial.ipynb | An extremely simplistic demo that uses a single set of galaxies as both test and training set. This notebook is only useful for estimating rudimentary photo-z for a small number of galaxies. It has a "Future Work" section to describe the next steps. | Jacob O. Hjortlund |
| models.py | An extremely simplistic demo that uses a single set of galaxies as both test and training set. This notebook is only useful for estimating rudimentary photo-z for a small number of galaxies. It has a "Future Work" section to describe the next steps. | Jacob O. Hjortlund |
| utils.py | An extremely simplistic demo that uses a single set of galaxies as both test and training set. This notebook is only useful for estimating rudimentary photo-z for a small number of galaxies. It has a "Future Work" section to describe the next steps. | Jacob O. Hjortlund |
| Models | An extremely simplistic demo that uses a single set of galaxies as both test and training set. This notebook is only useful for estimating rudimentary photo-z for a small number of galaxies. It has a "Future Work" section to describe the next steps. | Jacob O. Hjortlund |
| Data | An extremely simplistic demo that uses a single set of galaxies as both test and training set. This notebook is only useful for estimating rudimentary photo-z for a small number of galaxies. It has a "Future Work" section to describe the next steps. | Jacob O. Hjortlund |


# note

Full datasets used for hyperparameter optimization and training, along with the corresponding code, will be made available in the future.