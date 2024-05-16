# Restoring Data Balance for binary classification tasks

Supporting repository for the paper [*Restoring balance: principled under/oversampling of data for optimal classification*](https://arxiv.org/abs/2405.09535).

This repository allows for a numerical and theoretical analysis of the behavior of linear classifiers under binary classification tasks with imbalanced multi-state values (Potts) data. The script to reproduce theoretical curves of standard generalization metrics
assumes different first-order statistics for each class ${\bf M_{\pm} }$ and common second-order statistics ${\bf C}$, while the script for numerical simulations takes as input the two classes of datapoints with labels $-1,1$.

A demo notebook shows how to run both the numerics and the analytics.

### Dependencies
The scripts are entirely written in python 3 and use standard scientific computing packages.

### How to cite this repo:

If you use this repo, please cite this paper:

*Restoring balance: principled under/oversampling of data for optimal classification*, Emanuele Loffredo, Mauro Pastore, Simona Cocco, Remi Monasson [arXiv](https://arxiv.org/abs/2405.09535)

If you have any question or comment, feel free to write [Emanuele](mailto:emanuele.loffredo@phys.ens.fr) or [Mauro](mailto:mauro.pastore@phys.ens.fr).
