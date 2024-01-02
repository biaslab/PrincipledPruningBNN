# Principled Pruning of Bayesian Neural Networks through Variational Free Energy Minimization
*By Jim Beckers, Bart van Erp, Ziyue Zhao, Kirill Kondrashov and Bert de Vries*
### Published in the IEEE Open Journal of Signal Processing
---
**Abstract**

Bayesian model reduction provides an efficient approach for comparing the performance of all nested sub-models of a model, without re-evaluating any of these sub-models. Until now, Bayesian model reduction has been applied mainly in the computational neuroscience community on simple models. In this paper, we formulate and apply Bayesian model reduction to perform principled pruning of Bayesian neural networks, based on variational free energy minimization. Direct application of Bayesian model reduction, however, gives rise to approximation errors. Therefore, a novel iterative pruning algorithm is presented to alleviate the problems arising with naive Bayesian model reduction, as supported experimentally on the publicly available UCI datasets for different inference algorithms. This novel parameter pruning scheme solves the shortcomings of current state-of-the-art pruning methods that are used by the signal processing community. The proposed approach has a clear stopping criterion and minimizes the same objective that is used during training. Next to these benefits, our experiments indicate better model performance in comparison to state-of-the-art pruning schemes.

---
This repository contains all experiments of the paper.
