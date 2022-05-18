# Multi-objective, multi-fidelity (MOMF) Bayesian Optimization (BO) Algorithm

A method for simultaneous multi-objective and multi-fidelity Bayesian optimization based on expected hypervolume improvement. For more details on the Algorithm please see the following paper outlining the basic functioning. 

Faran Irshad & Andreas Döpp, Expected hypervolume improvement for simultaneous multi-objective and multi-fidelity optimization, arXiv:2112.13901 (2021)
https://arxiv.org/abs/2112.13901

This algorithm makes use of BoTorch (https://botorch.org), which is a Bayesian Optimization framework developed on the Pytorch domain. The acquisition function is available now in the recent version of botorch. To run the tutorial notebook you can import the MOMF acquisiton function from the botorch/acquisition/multi_objective/multi_fidelity.py. The test function for MOMF can also be found in botorch/test_functions/multi_objective_multi_fidelity.py while there is a complete tutorial notebook available under botorch/tutorials/Multi_objective_multi_fidelity_BO.ipynb. This tutorial notebook already imports the MOMF and the test functions and demonstrates the working of the MOMF acquisition function
