# Multi-objective, multi-fidelity (MOMF) Bayesian Optimization (BO) Algorithm

A method for simultaneous multi-objective and multi-fidelity Bayesian optimization based on expected hypervolume improvement. For more details on the Algorithm please see the following paper outlining the basic functioning. 

Faran Irshad & Andreas Döpp, Expected hypervolume improvement for simultaneous multi-objective and multi-fidelity optimization, arXiv:2112.13901 (2021)
https://arxiv.org/abs/2112.13901

This algorithm makes use of BoTorch (https://botorch.org), which is a Bayesian Optimization framework developed on the Pytorch domain. The acquisition function is currently under review in a pull-request in the BoTorch framework. The same code is also made available here. To run the tutorial notebook you can import the MOMF acquisiton function from the momf.py file attached. This also contains the test_functions that the tutorial notebook uses. 
