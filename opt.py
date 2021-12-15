#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:15:32 2021

@author: f.irshad
"""
import torch
import numpy as np
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.cost_aware import GenericCostAwareUtility

from botorch.acquisition.acquisition import AcquisitionFunction


# Setting up a global variable tkwargs which has the information of which CPU or GPU 
# to initialize the tensors on. 
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    tkwargs = {
            "dtype": torch.double,
            "device": torch.cuda.current_device(),}
else:
    tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cpu"),}

# Function to reset the global variable value to which ever GPU you want to use.
def set_GPU(gpu_number):
    
    global tkwargs
    print('Current GPU ID',tkwargs['device'])
    torch.cuda.set_device(gpu_number)
    tkwargs = {
            "dtype": torch.double,
            "device": torch.cuda.current_device()}
    print('New GPU ID',tkwargs['device'])
# This is the Matern Kernel Model that models all the 4 (3 inputs+1 Fidelity) 
# with 2 outputs.
def initialize_model(train_x, train_obj):
    
    model = SingleTaskGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=train_obj.shape[-1])    
    )
    #model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# This is the wrapper function for calling the synthetic function for MO 
# Appends a 1 (Highest Fidelity) to the input X.
def problem_MO(x,dim_y,gpu_number):
    X=x.cpu()
    X=X.numpy()
    h_fid=np.ones((X.shape[0],1))
    X=np.hstack((X,h_fid))
    print('Sim Input',X)
    y=np.zeros((X.shape[0],dim_y))
    for num,i in enumerate(X): 
        y[num,:]=func(i)
    y_out=torch.from_numpy(y).to(**tkwargs)
    print('Sim Output MO',y_out)
    
    return y_out


# The wrapper for the 1-step MOMF optimization. This is for the single-step where
# a fidelity objective is also appended to the output.
def problem(x,dim_y,gpu_number):
    X=x.cpu()
    X=X.numpy()
    print('Sim Input',X)
    y=np.zeros((X.shape[0],dim_y-1))
    fid=0.7*X[:,-1]
    for num,i in enumerate(X):
        y[num,:]=func(i)
    y_out=torch.from_numpy(y).to(**tkwargs)
    fid_out=torch.from_numpy(fid).to(**tkwargs)
    y_MO=torch.cat([y_out,torch.reshape(fid_out,(-1,1))],1)
    print('Sim Output',y_MO)
    
    return y_MO

# The wrapper for the 2-step MOMF optimization. 
def problem_2step(x,dim_y,gpu_number):
    X=x.cpu()
    X=X.numpy()
    print('Sim Input',X)
    y=np.zeros((X.shape[0],dim_y))
    for num,i in enumerate(X):
        y[num,:]=func(i)
    y_MO=torch.from_numpy(y).to(**tkwargs)
    print('Sim Output',y_MO)
    return y_MO

# Generating Initial data for the MO.
def generate_initial_data_MO(dim_x,dim_y,points,gpu_number):
    train_x=np.random.rand(points,dim_x)
    train_x=torch.from_numpy(train_x)
    train_x=train_x.to(**tkwargs)
    train_obj=problem_MO(train_x,dim_y,gpu_number)
    return train_x, train_obj

def generate_initial_data(dim_x,dim_y,points,gpu_number):
    # generate training data with random distribution
    train_x=np.random.rand(points,dim_x)
    # replace last dimension with custom distribution that takes cost into account
    x1 = np.linspace(1,2,101);popt=[0.01,4.8,0]
    p1 = (1/cfunc(x1,*popt, return_torch=False)); p1 = p1/np.sum(p1)
    train_x[:,-1] = np.random.choice(x1-1, size = points, p=p1)
    train_x=torch.from_numpy(train_x)
    train_x=train_x.to(**tkwargs)
    train_obj=problem(train_x,dim_y,gpu_number)
    return train_x, train_obj
	
def generate_initial_data_2step(dim_x,dim_y,points,gpu_number):
    # generate training data with random distribution
    train_x=np.random.rand(points,dim_x)
    # replace last dimension with custom distribution that takes cost into account
    x1 = np.linspace(1,2,101);popt=[0.01,4.8,0]
    p1 = (1/cfunc(x1,*popt, return_torch=False)); p1 = p1/np.sum(p1)
    train_x[:,-1] = np.random.choice(x1-1, size = points, p=p1)
    train_x=torch.from_numpy(train_x)
    train_x=train_x.to(**tkwargs)
    train_obj=problem_2step(train_x,dim_y,gpu_number)
    return train_x, train_obj
	
# This contains the functions that we want to optimize.
def func(X):
    x1=1-2*(X[0]-0.6)**2
    x2=X[1]
    x3=1-3*(X[2]-0.5)**2
    x4=1-1*(X[3]-0.8)**2
    s=X[4]
    
    term1 = (2/3) * np.exp(x1+x2);
    term2 = -x4 * np.sin(x3)*(0.9+0.1*s);
    term3 = x3;
    y1 = ((term1 + term2 + term3-0.1*(1-s)))*(0.9+0.1*(s));
    y1=np.reshape(((5-y1)/4)-0.7,(-1,1))
    
    term1a = (x1+0.001*(1-s)) / 2;
    term1b = np.sqrt(1 + ((x2)+x3**2)*((x4))/((x1)**2+0.0001));
    term1 = term1a * term1b;

    term2a = (x1) + 3*(x4);
    term2b = np.exp(1 + np.sin(x3));
    term2 = term2a * term2b;

    y2 = (term1 + term2-0.1*(1-s))*(0.9+0.1*(s));
    y2=np.reshape((y2/22)-0.8,(-1,1))
    y=np.hstack((y1,y2))
    
    return y
# This is the optimizer function for normal Multi-objective example.
def optimize_qehvi_and_get_observation_MO(model, train_obj, sampler,dim_y,ref_point,standard_bounds,BATCH_SIZE_MO,gpu_number):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(num_outcomes=dim_y, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE_MO,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True
        
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=standard_bounds)
    new_obj = problem_MO(new_x,dim_y,gpu_number)
    return new_x,new_obj

# Writing a class for the penalization. The botorch wrapper is written for
# Penalized_acq=raw_acq - regularization parameter*penalizer
# This one is written for the division Penalized_acq=raw_acq/reg_parameter/penalizer
class PenalizedAcquisitionFunction1(AcquisitionFunction):

    def __init__(
        self,
        raw_acqf: AcquisitionFunction,
        penalty_func: torch.nn.Module,
        regularization_parameter: float,
    ) -> None:
        r""".

        Args:
            raw_acqf: The raw acquisition function that is going to be regularized.
            penalty_func: The regularization function.
            regularization_parameter: Regularization parameter used in optimization.
        """
        super().__init__(model=raw_acqf.model)
        self.raw_acqf = raw_acqf
        self.penalty_func = penalty_func
        self.regularization_parameter = regularization_parameter

    def forward(self, X):
        raw_value = self.raw_acqf(X=X)
        penalty_term = self.penalty_func(X)
        return raw_value/(self.regularization_parameter * penalty_term)

# This is the penalizer wrapper that uses a cost function defined in the main script/jupyter notebook
class Mypenalizer(torch.nn.Module):
    
     def __init__(self):
        super().__init__()
        
     def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        param=[0.01,4.81,0]
        cost = torch.empty(X.shape[0],  **tkwargs)
        for num,i in enumerate(X):
            if i[...,-1]<0 or i[...,-1]>1:
                print('Error cost out of bounds')
                cost[num]= torch.tensor([3])
            else :
                cost[num]= cfunc(1+i[...,-1],*param)
        return cost
        
# This is the cost function that takes in the fidelity input and returns the cost C(s) 
def cfunc(x,A,B,c, return_torch=True):
    if return_torch:
        x=x.detach().cpu()
        x=x.numpy()
    val=c+A*np.exp(B*x)
    if return_torch:
        val=torch.tensor(val)
        val=val.to(**tkwargs)
    return val



# 1-step MOMF optimize and get observation function
# Optimizing the penalized qehvi. The penalizer is the cost of each simulation
# depending on Fidelity parameter. 
def optimize_qehvi_and_get_observation(model, train_obj, sampler,dim_y,ref_point,standard_bounds,BATCH_SIZE_MO,gpu_number):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(ref_point=torch.tensor([0,0,0]), Y=train_obj)
    # Raw unpenalized function
    raw_acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    )
    # Penalizer defined above
    penalty=Mypenalizer()  
    acq_func=PenalizedAcquisitionFunction1(raw_acq_func,
             penalty,regularization_parameter=1) # This is the penalized Acq_func
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE_MO,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=standard_bounds)
    new_obj = problem(new_x,dim_y,gpu_number)
	
    return new_x,new_obj


# This is similar to above function but works with earlier versions of botorch where 
# the NondominatedPartioning require the num_outcomes as input instead of ref_point.
def optimize_qehvi_and_get_observation_old(model, train_obj, sampler,dim_y,ref_point,standard_bounds,BATCH_SIZE_MO,gpu_number):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(num_outcomes=dim_y, Y=train_obj)
    # Raw unpenalized function
    raw_acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    )
	
    penalty=Mypenalizer()  
    acq_func=PenalizedAcquisitionFunction1(raw_acq_func,
             penalty,regularization_parameter=1) # This is the penalized Acq_func
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE_MO,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=standard_bounds)
    new_obj = problem(new_x,dim_y,gpu_number)
	
    return new_x,new_obj
    

# 2-step MOMF optimize and get observation function
# Optimizing a fixed feature EHVI where the fidelity is fixed to 1. 
def optimize_qehvi_and_get_observation_2step(model, train_obj, sampler,dim_y,ref_point,standard_bounds,BATCH_SIZE_MO,gpu_number):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(num_outcomes=dim_y, Y=train_obj)
    raw_acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    )
    num=standard_bounds.shape[1]
    fix_acqf = FixedFeatureAcquisitionFunction(
        acq_function=raw_acq_func,
        d=num,
        columns=[num-1],
        values=[1],
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=fix_acqf,
        bounds=standard_bounds[:,:-1],
        q=BATCH_SIZE_MO,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=standard_bounds[:,:-1])
    return new_x
param=[0.01,4.81,0]

def set_param(x):
    global param
    param=x
# This is a similar cost function for the MFMES acquisition function optimization
def mycost(X,deltas):
    cost = torch.empty(X.shape[0],  **tkwargs)
    for num,i in enumerate(X[:,0,:]):
        if i[...,-1]<0 or i[...,-1]>1:
            print('Error cost out of bounds')
            cost[num]= torch.tensor([3])
        else:
            cost[num]= cfunc(1+i[...,-1],*param)          
    return deltas/cost

# This function optimizes the fidelity only while keeping the input fixed to 
# what is suggested by the EHVI.
def optimize_qMFMES_and_get_observation(model,x,dim_y,candidate_set,bounds,BATCH_SIZE_MF,gpu_number):
    """Optimizes qMFMES and returns a new candidate, observation, and cost."""
    cost_aware_utility = GenericCostAwareUtility(mycost)
    raw_qMFMES_acqf=qMultiFidelityMaxValueEntropy(model=model, candidate_set=candidate_set,cost_aware_utility=cost_aware_utility)
    print('Suggested X',x)
    num=x.shape[1]
    fix_acqf = FixedFeatureAcquisitionFunction(
        acq_function=raw_qMFMES_acqf,
        d=num+1,
        columns=list(range(x.shape[1])),
        values=x,
    )
    
    candidates, acq_value = optimize_acqf(
        acq_function=fix_acqf,
        bounds=bounds[:,:-num],
        q=BATCH_SIZE_MF,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    new_x = candidates.detach()
    new_x=torch.cat([x,new_x],1)
    new_obj= problem_2step(new_x,dim_y,gpu_number)
    sum_obj=torch.sum(new_obj,axis=1)
    sum_obj=sum_obj.reshape(-1,1)
    

    return new_x, sum_obj,new_obj


