# Praise be the LORD And Jesus Christ

# Import libraries
import torch
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __str__(self):
        ...
    @abstractmethod
    def __call__(self, model_parameters: torch.Tensor):
        ...


class Adam(Optimizer):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, **kwds):
        self.__lr = lr
        self.__betas = betas
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__amsgrad = amsgrad
        self.__kwds = kwds
        
    def __str__(self):
        return "Adam"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.Adam(
            model_parameters,
            lr=self.__lr,
            betas=self.__betas,
            eps=self.__eps,
            weight_decay=self.__weight_decay,
            amsgrad=self.__amsgrad,
            **self.__kwds
            )

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False, **kwds):
        self.__lr = lr
        self.__momentum = momentum
        self.__dampening = dampening
        self.__weight_decay = weight_decay
        self.__nesterov = nesterov
        self.__kwds = kwds
        
    def __str__(self):
        return "SGD"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.SGD(
            model_parameters,
            lr=self.__lr,
            momentum=self.__momentum,
            dampening=self.__dampening,
            weight_decay=self.__weight_decay,
            nesterov=self.__nesterov,
            **self.__kwds
            )
        
class RMSprop(Optimizer):
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, **kwds):
        self.__lr = lr
        self.__alpha = alpha
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        self.__centered = centered
        self.__kwds = kwds
        
    def __str__(self):
        return "RMSprop"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.RMSprop(
            model_parameters,
            lr=self.__lr,
            alpha=self.__alpha,
            eps=self.__eps,
            weight_decay=self.__weight_decay,
            momentum=self.__momentum,
            centered=self.__centered,
            **self.__kwds
            )
        
class Adadelta(Optimizer):
    def __init__(self, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, **kwds):
        self.__lr = lr
        self.__rho = rho
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__kwds = kwds
        
    def __str__(self):
        return "Adadelta"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.Adadelta(
            model_parameters,
            lr=self.__lr,
            rho=self.__rho,
            eps=self.__eps,
            weight_decay=self.__weight_decay,
            **self.__kwds
            )
        
class Adagrad(Optimizer):
    def __init__(self, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, **kwds):
        self.__lr = lr
        self.__lr_decay = lr_decay
        self.__weight_decay = weight_decay
        self.__initial_accumulator_value = initial_accumulator_value
        self.__eps = eps
        self.__kwds = kwds
        
    def __str__(self):
        return "Adagrad"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.Adagrad(
            model_parameters,
            lr=self.__lr,
            lr_decay=self.__lr_decay,
            weight_decay=self.__weight_decay,
            initial_accumulator_value=self.__initial_accumulator_value,
            eps=self.__eps,
            **self.__kwds
            )
        
class Adamax(Optimizer):
    def __init__(self, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwds):
        self.__lr = lr
        self.__betas = betas
        self.__eps = eps
        self.__weight_decay = weight_decay
        self.__kwds = kwds
        
    def __str__(self):
        return "Adamax"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.Adamax(
            model_parameters,
            lr=self.__lr,
            betas=self.__betas,
            eps=self.__eps,
            weight_decay=self.__weight_decay,
            **self.__kwds
            )

class ASGD(Optimizer):
    def __init__(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0, **kwds):
        self.__lr = lr
        self.__lambd = lambd
        self.__alpha = alpha
        self.__t0 = t0
        self.__weight_decay = weight_decay
        self.__kwds = kwds
        
    def __str__(self):
        return "ASGD"
    
    def __call__(self, model_parameters: torch.Tensor):
        return torch.optim.ASGD(
            model_parameters,
            lr=self.__lr,
            lambd=self.__lambd,
            alpha=self.__alpha,
            t0=self.__t0,
            weight_decay=self.__weight_decay,
            **self.__kwds
            )
