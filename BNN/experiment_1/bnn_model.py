#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: David Pierce Walker-Howell

Most of this model is adapted from 
https://github.com/french-paragon/BayesianMnist.git
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

class VIModule(nn.Module) :
	"""
	A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
	"""
	
	def __init__(self, *args, **kwargs) :
		super().__init__(*args, **kwargs)
		
		self._internalLosses = []
		self.lossScaleFactor = 1
		
	def addLoss(self, func) :
		self._internalLosses.append(func)
		
	def evalLosses(self) :
		t_loss = 0
		
		for l in self._internalLosses :
			t_loss = t_loss + l(self)
			
		return t_loss
	
	def evalAllLosses(self) :
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		return t_loss


class MeanFieldGaussianFeedForward(VIModule) :
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""
	
	def __init__(self, 
			  in_features, 
			  out_features, 
			  bias = True,  
			  groups=1, 
			  weightPriorMean = 0, 
			  weightPriorSigma = 1.,
			  biasPriorMean = 0, 
			  biasPriorSigma = 1.,
			  initMeanZero = False,
			  initBiasMeanZero = False,
			  initPriorSigmaScale = 0.01) :
		
		
		super(MeanFieldGaussianFeedForward, self).__init__()
		
		self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}
		
		self.in_features = in_features
		self.out_features = out_features
		self.has_bias = bias
		
		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5))
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))
			
		self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)), 
								   torch.ones(out_features, int(in_features/groups)))
		
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())
		
		if self.has_bias :
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))
			
			self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))
			
			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
			self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())
			
			
	def sampleTransform(self, stochastic=True) :
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
		
		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)
		
	def getSampledWeights(self) :
		return self.samples['weights']
	
	def getSampledBias(self) :
		return self.samples['bias']
	
	def forward(self, x, stochastic=True) :
		
		self.sampleTransform(stochastic=stochastic)
		
		return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)
	
class RichterPredictorBNN(VIModule):
    """
    Bayesian Neural Network for classification on the Richter Predictor Dataset
    to classify damage grades
    """
    def __init__(self,
                in_features,
                linear_w_prior_sigma = 1.,
                linear_bias_prior_sigma=5.,
                p_mc_dropout=0.5):
        
        super().__init__()
        self.p_mc_dropout = p_mc_dropout

        self.in_features = in_features
        self.linear2_in_features = 50
        self.linear3_in_features = 50
        self.linear4_in_features = 50
        
        # define a set of linear layers
        self.linear1 = MeanFieldGaussianFeedForward(self.in_features, self.linear2_in_features,
                                                    weightPriorSigma=linear_w_prior_sigma,
                                                    biasPriorSigma=linear_bias_prior_sigma,
                                                    initPriorSigmaScale=1e-4)
        self.linear2 = MeanFieldGaussianFeedForward(self.linear2_in_features, self.linear3_in_features,
                                                    weightPriorSigma=linear_w_prior_sigma,
                                                    biasPriorSigma=linear_bias_prior_sigma,
                                                    initPriorSigmaScale=1e-4)

        self.linear3 = MeanFieldGaussianFeedForward(self.linear3_in_features, self.linear4_in_features,
                                                    weightPriorSigma=linear_w_prior_sigma,
                                                    biasPriorSigma=linear_bias_prior_sigma,
                                                    initPriorSigmaScale=1e-4)
        
        self.linear4 = MeanFieldGaussianFeedForward(self.linear4_in_features, 3,
                                                    weightPriorSigma=linear_w_prior_sigma,
                                                    biasPriorSigma=linear_bias_prior_sigma,
                                                    initPriorSigmaScale=1e-4)
        
    def forward(self, x, stochastic=True):
        """
        Define the forward pass of the BNN
        """
        
        x = nn.functional.relu(self.linear1(x, stochastic=stochastic))
        x = nn.functional.relu(self.linear2(x, stochastic=stochastic))
        x = nn.functional.relu(self.linear3(x, stochastic=stochastic))
        x = nn.functional.relu(self.linear4(x, stochastic=stochastic))

        return nn.functional.log_softmax(x, dim=-1)
