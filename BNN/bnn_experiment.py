#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Pierce Howell

Most of this model is adapted from 
https://github.com/french-paragon/BayesianMnist.git
"""

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse as args
from bnn_model import RichterPredictorBNN
from dataset import get_data


def saveModels(models, savedir) :
	
	for i, m in enumerate(models) :
		
		saveFileName = os.path.join(savedir, "model{}.pth".format(i))
		
		torch.save(m.state_dict(), os.path.abspath(saveFileName))
	
def loadModels(savedir, in_features) :
	
    models = []

    for f in os.listdir(savedir) :
        
        model = RichterPredictorBNN(in_features=in_features, p_mc_dropout=None)		
        model.load_state_dict(torch.load(os.path.abspath(os.path.join(savedir, f))))
        #model = torch.load(os.path.abspath(os.path.join(savedir, f)))
        models.append(model)
    return(models)


def get_most_likely_class(samples):
    """
    Returns the most likely class after running make multiple passes with a single
    BNN.
    """
    mean_pred_probs = torch.mean(samples, dim=0)
    return(torch.argmax(mean_pred_probs, dim=1))

if __name__ == "__main__":
    parser = args.ArgumentParser(description="Training a BNN for Richter Predictor")

    parser.add_argument('--no-train', action="store_true", help='Load a model instead of training')
    parser.add_argument('--num-networks', type=int, default=1, help="Number of networks to train")
    parser.add_argument('--num-epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=5.0e-3, help="Learning rate of optimizer")
    parser.add_argument('--num-pred-val', type=int, default=10, help="Number of times to run indiviual validation through the BNN")
    parser.add_argument('--save-dir', default=None, help="Directory where the model is saved")
    args = parser.parse_args()
    
    num_networks = args.num_networks
    lr = args.learning_rate
    num_epochs = args.num_epochs
    num_pred_val = args.num_pred_val
    batch_size = args.batch_size


    # TODO: get the training and validation sets
    X_train, y_train, X_val, y_val = get_data()
    N = len(X_train)
    train_data = [ [X_train[i], y_train[i]] for i in range(X_train.shape[0])]
    val_data = [ [X_val[i], y_val[i]] for i in range(X_val.shape[0])]
    in_features = X_train.shape[1] # number of input features
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
    

    
    
    num_batches = len(train_loader)
    digitsBatchLen = len(str(num_batches))

    models = []

    if(args.no_train):
        models = loadModels(args.save_dir, in_features)

    else:
        # train the network
        print("Training...")
        for i in np.arange(num_networks):

            print("Training model {}/{}:".format(i+1, num_networks))
            
            model = RichterPredictorBNN(in_features=in_features, p_mc_dropout=None)
            loss = torch.nn.NLLLoss(reduction='mean')

            optimizer = Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()

            # epochs
            for n in np.arange(num_epochs):

                for batch_id, batch in enumerate(train_loader):
                    
                    X, y = batch

                    # make a prediciton with the model
                    y_pred = model(X, stochastic=True)
                    log_prob = loss(y_pred, y)
                    vi_model_losses = model.evalAllLosses()
                    
                    # negative estimate of ELBO
                    f =  N*log_prob + vi_model_losses

                    optimizer.zero_grad()
                    f.backward()

                    optimizer.step()
                    print("\r", ("\tEpoch {}/{}: Train step {"+(":0{}d".format(digitsBatchLen))+"}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
                                                                                                        n+1, num_epochs,																				batch_id+1,
                                                                                                        num_batches,
                                                                                    torch.exp(-log_prob.detach().cpu()).item(),
                                                                                                        vi_model_losses.detach().cpu().item(),
                                                                                                        f.detach().cpu().item()), end="")
            print("")
            models.append(model)
    
        if(args.save_dir is not None):
            saveModels(models, args.save_dir)

    # validation
    with torch.no_grad():
        

        samples = torch.zeros((num_pred_val, len(val_data), 3))

        # load validation data (all batches)
        X, y = next(iter(val_loader))
        
        # run each data sample through the BNN multiple times
        for i in np.arange(num_pred_val):
            
            # randomly pick one of the trained models
            model_id = np.random.randint(num_networks)
            model = models[model_id]

            samples[i,:,:] = torch.exp(model(X))
    
    withinSampleMean = torch.mean(samples, dim=0)
    samplesMean = torch.mean(samples, dim=(0,1))
    y_pred = get_most_likely_class(samples)
    
    #mse_loss = torch.nn.MSELoss()
    #print("MSE Loss", mse_loss(y_pred.float(), y.float()))
    
    ce_loss = torch.nn.CrossEntropyLoss()
    print("Cross Entropy Loss", ce_loss(withinSampleMean, y))
    
    withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
    acrossSamplesStd = torch.std(withinSampleMean, dim=0)
    
    print("")
    print("Class prediction analysis:")
    print("\tMean class probabilities:")
    print(samplesMean)
    print("\tPrediction standard deviation per sample:")
    print(withinSampleStd)
    print("\tPrediction standard deviation across samples:")
    print(acrossSamplesStd)

    plt.figure("Seen class probabilities")
    plt.bar(np.arange(3), samplesMean.numpy())
    plt.xlabel('digits')
    plt.ylabel('digit prob')	
    plt.ylim([0,1])
    plt.xticks(np.arange(3))
    
    plt.figure("Seen inner and outter sample std")
    plt.bar(np.arange(3)-0.2, withinSampleStd.numpy(), width = 0.4, label="Within sample")
    plt.bar(np.arange(3)+0.2, acrossSamplesStd.numpy(), width = 0.4, label="Across samples")
    plt.legend()
    plt.xlabel('digits')
    plt.ylabel('std digit prob')
    plt.xticks(np.arange(3))

    #plt.show()
