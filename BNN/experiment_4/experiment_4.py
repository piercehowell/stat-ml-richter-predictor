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
from dataset import get_data, CustomDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import f1_score
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR


def saveModels(models) :
	
    for i, m in enumerate(models):
        
        saveFileName = os.path.join("models/model_{}.pth".format(i))
		
        torch.save(m.state_dict(), "models/model_{}.pth".format(i))
	
def loadModels(in_features) :
	
    models = []
    for f in os.listdir("models") :
        
        model = RichterPredictorBNN(in_features=in_features, p_mc_dropout=None)		
        model.load_state_dict(torch.load(os.path.abspath(os.path.join("models", f))))
        models.append(model)
    
    return(models)


def get_most_likely_class(samples):
    """
    Returns the most likely class after running make multiple passes with a single
    BNN.
    """
    mean_pred_probs = torch.mean(samples, dim=0)

    y_pred_probs, y_pred = torch.max(mean_pred_probs, dim=1)
    return y_pred, y_pred_probs, mean_pred_probs

if __name__ == "__main__":
    parser = args.ArgumentParser(description="Training a BNN for Richter Predictor")

    parser.add_argument('--no-train', action="store_true", help='Load a model instead of training. And runs on the testing data')
    parser.add_argument('--num-networks', type=int, default=1, help="Number of networks to train")
    parser.add_argument('--num-epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=5.0e-3, help="Learning rate of optimizer")
    parser.add_argument('--num-pred-val', type=int, default=10, help="Number of times to run indiviual validation through the BNN")
    parser.add_argument('--save-model', action="store_true", help="Directory where the model is saved")
    args = parser.parse_args()
    
    run_test = True
    # ----------------------- SAVING ARGUMENTS-------------------------
    num_networks = args.num_networks
    lr = args.learning_rate
    num_epochs = args.num_epochs
    num_pred_val = args.num_pred_val
    batch_size = args.batch_size
    no_train = args.no_train
    # -------------------------------------------------------------------


    # ---------------------- GET THE DATA -------------------------------
    # TODO: get the training and validation sets
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    N = len(X_train)
    train_data = [ [X_train[i], y_train[i]] for i in range(X_train.shape[0])]
    val_data = [ [X_val[i], y_val[i]] for i in range(X_val.shape[0])]
    in_features = X_train.shape[1] # number of input features
    print(in_features)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
    test_data = CustomDataset(X_test, y_test)
    
    
    num_batches = len(train_loader)

    
    # -----------------------------------------------------------------------

    # All models trained are appended to this model list
    models = []
    
    # define loss functions and metrics
    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/BNN_{}'.format(timestamp))

    if(args.no_train):
        pass

    # ------------------- TRAINING + VALIDATION --------------------------------------------------------------#
    else:
        # train the network
        print("Training...")
        for i in np.arange(num_networks):

            print("Training model {}/{}:".format(i+1, num_networks))
            
            model = RichterPredictorBNN(in_features=in_features, p_mc_dropout=None)
            
            loss = torch.nn.NLLLoss(reduction='mean')

            optimizer = Adam(model.parameters(), lr=lr)
            scheduler = ExponentialLR(optimizer, gamma=0.9)
            optimizer.zero_grad()

            # epochs
            for nepoch in np.arange(num_epochs):
                
                #---------------------------- TRAINING STEP ----------------------------------------------------#
                running_loss = 0.0
                for batch_id, batch_data in enumerate(tqdm(train_loader, 0)):
                    
                    X, y = batch_data

                    # make a prediciton with the model
                    y_pred = model(X, stochastic=True)
                    log_prob = loss(y_pred, y)
                    vi_model_losses = model.evalAllLosses()
                    
                    # negative estimate of ELBO
                    f =  N*log_prob + vi_model_losses

                    optimizer.zero_grad()
                    f.backward()

                    optimizer.step()

                    running_loss += f.item()
                    if batch_id % 1000 == 999:
                        last_loss = running_loss / 1000
                        writer.add_scalar("ELBO Loss/train", last_loss, nepoch * len(train_loader) + batch_id + 1)
                        running_loss = 0.0

                if(nepoch % 10 == 0):
                    if(args.save_model is True):
                        saveModels(models)

                scheduler.step()
                #----------------------------------------------------------------------------------------------#
                #-------------------- VALIDATION STEP--------------------------------------------------#
                model.eval()
                val_mse_loss = 0.0
                X, y = next(iter(val_loader))
                samples = torch.zeros((num_pred_val, len(val_data), 3))

                with torch.no_grad():
                    # run each data sample through the BNN multiple times
                    for k in np.arange(num_pred_val):
                        
                        samples[k, :, :] = torch.exp(model(X))


                y_pred, y_pred_probs, mean_y_pred_probs = get_most_likely_class(samples)
                val_mse_loss = mse_loss(y_pred.float(), y.float())
                val_ce_loss = ce_loss(mean_y_pred_probs, y)
                val_f1 = f1_score(y, y_pred, average='micro')
                writer.add_scalars("Validation MSE Loss", {'Validation': val_mse_loss}, nepoch)
                writer.add_scalars("Validation Cross Entropy Loss", {'Validation' : val_ce_loss}, nepoch)
                writer.add_scalars("Validation Micro F1 Score", {"Validation" : val_f1}, nepoch)
                writer.flush()

                print("** Validation Results for Network {} Epoch {} **".format(i, nepoch))
                print("-------------------------------------------------------------")
                print("(Validation) Mean Square Error Loss: {}".format(val_mse_loss))
                print("(Validation) Cross_Entropy Loss: {}".format(val_ce_loss))
                print("(Validation) Micro F1 Score: {}".format(val_f1))
                print("")
                #---------------------------------------------------------------------------------------#

            models.append(model)
            print("** Final Results for Network {} **".format(i))
            print("-------------------------------------------------------------")
            print("(Validation) Mean Square Error Loss: {}".format(val_mse_loss))
            print("(Validation) Cross_Entropy Loss: {}".format(val_ce_loss))
            print("(Validation) Micro F1 Score: {}".format(val_f1))
            print("")
            

            # how about calibration curve
            from sklearn.calibration import calibration_curve

            # do it individually for each class
            y1_true = np.copy(y); y1_true[y == 0] = 1; y1_true[y != 0] = 0
            y1_pred = np.copy(mean_y_pred_probs[:, 0])
            prob_y1_true, prob_y1_pred = calibration_curve(y1_true, y1_pred, n_bins=30) 
            plt.figure("Calibration Curve for Damage Grade 1")
            plt.plot(prob_y1_pred, prob_y1_true)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel("Predicted Probability")
            plt.ylabel("Empirical Frequency")
            plt.savefig('figures/exp4_cal_curve_1.png')

            y2_true = np.copy(y); y2_true[y == 1] = 1; y2_true[y != 1] = 0
            y2_pred = np.copy(mean_y_pred_probs[:, 1])
            prob_y2_true, prob_y2_pred = calibration_curve(y2_true, y2_pred, n_bins=30) 
            plt.figure("Calibration Curve for Damage Grade 2")
            plt.plot(prob_y2_pred, prob_y2_true)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel("Predicted Probability")
            plt.ylabel("Empirical Frequency")
            plt.savefig('figures/exp4_cal_curve_2.png')

            y3_true = np.copy(y); y3_true[y == 2] = 1; y3_true[y != 2] = 0
            y3_pred = np.copy(mean_y_pred_probs[:, 2])
            prob_y3_true, prob_y3_pred = calibration_curve(y3_true, y3_pred, n_bins=30) 
            plt.figure("Calibration Curve for Damage Grade 3")
            plt.plot(prob_y3_pred, prob_y3_true)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel("Predicted Probability")
            plt.ylabel("Empirical Frequency")
            plt.savefig('figures/exp4_cal_curve_3.png')

            
    
        if(args.save_model is True):
            saveModels(models)


    # # -------------------------------------- TEST ------------------------------------------------- #
    if(no_train or run_test):
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

        # load in the models
        models = loadModels(in_features)
        model = models[0]
        samples = torch.zeros((num_pred_val, len(test_data), 3))
        #test_loader = torch.utils.data.DataLoader(X_test, batch_size=len(X_test))
        with torch.no_grad():
            X, y = next(iter(test_loader))
            # run each data sample through the BNN multiple times
            for k in np.arange(num_pred_val):
                samples[k, :, :] = torch.exp(model(X))
                
        y_pred, y_pred_probs, mean_y_pred_probs = get_most_likely_class(samples)
        
        test_mse_loss = mse_loss(y_pred.float(), y.float())
        test_ce_loss = ce_loss(mean_y_pred_probs, y)
        test_f1 = f1_score(y, y_pred, average='micro')

        print("** Test Results **")
        print("-------------------------------------------------------------")
        print("(Test) Mean Square Error Loss: {}".format(test_mse_loss))
        print("(Test) Cross_Entropy Loss: {}".format(test_ce_loss))
        print("(Test) Micro F1 Score: {}".format(test_f1))
        print("")

        # do it individually for each class
        y1_true = np.copy(y); y1_true[y == 0] = 1; y1_true[y != 0] = 0
        y1_pred = np.copy(mean_y_pred_probs[:, 0])
        prob_y1_true, prob_y1_pred = calibration_curve(y1_true, y1_pred, n_bins=30) 
        plt.figure("(Test) Calibration Curve for Damage Grade 1")
        plt.title("(Test) Calibration Curve for Damage Grade 1")
        plt.plot(prob_y1_pred, prob_y1_true)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        plt.savefig('figures/exp4_test_cal_curve_1.png')

        y2_true = np.copy(y); y2_true[y == 1] = 1; y2_true[y != 1] = 0
        y2_pred = np.copy(mean_y_pred_probs[:, 1])
        prob_y2_true, prob_y2_pred = calibration_curve(y2_true, y2_pred, n_bins=30) 
        plt.figure("(Test) Calibration Curve for Damage Grade 2")
        plt.title("(Test) Calibration Curve for Damage Grade 2")
        plt.plot(prob_y2_pred, prob_y2_true)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        plt.savefig('figures/exp4_test_cal_curve_2.png')

        y3_true = np.copy(y); y3_true[y == 2] = 1; y3_true[y != 2] = 0
        y3_pred = np.copy(mean_y_pred_probs[:, 2])
        prob_y3_true, prob_y3_pred = calibration_curve(y3_true, y3_pred, n_bins=30) 
        plt.figure("(Test) Calibration Curve for Damage Grade 3")
        plt.title("(Test) Calibration Curve for Damage Grade 3")
        plt.plot(prob_y3_pred, prob_y3_true)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        plt.savefig('figures/exp4_test_cal_curve_3.png')

    
    plt.show()