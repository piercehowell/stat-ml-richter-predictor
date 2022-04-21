#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Pierce Howell
"""
import sched
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse as args
from bnn_model import RichterPredictorBNN
from dataset import get_data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import f1_score
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.calibration import calibration_curve


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
    
    # ----------------------- SAVING ARGUMENTS-------------------------
    num_networks = args.num_networks
    lr = args.learning_rate
    num_epochs = args.num_epochs
    num_pred_val = args.num_pred_val
    batch_size = args.batch_size
    no_train = args.no_train
    # -------------------------------------------------------------------

    # ------------------------------- DATA ------------------#
    class Data:
        """
        This class organizes the data for BNN active learning
        """
        def __init__(self):

            self.train_data = ...
            self.val_data = ...
            self.test_data = ...
            self.seed_data = ...

    my_data = Data()
    # ---------------------------------------------------------

    # define loss functions and metrics
    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/BNN_{}'.format(timestamp))

    # train the network
    model = RichterPredictorBNN(in_features=in_features, p_mc_dropout=None)
    loss = torch.nn.NLLLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    optimizer.zero_grad()

    # ---------------------- CALIBRATION CURVE FUNCTION ---------------
    def plot_calibration_curves(samples):

        y_pred, y_pred_probs, mean_y_pred_probs = get_most_likely_class(samples)

        # do it individually for each class
        y1_true = np.copy(y); y1_true[y == 0] = 1; y1_true[y != 0] = 0
        y1_pred = np.copy(mean_y_pred_probs[:, 0])
        prob_y1_true, prob_y1_pred = calibration_curve(y1_true, y1_pred, n_bins=30) 
        plt.figure("Calibration Curve for Damage Grade 1")
        plt.plot(prob_y1_pred, prob_y1_true)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        #plt.savefig('figures/exp4_cal_curve_1.png')

        y2_true = np.copy(y); y2_true[y == 1] = 1; y2_true[y != 1] = 0
        y2_pred = np.copy(mean_y_pred_probs[:, 1])
        prob_y2_true, prob_y2_pred = calibration_curve(y2_true, y2_pred, n_bins=30) 
        plt.figure("Calibration Curve for Damage Grade 2")
        plt.plot(prob_y2_pred, prob_y2_true)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        #plt.savefig('figures/exp4_cal_curve_2.png')

        y3_true = np.copy(y); y3_true[y == 2] = 1; y3_true[y != 2] = 0
        y3_pred = np.copy(mean_y_pred_probs[:, 2])
        prob_y3_true, prob_y3_pred = calibration_curve(y3_true, y3_pred, n_bins=30) 
        plt.figure("Calibration Curve for Damage Grade 3")
        plt.plot(prob_y3_pred, prob_y3_true)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        #plt.savefig('figures/exp4_cal_curve_3.png')
        return
    #----------------------------------------------------------------

    # ----------------------- VALIDATION STEP ---------------------------
    val_data_loader = torch.utils.data.DataLoader(my_data.val_data, batch_size=len(my_data.val_data))
    
    def validation_step():
        model.eval()
        val_mse_loss = 0.0
        X, y = next(iter())

        samples = torch.zeros((num_pred_val, len(my_data.val_data), 3))

        with torch.no_grad():
            # run each data sample through the BNN multiple times
            for k in np.arange(num_pred_val):
                
                samples[k, :, :] = torch.exp(model(X))


        y_pred, y_pred_probs, mean_y_pred_probs = get_most_likely_class(samples)
        val_mse_loss = mse_loss(y_pred.float(), y.float())
        val_ce_loss = ce_loss(mean_y_pred_probs, y)
        val_f1 = f1_score(y, y_pred, average='micro')

        print("** Validation Results**")
        print("-------------------------------------------------------------")
        print("(Validation) Mean Square Error Loss: {}".format(val_mse_loss))
        print("(Validation) Cross_Entropy Loss: {}".format(val_ce_loss))
        print("(Validation) Micro F1 Score: {}".format(val_f1))
        print("")
        return(samples)
    #--------------------------------------------------------------------

    
    # ------------------ SEED SAMPLE TRAINING -------------------------
    seed_data_loader = torch.utils.data.DataLoader(my_data.seed_data, batch_size=batch_size)
    N = len(my_data.seed_data)

    for nepochs in np.arange(num_epochs):

        for id, seed_data in enumerate(tqdm(seed_data_loader, 0)):

            X, y = seed_data

            # make a prediction with the model
            y_pred = model(X, stochastic=True)
            log_prob = loss(y_pred, y)
            vi_model_losses = model.evalAllLosses()
                    
            # negative estimate of ELBO
            f =  N*log_prob + vi_model_losses

            optimizer.zero_grad()
            f.backward()

            optimizer.step()

        validation_step()
        scheduler.step()

    samples = validation_step()
    plot_calibration_curves(samples)
    # ---------------------------------------------------------------------


    def active_data_selection():
        """
        Feed in the currently trained model
        """
        global my_data
        train_loader = torch.utils.data.DataLoader(my_data.train_data, batch_size=batch_size)
        X, y = next(iter(train_loader))
        samples = torch.zeros((num_pred_val, batch_size, 3))

        with torch.no_grad():
            
            for k in np.arange(num_pred_val):

                samples[k, :, :] = torch.exp(model(X))

        y_pred, y_pred_probs, mean_y_pred_probs = get_most_likely_class(samples)

        # calculate the entropy for each estimate (tells how uncertain it is about each)
        # estimate
        H = -1 * np.sum(mean_y_pred_probs * np.log(mean_y_pred_probs), axis=1)

        # get the indicies of the k most uncertain elements
        ind = np.argpartition(H, -k)[-k:]

        # get the data of the must uncertain predictions, add to 
        # the seed data for retraining
        top_k_train_X = X[ind]
        top_k_train_y = y[ind]
        #my_data.update_seed(top_k_train_X, top_k_train_y)

        return

    # ------------------- ACTIVE LEARNING TRAINING ---------------------#

    for r in range(num_rounds):

        active_data_selection(model)

        seed_data_loader = torch.utils.data.DataLoader(my_data.seed_data, batch_size=batch_size)
        N = len(my_data.seed_data)

        for nepochs in np.arange(num_epochs):

            for id, seed_data in enumerate(tqdm(seed_data_loader, 0)):

                X, y = seed_data

                # make a prediction with the model
                y_pred = model(X, stochastic=True)
                log_prob = loss(y_pred, y)
                vi_model_losses = model.evalAllLosses()
                        
                # negative estimate of ELBO
                f =  N*log_prob + vi_model_losses

                optimizer.zero_grad()
                f.backward()

                optimizer.step()

            validation_step()
            scheduler.step()

    #-------------------------------------------------------------------#