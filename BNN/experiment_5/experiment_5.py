#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Pierce Howell
"""
import sched
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse as args
from bnn_model import RichterPredictorBNN
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import f1_score
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.calibration import calibration_curve
from torch.utils.data import Dataset


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
    parser.add_argument('--num-rounds', type=int, default=10, help="Number of rounds for active learning")
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
    num_rounds = args.num_rounds
    K=250
    num_epochs_in_rounds = 3
    # -------------------------------------------------------------------

    # ------------------------------- DATA ------------------#
    class Data:
        """
        This class organizes the data for BNN active learning
        """
        def __init__(self):

            data_dir = "../../data/"
            train_df = pd.read_csv(os.path.join(data_dir, "TRAIN.csv"))
            test_df = pd.read_csv(os.path.join(data_dir, "TEST.csv"))

            # features (ALL)
            categ_features = ["position", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "plan_configuration", "legal_ownership_status", 'land_surface_condition']
            numerical_features = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
            geo_id_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
            binary_features = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                  'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
                  'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
                  'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_engineered',
                  'has_superstructure_other']

            # transform categorical feautres
            X_train_categ = train_df[categ_features].to_numpy()
            X_test_categ = test_df[categ_features].to_numpy()
            enc = OrdinalEncoder()
            enc.fit(X_train_categ)
            X_train_categ = enc.transform(X_train_categ)
            X_test_categ = enc.transform(X_test_categ)

            # normalize numerical features
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train_numer = train_df[numerical_features].values
            X_test_numer = test_df[numerical_features].values
            X_train_numer = min_max_scaler.fit_transform(X_train_numer)
            X_test_numer = min_max_scaler.transform(X_test_numer)

            # Get geoid and binary features
            X_train_geo = train_df[geo_id_features].values
            X_test_geo = test_df[geo_id_features].values
            X_train_bin = train_df[binary_features].values
            X_test_bin = test_df[binary_features].values

            X_train = np.concatenate((X_train_geo, X_train_categ, X_train_numer, X_train_bin), axis=1)
            #X_train = X_train[:100000]
            X_test = np.concatenate((X_test_geo, X_test_categ, X_test_numer, X_test_bin), axis=1)

            # get the labels
            y_train = train_df["damage_grade"].to_numpy() - 1
            #y_train = y_train[:100000]
            y_test = test_df["damage_grade"].to_numpy() - 1

            # split the X, y into training and seed (seed is 5% of original dataset)
            X_train_all = np.copy(X_train); y_train_all = np.copy(y_train)
            X_train, X_seed, y_train, y_seed = train_test_split(X_train, y_train, test_size=0.05)

            # train val split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

            self.in_features = X_train.shape[1]
            self.train_data = CustomDataset(X_train, y_train)
            self.val_data = CustomDataset(X_val ,y_val)
            self.test_data = CustomDataset(X_test, y_test)
            self.seed_data = CustomDataset(X_seed, y_seed)
            self.all_train_data = CustomDataset(X_train_all, y_train_all)
    

    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            label = self.y[idx]
            x = self.X[idx]
            return x, label

        def delete(self, ind):

            self.X = np.delete(self.X, ind, 0)
            self.y = np.delete(self.y, ind, 0)

        def add(self, X, y):
            self.X = np.append(self.X, X, axis=0)
            self.y = np.append(self.y, y, axis=0)


    my_data = Data()
    # ---------------------------------------------------------

    # define loss functions and metrics
    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/BNN_{}'.format(timestamp))

    # train the network
    model = RichterPredictorBNN(in_features=my_data.in_features, p_mc_dropout=None)
    loss = torch.nn.NLLLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    optimizer.zero_grad()

    # ---------------------- CALIBRATION CURVE FUNCTION ---------------
    def plot_calibration_curves(samples, y):

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
        X, y = next(iter(val_data_loader))

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
        return(samples, y)
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

    samples, y_val = validation_step()
    plot_calibration_curves(samples, y_val)
    # # ---------------------------------------------------------------------

    # ---------------------- DEFINE ACTIVE DATA SELECTION FUNCTION------------#

    def active_data_selection():
        """
        Feed in the currently trained model
        """
        global my_data
        
        train_loader = torch.utils.data.DataLoader(my_data.train_data, batch_size=len(my_data.train_data))
        X, y = next(iter(train_loader))
        samples = torch.zeros((num_pred_val, len(my_data.train_data), 3))
        
        with torch.no_grad():
            
            for k in np.arange(num_pred_val):

                samples[k, :, :] = torch.exp(model(X))

        y_pred, y_pred_probs, mean_y_pred_probs = get_most_likely_class(samples)
        
        # calculate the entropy for each estimate (tells how uncertain it is about each)
        # estimate
        #print(mean_y_pred_probs)
        H = -1 * np.sum(mean_y_pred_probs.numpy() * np.log(mean_y_pred_probs.numpy()+1E-16), axis=1)
        #
        # get the indicies of the k most uncertain elements
        ind = np.argpartition(H, -K)[-K:]

        # get the data of the must uncertain predictions, add to 
        # the seed data for retraining
        top_k_train_X = X[ind]
        top_k_train_y = y[ind]
        top_k_train_data = [top_k_train_X, top_k_train_y]
        my_data.train_data.delete(ind)
        
        # print("Before seed data shape", my_data.seed_data.shape)
        my_data.seed_data.add(top_k_train_X, top_k_train_y)
        print("Seed Data Shape", len(my_data.seed_data))
        # print("After seed data shape", my_data.seed_data.shape)
        return

    # ---------------------------------------------------------------------------
    # ------------------- ACTIVE LEARNING TRAINING ---------------------#

    for r in range(num_rounds):

        active_data_selection()
       
        seed_data_loader = torch.utils.data.DataLoader(my_data.seed_data, batch_size=batch_size)
        N = len(my_data.seed_data)

        for nepochs in np.arange(num_epochs_in_rounds):

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

        print("Finished with round {}".format(r))

    #-------------------------------------------------------------------#

    # ------------------- TRAINING ON SAME AMOUNT OF DATA IN TOTALED OF ACTIVE LEARNING

    print("Active Learning Finished; Now statically training a new model on the same set as Seed final.")
    all_train_X = np.copy(my_data.all_train_data.X); all_train_y = np.copy(my_data.all_train_data.y)
    indices = np.arange(all_train_X.shape[0])
    np.random.shuffle(indices)

    all_train_X = all_train_X[indices]
    all_train_y = all_train_y[indices]
    num_final_seed = len(my_data.seed_data)
    all_train_data = CustomDataset(all_train_X[:num_final_seed], all_train_y[:num_final_seed])

    tot_train_data_loader = torch.utils.data.DataLoader(all_train_data, batch_size=batch_size)
    N = len(my_data.seed_data)

    # train the network
    model = RichterPredictorBNN(in_features=my_data.in_features, p_mc_dropout=None)
    loss = torch.nn.NLLLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    optimizer.zero_grad()

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

    samples, y_val = validation_step()
    plot_calibration_curves(samples, y_val)



    #
    plt.show()