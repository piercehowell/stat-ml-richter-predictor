# stat-ml-richter-predictor
Final Project for ECE 6524 (Statistical Machine Learning / Spring 2022) 

### Bayesian Neural Network Model

To run the bayesian neural network model, navigate to the directory `BNN/experiment_4`. A trained model is saved, and can be evaluated on the held out test dataset with

```cmd
python3 experiment_4.py --num-pred-val=50 --no-train
```

To retrain the model, run
```cmd
python3 experiment_4.py --num-epochs=10 --learning-rate=0.01 --num-pred-val=50 --batch-size=64 --save-model
```
