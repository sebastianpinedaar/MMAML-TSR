import learn2learn as l2l
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
import argparse
import os
import json
import copy

sys.path.append("pre_processing")
sys.path.append("models")
sys.path.append("tools")

from ts_dataset import TSDataset
from base_models import LSTMModel
from metrics import torch_mae as mae
from pytorchtools import EarlyStopping

def test(loss_fn, maml, model, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, noise_level, noise_type, is_test = True, horizon = 10):
    
    total_tasks_test = len(test_data_ML)

    learner = maml.clone()  # Creates a clone of model
    learner.cuda()
    accum_error = 0.0
    accum_std = 0.0
    count = 0.0
    grid = [0., noise_level]

    input_dim = test_data_ML.x.shape[-1]
    window_size = test_data_ML.x.shape[-2]
    output_dim = test_data_ML.y.shape[-1]

    if is_test:
        step = total_tasks_test//100

    else:
        step = 1

    step = 1 if step == 0 else step
    
    for task in range(0, (total_tasks_test-horizon-1), step):

        temp_file_idx = test_data_ML.file_idx[task:task+horizon+1]
        if(len(np.unique(temp_file_idx))>1):
            continue
        

        learner = maml.clone() 

        x_spt, y_spt = test_data_ML[task]
        x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)


        x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
        x_qry = to_torch(x_qry)
        y_qry = to_torch(y_qry)

        epsilon = grid[np.random.randint(0,len(grid))]

        if noise_type == "additive":
            y_spt = y_spt+epsilon
            y_qry = y_qry+epsilon

        else:
            y_spt = y_spt*(1+epsilon)
            y_qry = y_qry*(1+epsilon)


        for step in range(adaptation_steps):

            pred = learner(model.encoder(x_spt))
            error = loss_fn(pred, y_spt)      
            learner.adapt(error)
    
        y_pred = learner(model.encoder(x_qry))
        
        y_pred = torch.clamp(y_pred, 0, 1)
        error = mae(y_pred, y_qry)       
        accum_error += error.data
        accum_std += error.data**2
        count += 1
        
    error = accum_error/count
    
    return error.cpu().numpy()   


def to_torch(numpy_tensor):
    
    return torch.tensor(numpy_tensor).float().cuda()

def main(args):

    meta_info = {"POLLUTION": [5, 50, 14],
                 "HR": [32, 50, 13],
                 "BATTERY": [20, 50, 3] }

    output_directory = "output/"
    horizon = 10
    output_dim = 1
    
    dataset_name = args.dataset 
    save_model_file = args.save_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    learning_rate = args.learning_rate
    meta_learning_rate = args.meta_learning_rate
    adaptation_steps = args.adaptation_steps
    batch_size = args.batch_size
    model_name = args.model
    patience_stopping = args.stopping_patience
    epochs = args.epochs
    noise_level = args.noise_level
    noise_type = args.noise_type
    ml_horizon = args.ml_horizon
    experiment_id = args.experiment_id

    window_size, task_size, input_dim = meta_info[dataset_name]

    task_size = args.task_size

    #currently only LSTM is supported as task network
    assert model_name in ("LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    grid = [0., noise_level]

    #load data
    train_data_ML = pickle.load( open( "../data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

    #fixing loss
    loss_fn = mae
    #loss_fn = nn.SmoothL1Loss()

    #json for outputting results
    results_list = []
    results_dict = {}
    results_dict["Experiment_id"] = experiment_id
    results_dict["Model"] = model_name
    results_dict["Dataset"] = dataset_name
    results_dict["Learning rate"] = learning_rate
    results_dict["Noise level"] = noise_level
    results_dict["Task size"] = task_size
    results_dict["Evaluation loss"] = "MAE Test"
    results_dict["Vrae weight"] = "-"
    results_dict["Training"] = "MAML"
    results_dict["ML-Horizon"] = ml_horizon
    results_dict["Meta-learning rate"] = meta_learning_rate

    for trial in range(lower_trial, upper_trial):

        output_directory = "../logs/MAML_output/"+str(trial)+"/"

        save_model_file_encoder = output_directory +  experiment_id + "_"+"encoder_"+save_model_file
        save_model_file_output_layer = output_directory  + experiment_id + "_"+ save_model_file

        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)

        with open(output_directory+"/results.txt", "a+") as f:
            f.write("Learning rate :%f \n"% learning_rate)
            f.write("Meta-learning rate: %f \n" % meta_learning_rate)
            f.write("Adaptation steps: %f \n" % adaptation_steps)
            f.write("Noise level: %f \n" % noise_level)
            f.write("\n")   

       
        encoder = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
        output_layer = nn.Linear(120, 1)

        encoder.cuda()
        output_layer.cuda()
        
        maml = l2l.algorithms.MAML(output_layer, lr=learning_rate, first_order=False)
        opt = optim.Adam(list(maml.parameters()) + list(encoder.parameters()), lr=meta_learning_rate)

        total_num_tasks = train_data_ML.x.shape[0]

        early_stopping_encoder = EarlyStopping(patience=patience_stopping, model_file=save_model_file_encoder, verbose=True)
        early_stopping_output_layer = EarlyStopping(patience=patience_stopping, model_file=save_model_file_output_layer, verbose=True)

        for iteration in range(epochs):

            # Creates a clone of model
            opt.zero_grad()
            iteration_error = 0.0
            
            print(iteration)
            for task in range(batch_size):
                learner = maml.clone()
                task = np.random.randint(0,total_num_tasks-horizon)

                if train_data_ML.file_idx[task+1] != train_data_ML.file_idx[task]:
                    continue

                x_spt, y_spt = train_data_ML[task]
                x_qry, y_qry = train_data_ML[task+ml_horizon]

                x_qry = x_qry.reshape(-1, window_size, input_dim)
                y_qry = y_qry.reshape(-1, output_dim)
        
                x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
                x_qry = to_torch(x_qry)
                y_qry = to_torch(y_qry)
                

                #data augmentation
                epsilon = grid[np.random.randint(0,len(grid))]

                if noise_type == "additive":
                    y_spt = y_spt+epsilon
                    y_qry = y_qry+epsilon

                else:
                    y_spt = y_spt*(1+epsilon)
                    y_qry = y_qry*(1+epsilon)
                
                # Fast adapt
                for _ in range(adaptation_steps):
                    
                    pred = learner(encoder.encoder(x_spt))
                    error = loss_fn(pred, y_spt)
                    learner.adapt(error)

                pred = learner(encoder.encoder(x_qry))
                evaluation_error = loss_fn(pred, y_qry)
                iteration_error += evaluation_error
                

            # Meta-update the model parameters

            iteration_error /= batch_size
            iteration_error.backward()
            
            opt.step()
            
            if iteration%1 == 0:
                val_error = test(loss_fn, maml, encoder, model_name, dataset_name, validation_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=10)
                test_error = test(loss_fn, maml, encoder, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type, horizon=10)
                
                print("Val error:", val_error)
                print("Test error:", test_error)

                if iteration> 10:
                    early_stopping_encoder(val_error, encoder)
                    early_stopping_output_layer(val_error, maml)

                if early_stopping_encoder.early_stop:
                    print("Early stopping")
                    break

        encoder.load_state_dict(torch.load(save_model_file_encoder))
        maml.load_state_dict(torch.load(save_model_file_output_layer))

        validation_error_horizon_10 = test(loss_fn, maml, encoder, model_name, dataset_name, validation_data_ML, adaptation_steps, learning_rate,0, noise_type)
        initial_val_error_horizon_10 = test(loss_fn, maml, encoder, model_name, dataset_name, validation_data_ML, 0, learning_rate,0, noise_type)

        test_error_horizon_10 = test(loss_fn, maml, encoder, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type)
        initial_test_error_horizon_10 = test(loss_fn, maml, encoder, model_name, dataset_name, test_data_ML, 0, learning_rate, 0, noise_type)

        test_error_horizon_1 = test(loss_fn, maml, encoder, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type, horizon=1)
        initial_test_error_horizon_1 = test(loss_fn, maml, encoder, model_name, dataset_name, test_data_ML, 0, learning_rate, 0, noise_type, horizon=1)


        with open(output_directory+"/results.txt", "a+") as f:
            f.write("Dataset :%s \n"% dataset_name)
            f.write("Test error 1 hor.: %f \n" % test_error_horizon_1)
            f.write("Test error 10 hor.: %f \n" % test_error_horizon_10)
            f.write("Initial Test error 1 hor.: %f \n" % initial_test_error_horizon_1)
            f.write("Initial Test error 10 hor.: %f \n" % initial_test_error_horizon_10)
            f.write("Validation error: %f \n" %validation_error_horizon_10)
            f.write("Initial validation error: %f \n" %initial_val_error_horizon_10)

            f.write("\n")
        
        print("Adaptation_steps:", adaptation_steps)
        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = adaptation_steps
        temp_results_dict["Horizon"] = 10
        temp_results_dict["Value"] = float(test_error_horizon_10)
        temp_results_dict["Val error"] = float(validation_error_horizon_10)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = 0
        temp_results_dict["Horizon"] = 10
        temp_results_dict["Value"] = float(initial_test_error_horizon_10 ) 
        temp_results_dict["Val error"] = float(initial_val_error_horizon_10)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)      

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = adaptation_steps
        temp_results_dict["Horizon"] = 1
        temp_results_dict["Value"] = float(test_error_horizon_1)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = 0
        temp_results_dict["Horizon"] = 1
        temp_results_dict["Value"] = float(initial_test_error_horizon_1)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)  

    try:
        os.mkdir("../Results/")
    except OSError as error:
        print(error)
        
    with open("../Results/"+experiment_id+".json", 'w') as outfile:
        json.dump(results_list, outfile)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY', default="POLLUTION")
    argparser.add_argument('--model', type=str, help='base model, possible: LSTM, FCN', default="LSTM")
    argparser.add_argument('--adaptation_steps', type=int, help='number of updates in the inner loop', default=1)
    argparser.add_argument('--learning_rate', type=float, help='learning rate for the inner loop', default=0.01)
    argparser.add_argument('--meta_learning_rate', type=float, help='learning rate for the outer loop', default=0.005)
    argparser.add_argument('--batch_size', type=int, help='batch size for the meta-upates (outer loop)', default=20)
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--stopping_patience', type=int, help='patience for early stopping', default=500)
    argparser.add_argument('--epochs', type=int, help='epochs', default=20000)
    argparser.add_argument('--noise_level', type=float, help='noise level', default=0.0)
    argparser.add_argument('--noise_type', type=str, help='noise type', default="additive")
    argparser.add_argument('--task_size', type=int, help='Task size', default=50)
    argparser.add_argument('--ml_horizon', type=int, help='Horizon for training in time series meta-learning', default=1)
    argparser.add_argument('--experiment_id', type=str, help='experiment_id for the experiments list', default="DEFAULT-ID")

    args = argparser.parse_args()

    main(args)