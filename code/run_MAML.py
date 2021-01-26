##only fine-tuning the last layer using learn to lern, wtih data augmentaiton
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

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from base_models import LSTMModel, FCN
from metrics import torch_mae as mae
import copy
from pytorchtools import EarlyStopping

def test2(loss_fn, maml, model, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, noise_level, noise_type, is_test = True, horizon = 10):
    

    total_tasks_test = len(test_data_ML)
    error_list =  []

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
        
        if model_name == "LSTM":
            model2 = LSTMModel( batch_size=None, seq_len = None, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
        elif model_name == "FCN":
            kernels = [8,5,3] if window_size != 5 else [4,2,1]
            model2 = FCN(time_steps = window_size,  channels=[input_dim, 128, 128, 128] , kernels=kernels)

        
        #model2.cuda()
        #model2.load_state_dict(copy.deepcopy(maml.module.state_dict()))
        #opt2 = optim.Adam(model2.parameters(), lr=learning_rate)
        learner = maml.clone() 

        x_spt, y_spt = test_data_ML[task]
        x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
        #x_qry = test_data_ML.x[(task+1)].reshape(-1, window_size, input_dim)
        #y_qry = test_data_ML.y[(task+1)].reshape(-1, output_dim)

        if model_name == "FCN":
            x_qry = np.transpose(x_qry, [0,2,1])
            x_spt = np.transpose(x_spt, [0,2,1])

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

        
        #learner.module.train()
        #model2.eval()
        for step in range(adaptation_steps):

            #model2.train()
            pred = learner(model.encoder(x_spt))
            error = loss_fn(pred, y_spt)

            #opt2.zero_grad()
            #error.backward()
              
            learner.adapt(error)
            #opt2.step()
    

        #model2.eval()
        #learner.module.eval()
        y_pred = learner(model.encoder(x_qry))
        
        y_pred = torch.clamp(y_pred, 0, 1)
        error = mae(y_pred, y_qry)
        
        accum_error += error.data
        accum_std += error.data**2
        count += 1
        
    error = accum_error/count

    #print("std:", accum_std/count)
    
    return error.cpu().numpy()   


def test(maml, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, with_early_stopping = False, horizon = 10):

    total_tasks_test = len(test_data_ML)
    error_list =  []

    learner = maml.clone()  # Creates a clone of model
    accum_error = 0.0
    count = 0

    task_size = test_data_ML.x.shape[-3]
    input_dim = test_data_ML.x.shape[-1]
    window_size = test_data_ML.x.shape[-2]
    output_dim = test_data_ML.y.shape[-1]

    for task in range(0, (total_tasks_test-horizon-1), total_tasks_test//100):


        temp_file_idx = test_data_ML.file_idx[task:task+horizon+1]
        if(len(np.unique(temp_file_idx))>1):
            continue       

        if model_name == "LSTM":
            model2 = LSTMModel( batch_size=None, seq_len = None, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
        elif model_name == "FCN":
            kernels = [8,5,3] if window_size != 5 else [4,2,1]
            model2 = FCN(time_steps = window_size,  channels=[input_dim, 128, 128, 128] , kernels=kernels)
        
        model2.cuda()
        model2.load_state_dict(copy.deepcopy(maml.module.state_dict()))
        opt2 = optim.SGD(model2.parameters(), lr=learning_rate)


        if with_early_stopping:

            x_spt, y_spt = test_data_ML[task]

            x_spt_val = x_spt[int(task_size*0.8):]
            y_spt_val = y_spt[int(task_size*0.8):]

            x_spt = x_spt[:int(task_size*0.8)]
            y_spt = y_spt[:int(task_size*0.8)]

            x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
            y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
            
            if model_name == "FCN":
                x_qry = np.transpose(x_qry, [0,2,1])
                x_spt = np.transpose(x_spt, [0,2,1])
                x_spt_val = np.transpose(x_spt_val, [0,2,1])

            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
            x_spt_val, y_spt_val = to_torch(x_spt_val), to_torch(y_spt_val)
            x_qry = to_torch(x_qry)
            y_qry = to_torch(y_qry)
            

        else:


            x_spt, y_spt = test_data_ML[task]
            x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
            y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
            
            if model_name == "FCN":
                x_qry = np.transpose(x_qry, [0,2,1])
                x_spt = np.transpose(x_spt, [0,2,1])

            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
            x_qry = to_torch(x_qry)
            y_qry = to_torch(y_qry)

        early_stopping = EarlyStopping(patience=2, model_file="temp/temp_file_"+model_name+".pt", verbose=True)
        
        #model2.eval()

        

        for step in range(adaptation_steps):

            model2.zero_grad()
            opt2.zero_grad()

            #model2.train()
            pred = model2(x_spt)
            error = mae(pred, y_spt)
                        
            error.backward()
  
            opt2.step()
    
            if with_early_stopping:
                with torch.no_grad():
                    #model2.eval()
                    pred = model2(x_spt_val)
                    error = mae(pred, y_spt_val)
                early_stopping(error, model2)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        if with_early_stopping:
            model2.load_state_dict(torch.load("temp/temp_file_"+model_name+".pt"))
        #model2.eval()
        pred = model2(x_qry)
        
        y_pred = torch.clamp(y_pred, 0, 1)
        error = mae(pred, y_qry)
        
        accum_error += error.data
        count += 1
        
    error = accum_error/count
    
    return error


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
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    learning_rate = args.learning_rate
    meta_learning_rate = args.meta_learning_rate
    adaptation_steps = args.adaptation_steps
    batch_size = args.batch_size
    model_name = args.model
    is_test = args.is_test
    patience_stopping = args.stopping_patience
    epochs = args.epochs
    noise_level = args.noise_level
    noise_type = args.noise_type
    ml_horizon = args.ml_horizon
    experiment_id = args.experiment_id

    window_size, task_size, input_dim = meta_info[dataset_name]

    task_size = args.task_size

    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    
    grid = [0., noise_level]

    train_data = pickle.load(  open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

    loss_fn = mae

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


    #loss_fn = nn.SmoothL1Loss()

    for trial in range(lower_trial, upper_trial):

        output_directory = "../../Models/"+dataset_name+"_"+model_name+"_MAML/"+str(trial)+"/"

        save_model_file_ = output_directory +  experiment_id + "_"+"encoder_"+save_model_file
        save_model_file_2 = output_directory  + experiment_id + "_"+ save_model_file
        load_model_file_ = output_directory + load_model_file

        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)

        with open(output_directory+"/results3.txt", "a+") as f:
            f.write("Learning rate :%f \n"% learning_rate)
            f.write("Meta-learning rate: %f \n" % meta_learning_rate)
            f.write("Adaptation steps: %f \n" % adaptation_steps)
            f.write("Noise level: %f \n" % noise_level)
            f.write("\n")   

        if model_name == "LSTM":
            model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
            model2 = nn.Linear(120, 1)
        elif model_name == "FCN":
            kernels = [8,5,3] if dataset_name!= "POLLUTION" else [4,2,1]
            model = FCN(time_steps = window_size,  channels=[input_dim, 128, 128, 128] , kernels=kernels)   
            model2 = nn.Linear(128, 1)

        model.cuda()
        model2.cuda()
        
        maml = l2l.algorithms.MAML(model2, lr=learning_rate, first_order=False)
        opt = optim.Adam(list(maml.parameters()) + list(model.parameters()), lr=meta_learning_rate)

        #torch.backends.cudnn.enabled = False
        total_num_tasks = train_data_ML.x.shape[0]
        #test2(maml, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate)
        #val_error = test(maml, model_name, dataset_name, validation_data_ML, adaptation_steps, learning_rate)

        early_stopping = EarlyStopping(patience=patience_stopping, model_file=save_model_file_, verbose=True)
        early_stopping2 = EarlyStopping(patience=patience_stopping, model_file=save_model_file_2, verbose=True)

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience =200, verbose=True)

        #early_stopping(val_error, maml)

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
                #task_qry = np.random.randint(1,horizon+1)


                x_spt, y_spt = train_data_ML[task]
                #x_qry, y_qry = train_data_ML[(task+1):(task+1+horizon)]
                x_qry, y_qry = train_data_ML[task+ml_horizon]
                #x_qry, y_qry = train_data_ML[task_qry]

                x_qry = x_qry.reshape(-1, window_size, input_dim)
                y_qry = y_qry.reshape(-1, output_dim)

                if model_name == "FCN":
                    x_qry = np.transpose(x_qry, [0,2,1])
                    x_spt = np.transpose(x_spt, [0,2,1])
                
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
                    
                    pred = learner(model.encoder(x_spt))
                    error = loss_fn(pred, y_spt)
                    learner.adapt(error)#, allow_unused=True)#, allow_nograd=True)

                pred = learner(model.encoder(x_qry))
                evaluation_error = loss_fn(pred, y_qry)
                iteration_error += evaluation_error
                #evaluation_error.backward()

            # Meta-update the model parameters
            #for p in maml.parameters():
                #p.grad.data.mul_(1.0 / batch_size)
            iteration_error /= batch_size
            iteration_error.backward()#retain_graph = True)
            #print("loss iteration:",iteration_error)
            opt.step()
            
            if iteration%1 == 0:
                val_error = test2(loss_fn, maml, model, model_name, dataset_name, validation_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=10)
                test_error = test2(loss_fn, maml, model, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type, horizon=10)
                #scheduler.step(val_error)
                print("Val error:", val_error)
                print("Test error:", test_error)

                if iteration> 10:
                    early_stopping(val_error, model)
                    early_stopping2(val_error, maml)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        model.load_state_dict(torch.load(save_model_file_))
        maml.load_state_dict(torch.load(save_model_file_2))

        validation_error = test2(loss_fn, maml, model, model_name, dataset_name, validation_data_ML, adaptation_steps, learning_rate,0, noise_type)
        initial_val_error = test2(loss_fn, maml, model, model_name, dataset_name, validation_data_ML, 0, learning_rate,0, noise_type)

        test_error = test2(loss_fn, maml, model, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type)
        initial_test_error = test2(loss_fn, maml, model, model_name, dataset_name, test_data_ML, 0, learning_rate, 0, noise_type)

        test_error2 = test2(loss_fn, maml, model, model_name, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type, horizon=1)
        initial_test_error2 = test2(loss_fn, maml, model, model_name, dataset_name, test_data_ML, 0, learning_rate, 0, noise_type, horizon=1)


        with open(output_directory+"/results3.txt", "a+") as f:
            f.write("Dataset :%s \n"% dataset_name)
            f.write("Test error: %f \n" % test_error)
            f.write("Test error2: %f \n" % test_error2)
            f.write("Initial Test error: %f \n" % initial_test_error)
            f.write("Initial Test error2: %f \n" % initial_test_error2)
            f.write("Validation error: %f \n" %validation_error)
            f.write("Initial validation error: %f \n" %initial_val_error)

            f.write("\n")
        
        print("Adaptation_steps:", adaptation_steps)
        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = adaptation_steps
        temp_results_dict["Horizon"] = 10
        temp_results_dict["Value"] = float(test_error)
        temp_results_dict["Val error"] = float(validation_error)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = 0
        temp_results_dict["Horizon"] = 10
        temp_results_dict["Value"] = float(initial_test_error ) 
        temp_results_dict["Val error"] = float(initial_val_error)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)      

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = adaptation_steps
        temp_results_dict["Horizon"] = 1
        temp_results_dict["Value"] = float(test_error2)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = 0
        temp_results_dict["Horizon"] = 1
        temp_results_dict["Value"] = float(initial_test_error2)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)  

    try:
        os.mkdir("../../Results/json_files/")
    except OSError as error:
        print(error)
        
    with open("../../Results/json_files/"+experiment_id+".json", 'w') as outfile:
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
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)
    argparser.add_argument('--stopping_patience', type=int, help='patience for early stopping', default=500)
    argparser.add_argument('--epochs', type=int, help='epochs', default=20000)
    argparser.add_argument('--noise_level', type=float, help='noise level', default=0.0)
    argparser.add_argument('--noise_type', type=str, help='noise type', default="additive")
    argparser.add_argument('--task_size', type=int, help='Task size', default=50)
    argparser.add_argument('--ml_horizon', type=int, help='Horizon for training in time series meta-learning', default=1)
    argparser.add_argument('--experiment_id', type=str, help='experiment_id for the experiments list', default="DEFAULT-ID")

    args = argparser.parse_args()

    main(args)