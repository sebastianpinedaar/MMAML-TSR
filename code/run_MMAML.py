# multimodal learning (with maml), using 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import os
import learn2learn as l2l
from multimodallearner import get_task_encoder_input
from multimodallearner import LSTMDecoder, Lambda, MultimodalLearner
from metalearner import MetaLearner
from meta_base_models import LinearModel, Task
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import json

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from base_models import LSTMModel, FCN
from metrics import torch_mae as mae
from pytorchtools import EarlyStopping, to_torch


def test(loss_fn, maml, multimodal_model, task_data, dataset_name, data_ML, adaptation_steps, learning_rate, noise_level, noise_type, is_test = True, horizon = 10):
    
    total_tasks = len(data_ML)
    task_size = data_ML.x.shape[-3]
    input_dim = data_ML.x.shape[-1]
    window_size = data_ML.x.shape[-2]
    output_dim = data_ML.y.shape[-1]

    if is_test:
        step = total_tasks//100

    else:
        step = 1

    step = 1 if step == 0 else step
    grid = [0., noise_level]
    accum_error = 0.0
    count = 1.0

    for task_idx in range(0, (total_tasks-horizon-1), step):

        temp_file_idx = data_ML.file_idx[task_idx:task_idx+horizon+1]
        if(len(np.unique(temp_file_idx))>1):
            continue
            
        learner = maml.clone() 

        x_spt, y_spt = data_ML[task_idx]
        x_qry = data_ML.x[(task_idx+1):(task_idx+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = data_ML.y[(task_idx+1):(task_idx+1+horizon)].reshape(-1, output_dim)
        task = task_data[task_idx:task_idx+1].cuda()

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

            x_encoding, _  = multimodal_model(x_spt, task, output_encoding=True)
            pred = learner(x_encoding)
            error = loss_fn(pred, y_spt)
            learner.adapt(error)

        x_encoding, _  = multimodal_model(x_qry, task, output_encoding=True)
        y_pred = learner(x_encoding)
        
        y_pred = torch.clamp(y_pred, 0, 1)
        error = mae(y_pred, y_qry)
        
        accum_error += error.data
        
        count += 1
        
    error = accum_error/count

    return error.cpu().numpy()


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    adaptation_steps = args.adaptation_steps
    meta_learning_rate = args.meta_learning_rate
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    task_size = args.task_size
    noise_level = args.noise_level
    noise_type = args.noise_type
    epochs = args.epochs
    loss_fcn_str = args.loss
    modulate_task_net = args.modulate_task_net
    weight_vrae = args.weight_vrae
    stopping_patience = args.stopping_patience
    ml_horizon = args.ml_horizon
    experiment_id = args.experiment_id

    meta_info = {"POLLUTION": [5, 14],
                 "HR": [32, 13],
                 "BATTERY": [20, 3]}

    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, input_dim = meta_info[dataset_name]

    grid = [0., noise_level]

    train_data_ML = pickle.load(
        open("../../Data/TRAIN-" + dataset_name + "-W" + str(window_size) + "-T" + str(task_size) + "-ML.pickle", "rb"))
    validation_data_ML = pickle.load(
        open("../../Data/VAL-" + dataset_name + "-W" + str(window_size) + "-T" + str(task_size) + "-ML.pickle", "rb"))
    test_data_ML = pickle.load(
        open("../../Data/TEST-" + dataset_name + "-W" + str(window_size) + "-T" + str(task_size) + "-ML.pickle", "rb"))

    total_tasks = len(train_data_ML)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = mae if loss_fcn_str == "MAE" else nn.SmoothL1Loss()

    ##multimodal learner parameters
    # paramters wto increase capactiy of the model
    n_layers_task_net = 2
    n_layers_task_encoder = 1
    n_layers_task_decoder = 1

    hidden_dim_task_net = 120
    hidden_dim_encoder = 120
    hidden_dim_decoder = 120

    # fixed values
    input_dim_task_net = input_dim
    input_dim_task_encoder = input_dim + 1
    output_dim_task_net = 1
    output_dim_task_decoder = input_dim + 1
    output_dim = 1

    results_list = []
    results_dict = {}
    results_dict["Experiment_id"] = experiment_id
    results_dict["Model"] = model_name
    results_dict["Dataset"] = dataset_name
    results_dict["Learning rate"] = learning_rate
    results_dict["Noise level"] = noise_level
    results_dict["Task size"] = task_size
    results_dict["Evaluation loss"] = "MAE Test"
    results_dict["Vrae weight"] = weight_vrae
    results_dict["Training"] = "MMAML"
    results_dict["Meta-learning rate"] = meta_learning_rate
    results_dict["ML-Horizon"] = ml_horizon

    for trial in range(lower_trial, upper_trial):

        output_directory = "../../Models/" + dataset_name + "_" + model_name + "_MMAML/" + str(trial) + "/"
        save_model_file_ = output_directory + experiment_id + "_" + save_model_file
        save_model_file_encoder = output_directory + experiment_id + "_"+ "encoder_" + save_model_file
        load_model_file_ = output_directory + load_model_file
        checkpoint_file = output_directory + "checkpoint_" + save_model_file.split(".")[0]

        writer = SummaryWriter()

        try:
            os.mkdir(output_directory)
        except OSError as error:
            print(error)

        task_net = LSTMModel(batch_size=batch_size,
                             seq_len=window_size,
                             input_dim=input_dim_task_net,
                             n_layers=n_layers_task_net,
                             hidden_dim=hidden_dim_task_net,
                             output_dim=output_dim_task_net)

        task_encoder = LSTMModel(batch_size=batch_size,
                                 seq_len=task_size,
                                 input_dim=input_dim_task_encoder,
                                 n_layers=n_layers_task_encoder,
                                 hidden_dim=hidden_dim_encoder,
                                 output_dim=1)

        task_decoder = LSTMDecoder(batch_size=1,
                                   n_layers=n_layers_task_decoder,
                                   seq_len=task_size,
                                   output_dim=output_dim_task_decoder,
                                   hidden_dim=hidden_dim_encoder,
                                   latent_dim=hidden_dim_decoder,
                                   device=device)

        lmbd = Lambda(hidden_dim_encoder, hidden_dim_task_net)

        multimodal_learner = MultimodalLearner(task_net, task_encoder, task_decoder, lmbd, modulate_task_net)
        multimodal_learner.to(device)

        output_layer = nn.Linear(120, 1)
        output_layer.to(device)

        maml = l2l.algorithms.MAML(output_layer, lr=learning_rate, first_order=False)
        opt = optim.Adam(list(maml.parameters()) + list(multimodal_learner.parameters()), lr=meta_learning_rate)

        early_stopping = EarlyStopping(patience=stopping_patience, model_file=save_model_file_, verbose=True)
        early_stopping_encoder = EarlyStopping(patience=stopping_patience, model_file=save_model_file_encoder, verbose=True)

        task_data_train = torch.FloatTensor(get_task_encoder_input(train_data_ML))
        task_data_validation = torch.FloatTensor(get_task_encoder_input(validation_data_ML))
        task_data_test = torch.FloatTensor(get_task_encoder_input(test_data_ML))

        val_loss_hist = []
        test_loss_hist = []
        total_num_tasks = train_data_ML.x.shape[0]

        for iteration in range(epochs):

            opt.zero_grad()
            iteration_error = 0.0
            vrae_loss_accum = 0.0

            multimodal_learner.train()

            for _ in range(batch_size):
                learner = maml.clone()
                task_idx = np.random.randint(0,total_num_tasks-ml_horizon-1)
                task = task_data_train[task_idx:task_idx+1].cuda()

                if train_data_ML.file_idx[task_idx+1] != train_data_ML.file_idx[task_idx]:
                    continue

                x_spt, y_spt = train_data_ML[task_idx]
                x_qry, y_qry = train_data_ML[task_idx + ml_horizon]
                x_qry = x_qry.reshape(-1, window_size, input_dim)
                y_qry = y_qry.reshape(-1, output_dim)

                x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
                x_qry = to_torch(x_qry)
                y_qry = to_torch(y_qry)

                # data augmentation
                epsilon = grid[np.random.randint(0, len(grid))]

                if noise_type == "additive":
                    y_spt = y_spt + epsilon
                    y_qry = y_qry + epsilon
                else:
                    y_spt = y_spt * (1 + epsilon)
                    y_qry = y_qry * (1 + epsilon)

                vrae_loss_accum = 0.0

                x_spt_encoding, (vrae_loss, _, _) = multimodal_learner(x_spt, task,output_encoding=True)
                
                for _ in range(adaptation_steps):
                
                    pred = learner(x_spt_encoding)
                    error = loss_fn(pred, y_spt)
                    learner.adapt(error)#, allow_unused=True)#, allow_nograd=True)
                                                                                        
     
                vrae_loss_accum += vrae_loss

                x_qry_encoding, _ = multimodal_learner(x_qry, task, output_encoding=True)
                pred = learner(x_qry_encoding)
                evaluation_error = loss_fn(pred, y_qry)
                iteration_error += evaluation_error

            iteration_error /= batch_size
            vrae_loss_accum /= batch_size
            iteration_error += weight_vrae*vrae_loss_accum
            iteration_error.backward()
            opt.step()

            multimodal_learner.eval()
            
            val_loss = test(loss_fn, maml, multimodal_learner, task_data_validation, dataset_name, validation_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=10)
            test_loss = test(loss_fn, maml, multimodal_learner, task_data_test, dataset_name, test_data_ML, adaptation_steps, learning_rate, 0, noise_type, horizon=10)
           
            early_stopping_encoder(val_loss, multimodal_learner)
            early_stopping(val_loss, maml)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            print("Epoch:", iteration)

            print("Train loss:", iteration_error)
            print("Val error:", val_loss)
            print("Test error:", test_loss)

            val_loss_hist.append(val_loss)
            test_loss_hist.append(test_loss)

            writer.add_scalar("Loss/train", iteration_error.cpu().detach().numpy(), iteration)
            writer.add_scalar("Loss/val", val_loss, iteration)
            writer.add_scalar("Loss/test", test_loss, iteration)

        multimodal_learner.load_state_dict(torch.load(save_model_file_encoder))
        maml.load_state_dict(torch.load(save_model_file_))

        val_loss = test(loss_fn, maml, multimodal_learner, task_data_validation, dataset_name, validation_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=10)
        test_loss = test(loss_fn, maml, multimodal_learner, task_data_test, dataset_name, test_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=10)

        val_loss1 = test(loss_fn, maml, multimodal_learner, task_data_validation, dataset_name, validation_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=1)
        test_loss1 = test(loss_fn, maml, multimodal_learner, task_data_test, dataset_name, test_data_ML, adaptation_steps, learning_rate, noise_level, noise_type,horizon=1)

        adaptation_steps_ = 0
        val_loss_0 = test(loss_fn, maml, multimodal_learner, task_data_validation, dataset_name, validation_data_ML, adaptation_steps_, learning_rate, noise_level, noise_type,horizon=10)
        test_loss_0 = test(loss_fn, maml, multimodal_learner, task_data_test, dataset_name, test_data_ML, adaptation_steps_, learning_rate, noise_level, noise_type,horizon=10)
        
        val_loss1_0 = test(loss_fn, maml, multimodal_learner, task_data_validation, dataset_name, validation_data_ML, adaptation_steps_, learning_rate, noise_level, noise_type,horizon=1)
        test_loss1_0 = test(loss_fn, maml, multimodal_learner, task_data_test, dataset_name, test_data_ML, adaptation_steps_, learning_rate, noise_level, noise_type,horizon=1)

        with open(output_directory + "/results3.txt", "a+") as f:
            f.write("\n \n Learning rate :%f \n"% learning_rate)
            f.write("Meta-learning rate: %f \n" % meta_learning_rate)
            f.write("Adaptation steps: %f \n" % adaptation_steps)
            f.write("Noise level: %f \n" % noise_level)
            f.write("vrae weight: %f \n" % weight_vrae)
            f.write("Test error: %f \n" % test_loss)
            f.write("Val error: %f \n" % val_loss)
            f.write("Test error 1: %f \n" % test_loss1)
            f.write("Val error 1: %f \n" % val_loss1)
            f.write("Test error 0: %f \n" % test_loss_0)
            f.write("Val error 0: %f \n" % val_loss_0)

        writer.add_hparams({"fast_lr": learning_rate,
                            "slow_lr": meta_learning_rate,
                            "adaption_steps": adaptation_steps,
                            "patience": stopping_patience,
                            "weight_vrae": weight_vrae,
                            "noise_level": noise_level,
                            "dataset": dataset_name,
                            "trial": trial},
                           {"val_loss": val_loss,
                            "test_loss": test_loss})

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = adaptation_steps
        temp_results_dict["Horizon"] = 10
        temp_results_dict["Value"] = float(test_loss)
        temp_results_dict["Val error"] = float(val_loss)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = 0
        temp_results_dict["Horizon"] = 10
        temp_results_dict["Value"] = float(test_loss_0)
        temp_results_dict["Val error"] = float(val_loss_0)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)      

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = adaptation_steps
        temp_results_dict["Horizon"] = 1
        temp_results_dict["Value"] = float(test_loss1)
        temp_results_dict["Final_epoch"] = iteration
        results_list.append(temp_results_dict)

        temp_results_dict = copy.copy(results_dict)
        temp_results_dict["Trial"] = trial
        temp_results_dict["Adaptation steps"] = 0
        temp_results_dict["Horizon"] = 1
        temp_results_dict["Value"] = float(test_loss1_0)
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
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY',
                           default="POLLUTION")
    argparser.add_argument('--model', type=str, help='base model, possible: LSTM, FCN', default="LSTM")
    argparser.add_argument('--adaptation_steps', type=int, help='number of updates in the inner loop', default=5)
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
    argparser.add_argument('--loss', type=str, help='Loss used in training, possible: MAE, SmoothL1', default="MAE")
    argparser.add_argument('--modulate_task_net', type=int,
                           help='Whether to use conditional layer for modulation or not', default=1)
    argparser.add_argument('--weight_vrae', type=float, help='Weight for VRAE', default=0.0)
    argparser.add_argument('--ml_horizon', type=int, help='Horizon for training in time series meta-learning', default=1)
    argparser.add_argument('--experiment_id', type=str, help='experiment_id for the experiments list', default="DEFAULT-ID")

    args = argparser.parse_args()

    main(args)
