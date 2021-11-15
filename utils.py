import pickle
import torch.nn as nn
import torch
import numpy as np
import os
from datetime import datetime
import random
import logging

eps = 1e-6

def pt_to_np(X):
    return X.detach().cpu().numpy()

def make_np(x):
    return x.detach().cpu().numpy()

def get_stats(X):
    return print("Mean: {:.3f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(
        X.mean(), X.std(), X.min(), X.max()))

def save_model(model, output_directory, rundata, is_last=False, model2=None):
#     curr_epoch = rundata['epoch']
    if is_last:
        suffix = 'last'
    else:
        suffix = 'best'
    
    model_save_dir = "{}/Dmodel_{}.pt".format(output_directory, suffix)
    data_save_dir = "{}/rundata_{}.pt".format(output_directory, suffix)
    torch.save(model.state_dict(), model_save_dir)
    logging.info("D model saved to {}".format(model_save_dir))
    with open(data_save_dir, 'wb') as handle:
        pickle.dump(rundata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if model2:
        model_save_dir = "{}/Gmodel_{}.pt".format(output_directory, suffix)
        torch.save(model2.state_dict(), model_save_dir)
        logging.info("G model saved to {}".format(model_save_dir))

def load_model(model, load_model):
    state_dict = torch.load(load_model, map_location=torch.device("cpu"))#map_location=lambda storage, loc: storage) 
    model.load_state_dict(state_dict)
    del state_dict
    logging.info("Loaded model from {}".format(load_model))

def load_rundata(load_rundata):
    with open(load_rundata, 'rb') as handle:
        rundata = pickle.load(handle)
    logging.info("Loaded rundata from {}".format(load_rundata))
    return rundata

def all_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_expr(args):
    args.n_frames = args.sr * args.duration
    args.stft_features = int(args.fft_size//2+1)
    args.stft_frames = int(np.ceil(args.n_frames/args.hop_size))+1

    t_stamp = '{0:%m%d%H%M}'.format(datetime.now())

    if args.ctn_tea:
        tea_opt = "CTN"
    else:
        tea_opt = "{}x{}".format(args.teacher_num_layers, args.teacher_hidden_size)
    if args.sisdr_loss:
        tea_opt += "_Lsisdr"
    output_directory = "{}/stu{}x{}_tea{}/seed{}/snr{}/lr{:.0e}/expr{}_bs{}_nfrm{}_GPU{}".format(
        args.save_dir, 
        args.student_num_layers, args.student_hidden_size, tea_opt, 
        args.seed, 
        args.snr_ranges[0],
        args.learning_rate,
        t_stamp,
        args.batch_size, args.n_frames, 
        args.device)

    print("Output Dir: {}".format(output_directory))
    if args.is_save:
        os.makedirs(output_directory, exist_ok=True)
        print("Created dir...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [PID %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(output_directory, "training.log")),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [PID %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    return output_directory

