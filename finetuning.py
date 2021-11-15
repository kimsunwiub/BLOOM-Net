import torch

import torchaudio
import torch.nn as nn
import torch.utils.data as data

import numpy as np
import time
import copy
from datetime import datetime
from argparse import ArgumentParser
import logging

from data import *
from models import *
from se_utils import *
from utils import *
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("-e", "--tot_epoch", type=int, default=10)
    
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--save_dir", type=str, default="saved_models/")
    
    parser.add_argument("--load_SEmodel", type=str, default=None)
    parser.add_argument("--load_SErundata", type=str, default=None)
    
    parser.add_argument("--snr_ranges", nargs='+', type=int, 
        default=[-5,10])
    
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--validate_every", type=int, default=2)
    
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fft_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
   
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--from_scratch', action='store_true')
    
    parser.add_argument("--ori_l1_dir", type=str, default=None) 
    parser.add_argument("--feat_seq_l2_dir", type=str, default=None)
    parser.add_argument("--feat_seq_l3_dir", type=str, default=None)
    parser.add_argument("--feat_seq_l4_dir", type=str, default=None)
    parser.add_argument("--feat_seq_l5_dir", type=str, default=None)
    parser.add_argument("--feat_seq_l6_dir", type=str, default=None)
    
    return parser.parse_args()
   
args = parse_arguments()
args.n_frames = args.sr * args.duration
args.stft_features = int(args.fft_size//2+1)
args.stft_frames = int(np.ceil(args.n_frames/args.hop_size))
eps = args.eps

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_num_threads(1)
devices = str(args.device)
args.devices = devices
os.environ["CUDA_VISIBLE_DEVICES"]=devices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
args.device = 0
all_seed(args.seed)

t_stamp = '{0:%m%d%H%M}'.format(datetime.now())
output_directory = "{}/comb_expr{}_lr{:.0e}_bs{}_nfrms{}_scrtch{}_GPU{}".format(
    args.save_dir, t_stamp, args.learning_rate, args.batch_size, args.n_frames, args.from_scratch, devices)

handlers = None
if args.is_save:
    os.makedirs(output_directory, exist_ok=True)
    print("Created dir {}".format(output_directory))
    handlers = [
            logging.FileHandler(
                os.path.join(output_directory, "training.log")),
            logging.StreamHandler(),
        ]

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [PID %(process)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )

tr_speech_ds = torchaudio.datasets.LIBRISPEECH(
    "{}/".format(args.data_dir), url="train-clean-100", download=True)
va_speech_ds = torchaudio.datasets.LIBRISPEECH(
    "{}/".format(args.data_dir), url="dev-clean", download=True)

tr_noise_ds = musan_train_prep_dataset(
    '{}/musan/noise/free-sound'.format(args.data_dir))
va_noise_ds = musan_train_prep_dataset(
    '{}/musan/noise/free-sound-va'.format(args.data_dir))

nworkers=0
kwargs = {'num_workers': nworkers, 'pin_memory': True, 'drop_last': True}
tr_speech_dataloader = data.DataLoader(dataset=tr_speech_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn= lambda x: data_processing(x, args.n_frames, "speech"),
    **kwargs)
va_speech_dataloader = data.DataLoader(dataset=va_speech_ds,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn= lambda x: data_processing(x, args.n_frames, "speech"),
    **kwargs)

tr_noise_dataloader = data.DataLoader(dataset=tr_noise_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn= lambda x: data_processing(x, args.n_frames, "noise"),
    **kwargs)
va_noise_dataloader = data.DataLoader(dataset=va_noise_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn= lambda x: data_processing(x, args.n_frames, "noise"),
    **kwargs)

model = BLOOMNETFT2(n_src=1).cuda()
if not args.from_scratch:
    all_layer_dirs = [
        args.ori_l1_dir, 
        args.feat_seq_l2_dir,
        args.feat_seq_l3_dir,
        args.feat_seq_l4_dir,
        args.feat_seq_l5_dir,
        args.feat_seq_l6_dir
    ]

    l_dir = all_layer_dirs[0]
    model_dict = torch.load(l_dir + "Dmodel_last.pt")
    model_dict['masker.mask_net1.0.weight'] = model_dict['masker.mask_net.0.weight']
    del model_dict['masker.mask_net.0.weight']
    model_dict['masker.mask_net1.1.weight'] = model_dict['masker.mask_net.1.weight']
    del model_dict['masker.mask_net.1.weight']
    model_dict['masker.mask_net1.1.bias'] = model_dict['masker.mask_net.1.bias']
    del model_dict['masker.mask_net.1.bias']
    model_dict['decoder1.filterbank._filters'] = model_dict['decoder.filterbank._filters']
    del model_dict['decoder.filterbank._filters']

    for i in range(1,6):
        l_dir = all_layer_dirs[i]
        i_dict = torch.load(l_dir + "Dmodel_last.pt")
        i_dict = edit_dict(i_dict, i)

        for k,v in i_dict.items():
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)        

G_optimizer = torch.optim.Adam(
    params=model.parameters(), lr=args.learning_rate)

# TODO: Change name to include G
load_epoch = 0
best_impr = 0.
tr_losses = []
va_losses = []
     
# Train Generator
logging.info("Started training SE")
prev_best_impr = 1000
    
for epoch in range(load_epoch, args.tot_epoch+load_epoch):
    model.train()
    tr_toc = time.time()
    tr_loss_ep = run_se_ctn_comb(args, model, tr_speech_dataloader, tr_noise_dataloader, G_optimizer)
    tr_losses.append(tr_loss_ep)
    tr_tic = time.time()

    logging.info ("Ep {} Train. Loss: {:.3f} SI-SDRi, Time: {:.2f}s".format(
        epoch, tr_loss_ep, tr_tic-tr_toc))

    if (epoch % args.validate_every) == 0: 
        model.eval()
        va_toc = time.time()
        with torch.no_grad():
            va_loss_ep = run_se_ctn_comb(
                args, model, va_speech_dataloader, va_noise_dataloader, 
                is_train=False)
        va_losses.append(va_loss_ep)

        va_tic = time.time()

        logging.info ("Ep {} Eval. Loss: {:.3f} SI-SDRi, Time: {:.2f}s".format(
            epoch, va_loss_ep, va_tic-va_toc))

        logging.info("Best impr: {:.3f}".format(va_loss_ep))
        if best_impr < prev_best_impr:
            prev_best_impr = copy.deepcopy(best_impr)
            best_impr = va_loss_ep
        rundata = {"epoch": epoch, "sisdr": best_impr, 
                   "tr_losses": tr_losses, "va_losses": va_losses}
        
        if args.is_save:
            if best_impr < prev_best_impr:
                save_model(model, output_directory, rundata)
            save_model(model, output_directory, rundata, is_last=True)
                

logging.info("Finished training SE")
rundata = {"epoch": epoch, "sisdr": best_impr, 
           "tr_losses": tr_losses, "va_losses": va_losses}
if args.is_save:
    save_model(model, output_directory, rundata, is_last=True)
