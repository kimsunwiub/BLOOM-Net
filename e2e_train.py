import torch
torch.set_num_threads(1)

import torchaudio
import torch.nn as nn
import torch.utils.data as data

import numpy as np
from time import time
from datetime import datetime
from argparse import ArgumentParser
import logging
import copy

from data import *
from models import *
from se_kd_utils import *
from utils import *
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("-e", "--tot_epoch", type=int, default=10)
    parser.add_argument("-r", "--G_num_layers", type=int, default=-1)
    parser.add_argument("-g", "--G_hidden_size", type=int, default=-1)
    
    parser.add_argument("--data_dir", type=str, default="/home/kimsunw/data/")
    parser.add_argument("--save_dir", type=str, default="~/pretraining/")
    
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
                
    parser.add_argument('--is_ctn', action='store_true')    
    parser.add_argument('--is_dprnn', action='store_true')
    parser.add_argument('--is_save', action='store_true')
    
    parser.add_argument('--no_skip_chan', action='store_true')    
    parser.add_argument('--is_mse', action='store_true')
    
    parser.add_argument("--nreps", type=int, default=3)
    parser.add_argument("--nblks", type=int, default=8)
    
    return parser.parse_args()
   
args = parse_arguments()
args.n_frames = args.sr * args.duration
args.stft_features = int(args.fft_size//2+1)
args.stft_frames = int(np.ceil(args.n_frames/args.hop_size))
eps = args.eps

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
args.device = 0
all_seed(args.seed)

t_stamp = '{0:%m%d%H%M}'.format(datetime.now())
output_directory = "{}/expr{}_G{}x{}_lr{:.0e}_bs{}_ctn{}rep{}blk{}_nosk{}_mse{}_nfrms{}_GPU{}".format(
    args.save_dir, t_stamp, args.G_num_layers, args.G_hidden_size, 
    args.learning_rate, args.batch_size, args.is_ctn, args.nreps, args.nblks, args.no_skip_chan, args.is_mse,
    args.n_frames, args.device)

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

kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}
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

# Init generator
if args.is_ctn:
    from asteroid.models import ConvTasNet
    if args.no_skip_chan:
        G_model = ConvTasNet(n_src=1, n_repeats=args.nreps, n_blocks=args.nblks, skip_chan=None).cuda()
    else:
        G_model = ConvTasNet(n_src=1, n_repeats=args.nreps, n_blocks=args.nblks).cuda()
elif args.is_dprnn:    
    from asteroid.models import DPRNNTasNet
    G_model = DPRNNTasNet(n_src=1, n_repeats=args.nreps).cuda()
else:
    G_model = SpeechEnhancementModel(
        args.G_hidden_size, args.G_num_layers, args.stft_features)
G_model = G_model.to(args.device)
G_optimizer = torch.optim.Adam(
    params=G_model.parameters(), lr=args.learning_rate)

# TODO: Change name to include G
load_epoch = 0
best_impr = 0.
tr_losses = []
va_losses = []
if args.load_SEmodel:    
    load_model(G_model, args.load_SEmodel)
    if args.load_SErundata:
        load_SErundata = load_rundata(args.load_SErundata)
        load_epoch = load_SErundata['epoch'] 
#         print (load_SErundata)
        best_impr = load_SErundata['sisdr'] 
        tr_losses = load_SErundata['tr_losses']
        va_losses = load_SErundata['va_losses']
        
        logging.info (
            "Loaded Model at Epoch {} with eval loss: {:.3f} SI-SDRi".format(
            load_epoch, best_impr))
      
# Train Generator
logging.info("Started training SE")
for epoch in range(load_epoch, args.tot_epoch+load_epoch):
    G_model.train()
    tr_toc = time()
    if args.is_ctn or args.is_dprnn:
        tr_loss_ep = run_se_ctn(
            args, G_model, tr_speech_dataloader, tr_noise_dataloader, 
            is_train=True, optimizer=G_optimizer)
    else:
        tr_loss_ep = run_se(
            args, G_model, tr_speech_dataloader, tr_noise_dataloader, 
            is_train=True, optimizer=G_optimizer)
    tr_losses.append(tr_loss_ep)
    tr_tic = time()

    logging.info ("Ep {} Train. Loss: {:.3f} SI-SDRi, Time: {:.2f}s".format(
        epoch, tr_loss_ep, tr_tic-tr_toc))

    if (epoch % args.validate_every) == 0: 
        G_model.eval()
        va_toc = time()
        with torch.no_grad():
            if args.is_ctn or args.is_dprnn:
                va_loss_ep = run_se_ctn(
                    args, G_model, va_speech_dataloader, va_noise_dataloader, 
                    is_train=False)
            else:
                va_loss_ep = run_se(
                    args, G_model, va_speech_dataloader, va_noise_dataloader, 
                    is_train=False)
        va_losses.append(va_loss_ep)

        va_tic = time()

        logging.info ("Ep {} Eval. Loss: {:.3f} SI-SDRi, Time: {:.2f}s".format(
            epoch, va_loss_ep, va_tic-va_toc))

        logging.info("Best impr: {:.3f}".format(va_loss_ep))
        prev_best_impr = copy.deepcopy(best_impr)
        best_impr = va_loss_ep
        rundata = {"epoch": epoch, "sisdr": best_impr, 
                   "tr_losses": tr_losses, "va_losses": va_losses}
        
        if args.is_save:
            if best_impr < prev_best_impr:
                save_model(G_model, output_directory, rundata)
            save_model(G_model, output_directory, rundata, is_last=True)

logging.info("Finished training SE")
rundata = {"epoch": epoch, "sisdr": best_impr, 
           "tr_losses": tr_losses, "va_losses": va_losses}
if args.is_save:
    save_model(G_model, output_directory, rundata, is_last=True)
