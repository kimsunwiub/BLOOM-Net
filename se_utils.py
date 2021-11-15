import numpy as np
import torch
import torch.nn as nn
   
from data import mix_signals_batch, prep_sig_ml

import time
import pickle
import copy
from os.path import exists
from utils import all_seed

from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape

eps = 1e-6

def stft(signal, fft_size, hop_size):
    window = torch.hann_window(fft_size, device=signal.device)
    S = torch.stft(signal, n_fft=fft_size, hop_length=hop_size, window=window)#, return_complex=False)
    return S

def get_magnitude(S):
    S_mag = torch.sqrt(S[..., 0] ** 2 + S[..., 1] ** 2 + 1e-20)
    return S_mag

def apply_mask(spectrogram, mask, device):
    assert (spectrogram[...,0].shape == mask.shape)
    spectrogram2 = torch.zeros(spectrogram.shape)
    spectrogram2[..., 0] = spectrogram[..., 0] * mask
    spectrogram2[..., 1] = spectrogram[..., 1] * mask
    return spectrogram2.to(device)

def istft(spectrogram, fft_size, hop_size):
    window = torch.hann_window(fft_size, device=spectrogram.device)
    y = torch.istft(spectrogram, n_fft=fft_size, hop_length=hop_size, window=window)
    return y

# Train Utils
def calculate_sdr(source_signal, estimated_signal, offset=None, scale_invariant=False):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    s = source_signal.clone()
    y = estimated_signal.clone()

    # add a batch axis if non-existant
    if len(s.shape) != 2:
        s = s.unsqueeze(0)
        y = y.unsqueeze(0)

    # truncate all signals in the batch to match the minimum-length
    min_length = min(s.shape[-1], y.shape[-1])
    s = s[..., :min_length]
    y = y[..., :min_length]

    if scale_invariant:
        alpha = s.mm(y.T).diag()
        alpha /= ((s ** 2).sum(dim=1) + eps)
        alpha = alpha.unsqueeze(1)  # to allow broadcasting
    else:
        alpha = 1

    e_target = s * alpha
    e_res = e_target - y

    numerator = (e_target ** 2).sum(dim=1)
    denominator = (e_res ** 2).sum(dim=1) + eps
    sdr = 10 * torch.log10((numerator / denominator) + eps)

    # if `offset` is non-zero, this function returns the relative SDR
    # improvement for each signal in the batch
    if offset is not None:
        sdr -= offset

    return sdr

def calculate_sisdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return calculate_sdr(source_signal, estimated_signal, offset, True)

def loss_sdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return -1.*torch.mean(calculate_sdr(source_signal, estimated_signal, offset))

def loss_sisdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return -1.*torch.mean(calculate_sisdr(source_signal, estimated_signal, offset))

def denoise_signal(args, mix_batch, G_model):
    """
    Return predicted clean speech.
    
    mix_batch and G_model: Located on GPU.
    """
    X = stft(mix_batch, args.fft_size, args.hop_size)
    X_mag = get_magnitude(X).permute(0,2,1)
    mask_pred = G_model(X_mag).permute(0,2,1)
    mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
    X_est = X * mask_pred
    est_batch = istft(X_est, args.fft_size, args.hop_size)
    return est_batch

def denoise_signal_ctn(args, mix_batch, G_model):
    return G_model(mix_batch).squeeze(1)

def run_iter(args, tot_s, tot_x, student_model, ori_student_model, teacher_model, student_optimizer=None, trtt=None):
    stu_res = []
    tea_res = []
    ori_res = []
    loss_res = []
    for idx in range(0,len(tot_s),args.batch_size):
        speech_batch = tot_s[idx:idx+args.batch_size].to(torch.device("cuda"))
        mix_batch = tot_x[idx:idx+args.batch_size]
        
        stu_e = denoise_signal(args, mix_batch.to(torch.device("cuda")), student_model)
        if args.ctn_tea:
            if trtt == 'test': # Compute signals individually since test signals are 10s long and overloads GPU memory.
                tea_e = []
                for x in mix_batch:
                    x = x[None,:]
                    tea_e_i = denoise_signal_ctn(args, x.to(torch.device("cuda")), teacher_model).squeeze(1)
                    tea_e_i = tea_e_i.detach().cpu()
                    _, tea_e_i, _ = prep_sig_ml(tea_e_i, stu_e)
                    tea_e.append(tea_e_i)
                tea_e = torch.stack(tea_e).squeeze(1)
            else:
                tea_e = denoise_signal_ctn(args, mix_batch.to(torch.device("cuda")), teacher_model)
                _, tea_e, _ = prep_sig_ml(tea_e, stu_e)
            tea_e = tea_e.detach().cpu().to(torch.device("cuda"))
        else:
            tea_e = denoise_signal(args, mix_batch.to(torch.device("cuda")), teacher_model)
        mix_batch = mix_batch.to(torch.device("cuda"))
        ori_e = denoise_signal(args, mix_batch, ori_student_model)

        # Truncate to same lengths
        _, s, _ = prep_sig_ml(speech_batch, stu_e)
        _, x, _ = prep_sig_ml(mix_batch, stu_e)

        # Standardize
        s = s/(s.std(1)[:,None] + eps)
        x = x/(x.std(1)[:,None] + eps)
        stu_e = stu_e/(stu_e.std(1)[:,None] + eps)
        ori_e = ori_e/(ori_e.std(1)[:,None] + eps)
        tea_e = tea_e/(tea_e.std(1)[:,None] + eps)

        stu_sdr = float(calculate_sisdr(s, stu_e).mean())
        tea_sdr = float(calculate_sisdr(s, tea_e).mean())
        ori_sdr = float(calculate_sisdr(s, ori_e).mean())
        
        stu_res.append(stu_sdr)
        tea_res.append(tea_sdr)
        ori_res.append(ori_sdr)

        if args.sisdr_loss:
            mix_offset = calculate_sisdr(tea_e, x) # Offset is computed with s_T as ground-truth. 
            loss_i = loss_sisdr(tea_e, stu_e, offset=mix_offset) 
        else:
            loss_i = loss_fn(tea_e, stu_e)
            
        if student_optimizer:
            student_optimizer.zero_grad()
            loss_i.backward()
            student_optimizer.step()
            
        loss_res.append(float(loss_i))
        del loss_i

    return np.mean(stu_res), np.mean(tea_res), np.mean(ori_res), np.mean(loss_res)       


def run_se(args, model, speech_dataloader, noise_dataloader, is_train=True, optimizer=None):
        total_loss = np.zeros(len(speech_dataloader))
        
        noise_iter = iter(noise_dataloader)
        for batch_idx, speech_batch in enumerate(speech_dataloader):
            try:
                noise_batch = next(noise_iter)
            except StopIteration:
                noise_iter = iter(noise_dataloader)
                noise_batch = next(noise_iter)
                
            mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
            X = stft(mix_batch, args.fft_size, args.hop_size)
            X_mag = get_magnitude(X)
            X_mag = X_mag.permute(0,2,1)
            mask_pred = model(X_mag)
            mask_pred = mask_pred.permute(0,2,1)
            mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
            X_est = X * mask_pred
            est_batch = istft(X_est, args.fft_size, args.hop_size)

            if is_train:
                optimizer.zero_grad()

            speech_batch = speech_batch.to(torch.device("cuda"))
            actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
            loss_i = loss_sisdr(speech_batch, est_batch, actual_sisdr)
            total_loss[batch_idx] = loss_i

            if is_train:
                loss_i.backward()
                optimizer.step()

            if (batch_idx % args.print_every) == 0: 
                print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                        batch_idx, total_loss[:batch_idx+1].mean())) # logging.info
        
        return total_loss[:batch_idx+1].mean()

def run_se_ctn(args, model, speech_dataloader, noise_dataloader, is_train=True, optimizer=None):
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        est_batch = model(mix_batch).squeeze(1)

        if is_train:
            optimizer.zero_grad()

        speech_batch = speech_batch.to(torch.device("cuda"))
        actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
        loss_i = loss_sisdr(speech_batch, est_batch)#, actual_sisdr)
        total_loss[batch_idx] = float(loss_i)

        if is_train:
            loss_i.backward()
            optimizer.step()
        
        del loss_i
        torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()

def run_se_ctn_trloss(args, model, speech_dataloader, noise_dataloader): 
    all_seed(0)
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        with torch.no_grad():
            est_batch = model(mix_batch).squeeze(1)
            speech_batch = speech_batch.to(torch.device("cuda"))
            actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
            loss_i = loss_sisdr(speech_batch, est_batch, actual_sisdr)
            total_loss[batch_idx] = float(loss_i)
            del loss_i
            torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()


def run_se_ctn_trloss_seq(args, models, speech_dataloader, noise_dataloader): 
    all_seed(0)
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        with torch.no_grad():
            est_batch = models[0](mix_batch).squeeze(1)
            for i,model in enumerate(models[1:]):
                est_batch = models[1+i](est_batch).squeeze(1)
            speech_batch = speech_batch.to(torch.device("cuda"))
            actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
            loss_i = loss_sisdr(speech_batch, est_batch, actual_sisdr)
            total_loss[batch_idx] = float(loss_i)
            del loss_i
            torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()


    

def generate_sample_scores(args, model, speech_dataloader, noise_dataloader, seed, prev_weights=None):
    st = time.time()
    all_seed(seed)
    if exists('{}/seed{}.npy'.format(args.save_dir, seed)):
        print("Exists")
    else:
        sisdr_t = torch.zeros(len(speech_dataloader) * args.batch_size)
        print("DEBUG: ", len(speech_dataloader) * args.batch_size)

        noise_iter = iter(noise_dataloader)
        for batch_idx, speech_batch in enumerate(speech_dataloader):
            if batch_idx % 10 == 0:
                print("{:.2f}. {}/{}".format(time.time()-st, batch_idx,len(speech_dataloader)))
            try:
                noise_batch = next(noise_iter)
            except StopIteration:
                noise_iter = iter(noise_dataloader)
                noise_batch = next(noise_iter)

            mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
            with torch.no_grad():
                est_batch = model(mix_batch).squeeze(1)
            est_sisdr = calculate_sisdr(speech_batch.to(torch.device("cuda")), est_batch)
            sisdr_t[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = est_sisdr.detach().cpu()

        np.save('{}/seed{}'.format(args.save_dir, seed), sisdr_t)
        
        
def generate_sample_scores_debug(args, model, speech_dataloader, noise_dataloader, seed, prev_weights=None):
    def make_np(x):
        return x.detach().cpu().numpy()
    st = time.time()
    all_seed(seed)
    if exists('{}/seed{}.npy'.format(args.save_dir, seed)):
        print("Exists")
    else:
        sisdr_t = torch.zeros(28539)

        noise_iter = iter(noise_dataloader)
        for batch_idx, speech_batch in enumerate(speech_dataloader):
            if batch_idx % 10 == 0:
                print("{:.2f}. {}/{}".format(time.time()-st, batch_idx,len(speech_dataloader)))
            try:
                noise_batch = next(noise_iter)
            except StopIteration:
                noise_iter = iter(noise_dataloader)
                noise_batch = next(noise_iter)

            mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
            actual_sisdr = calculate_sisdr(speech_batch.to(torch.device("cuda")), mix_batch)
            with torch.no_grad():
                est_batch = model(mix_batch).squeeze(1)
            est_sisdr = calculate_sisdr(speech_batch.to(torch.device("cuda")), est_batch)
            
            if batch_idx == 0 or batch_idx == 5:
                print(batch_idx)
                print("X SISDR: {}".format(make_np(actual_sisdr)))
                print("E SISDR: {}".format(make_np(est_sisdr)))
            if batch_idx == 5:        
                break     


    

def load_sample_weights(args, speech_dataloader, seed, load_dir, prev_weights=None):
    st = time.time()
    if prev_weights is not None:
        w = prev_weights
    else:
        if args.use_4p:
            w = torch.ones(28539*4) * 1./(28539*4)
        else:
            w = torch.ones(len(speech_dataloader) * args.batch_size) * 1./(len(speech_dataloader) * args.batch_size)
    alpha = 0.5
    shift=3
    phi = 0.2

    all_seed(seed)
    sisdr_t = torch.FloatTensor(np.load('{}/seed{}.npy'.format(load_dir, seed)))
    batch_idx = len(speech_dataloader) - 1
    
    ei = (batch_idx+1)*args.batch_size
    if args.use_4p:
        ei *= 4
    prev_sisdrs = sisdr_t[:ei]
    
    print("DEBUG: ", ei, len(w))
    p = copy.deepcopy(w[:ei])
    
    if args.use_r2:
        error_t = torch.sigmoid(shift-alpha*torch.Tensor(prev_sisdrs))
        L = torch.sum(error_t*p)
        beta = L/(1-L)
        print(L, beta)
        p = p * beta ** (1-error_t)
    elif args.use_cossim:
        eps = torch.sum((1-prev_sisdrs)*p)
        beta_t = 0.5 * torch.log((1-eps)/eps)
        print(eps, beta_t)
        p = p * torch.exp(-beta_t * prev_sisdrs)
    elif args.use_tanh:
        error_t = torch.tanh(shift-alpha*torch.Tensor(prev_sisdrs))
        p = p * torch.exp(error_t)
    else:
        error_t = 1 - torch.sigmoid(alpha * prev_sisdrs)
        above_thre_idx = error_t > phi
        under_thre_idx = error_t <= phi

        # eps_t = torch.sum(p[torch.abs(error_t - error_t.mean()) > error_t.std()/2])
        eps_t = torch.sum(p[above_thre_idx])
        beta_t = eps_t**2
        print(eps_t, beta_t)

        p[under_thre_idx] = (p[under_thre_idx] * beta_t)
    p = p/p.sum()
    
    return p


def run_se_ctn_seq(args, model, prev_model, speech_dataloader, noise_dataloader, optimizer):
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        with torch.no_grad():
            est_batch_temp = prev_model(mix_batch).squeeze(1)
        est_batch = model(est_batch_temp).squeeze(1)

        optimizer.zero_grad()

        speech_batch = speech_batch.to(torch.device("cuda"))
        actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
        loss_i = loss_sisdr(speech_batch, est_batch)#, actual_sisdr)
        total_loss[batch_idx] = float(loss_i)

        loss_i.backward()
        optimizer.step()
        
        del loss_i
        torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()


def run_se_ctn_allseq(args, models, speech_dataloader, noise_dataloader, optimizer, is_train=True):
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        
        est_batch = models[0](mix_batch).squeeze(1)
        for i,model in enumerate(models[1:]):
            est_batch = models[1+i](est_batch).squeeze(1)

        if is_train:
            optimizer.zero_grad()

        speech_batch = speech_batch.to(torch.device("cuda"))
        actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
        loss_i = loss_sisdr(speech_batch, est_batch)#, actual_sisdr)
        total_loss[batch_idx] = float(loss_i)

        if is_train:
            loss_i.backward()
            optimizer.step()
        
        del loss_i
        torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()

def get_te_impr_allseq(args, models, speech_dataloader, noise_dataloader, until_end=False, betas=None):
    total_loss = np.zeros(len(speech_dataloader))
    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        snr_db_batch = np.random.uniform(
            low=min(args.snr_ranges), high=max(args.snr_ranges), size=args.batch_size).astype(
            np.float32)

        s = speech_batch[0][0]
        n = noise_batch[0]
        slen = s.shape[-1]
        nlen = n.shape[-1]
        if slen > 160000:
            offset = np.random.randint(slen - 160000)
            s = s[:,offset:offset+160000]
            slen = s.shape[-1]
        while nlen <= slen:
            n = torch.cat(2*[n],1)
            nlen = n.shape[-1]
        offset = np.random.randint(nlen - slen)
        n = n[:,offset:offset+slen]
        s /= (s.std() + eps)
        n /= (n.std() + eps)

        x = mix_signals_batch(s, n, snr_db_batch)
        e = models[0](x.cuda()).squeeze(1)
        for i,model in enumerate(models[1:]):
            e = models[1+i](e).squeeze(1)
        actual_sisdr = calculate_sisdr(s,x)
        loss_i = loss_sisdr(s, e.detach().cpu(), actual_sisdr)
        total_loss[batch_idx] = float(loss_i)
        del loss_i
        torch.cuda.empty_cache()

        if ((batch_idx+1) % args.print_every) == 0:
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info
        if batch_idx == 50 and not until_end:
            break

    return total_loss[:batch_idx+1].mean()


def save_model_allseq(models, output_directory, rundata, is_last=False):
#     curr_epoch = rundata['epoch']
    if is_last:
        suffix = 'last'
    else:
        suffix = 'best'
    
    for i,model in enumerate(models):
        model_save_dir = "{}/Dmodel_{}_{}.pt".format(output_directory, i, suffix)
        data_save_dir = "{}/rundata_{}_{}.pt".format(output_directory, i, suffix)
        torch.save(model.state_dict(), model_save_dir)
        print("D model #{} saved to {}".format(model_save_dir, i))
        with open(data_save_dir, 'wb') as handle:
            pickle.dump(rundata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
def get_te_impr(args, model, speech_dataloader, noise_dataloader, until_end=False, betas=None):
    total_loss = np.zeros(len(speech_dataloader))
    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        snr_db_batch = np.random.uniform(
            low=min(args.snr_ranges), high=max(args.snr_ranges), size=args.batch_size).astype(
            np.float32)

        s = speech_batch[0][0]
        n = noise_batch[0]
        slen = s.shape[-1]
        nlen = n.shape[-1]
        if slen > 160000:
            offset = np.random.randint(slen - 160000)
            s = s[:,offset:offset+160000]
            slen = s.shape[-1]
        while nlen <= slen:
            n = torch.cat(2*[n],1)
            nlen = n.shape[-1]
        offset = np.random.randint(nlen - slen)
        n = n[:,offset:offset+slen]
        s /= (s.std() + eps)
        n /= (n.std() + eps)

        x = mix_signals_batch(s, n, snr_db_batch)
        if betas is not None:
            e = my_forward(model, x.cuda(), betas)
        else:
            e = model(x.cuda()).squeeze(1)
        actual_sisdr = calculate_sisdr(s,x)
        loss_i = loss_sisdr(s, e.detach().cpu(), actual_sisdr)
        total_loss[batch_idx] = float(loss_i)
        del loss_i
        torch.cuda.empty_cache()

        if ((batch_idx+1) % args.print_every) == 0:
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info
        if batch_idx == 50 and not until_end:
            break

    return total_loss[:batch_idx+1].mean()





def run_se_ctn_boost(args, model, p, speech_dataloader, noise_dataloader, optimizer, seed):
    all_seed(seed)
    total_loss = np.zeros(len(speech_dataloader))
    total_wloss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))

        optimizer.zero_grad()
        est_batch = model(mix_batch).squeeze(1)
        
        if args.use_4p:
            s = speech_batch.reshape(40,8000)
            x = mix_batch.reshape(40,8000)
            e = est_batch.reshape(40,8000)

            actual_sisdr = calculate_sisdr(s.to(torch.device("cuda")), x)
            loss = -1.*calculate_sisdr(s.to(torch.device("cuda")), e, actual_sisdr)
            w = p[batch_idx*args.batch_size*4:(batch_idx+1)*args.batch_size*4].to(torch.device("cuda")) 
            
#             print(batch_idx, actual_sisdr)
#             print(w)
            
        else:
            actual_sisdr = calculate_sisdr(speech_batch.to(torch.device("cuda")), mix_batch)
            loss = -1.*calculate_sisdr(speech_batch.to(torch.device("cuda")), est_batch)#, actual_sisdr)
            w = p[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]

        wloss = torch.sum(loss*w.to(torch.device("cuda"))) * float(28539)/args.batch_size
        wloss.backward()
        optimizer.step()
        total_loss[batch_idx] = float(loss.mean())
        total_wloss[batch_idx] = float(wloss)

        del wloss
        torch.cuda.empty_cache()

        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} wLoss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean(), total_wloss[:batch_idx+1].mean())) # logging.info

    return total_wloss[:batch_idx+1].mean()


def run_se_ctn_boost_debug(args, model, p, speech_dataloader, noise_dataloader, optimizer, seed):
    def make_np(x):
        return x.detach().cpu().numpy()

    all_seed(seed)
    total_loss = np.zeros(len(speech_dataloader))
    total_wloss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        actual_sisdr = calculate_sisdr(speech_batch.to(torch.device("cuda")), mix_batch)
        with torch.no_grad():
            est_batch = model.cuda()(mix_batch).squeeze(1)
        est_sisdr = calculate_sisdr(speech_batch.to(torch.device("cuda")), est_batch)
        impr = calculate_sisdr(speech_batch.to(torch.device("cuda")), est_batch, actual_sisdr)
        w = p[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]

        if batch_idx == 0 or batch_idx == 5:
            print(batch_idx)
            print("X SISDR: {}".format(make_np(actual_sisdr)))
            print("E SISDR: {}".format(make_np(est_sisdr)))
        if batch_idx == 5:        
            break
            

            
            
            
            
            
            
# COMB

from asteroid_filterbanks import make_enc_dec
from asteroid.masknn import TDConvNet
from asteroid.models.base_models import BaseModel
import warnings
from asteroid.masknn import norms
from asteroid.masknn.convolutional import *

def _unsqueeze_to_3d(x):
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x
    
def _shape_reconstructed(reconstructed, size):
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed

class BLOOMTDConvNet(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super(BLOOMTDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2 ** x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2 ** x
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2 ** x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net1 = nn.Sequential(nn.PReLU(), mask_conv)
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net2 = nn.Sequential(nn.PReLU(), mask_conv)
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net3 = nn.Sequential(nn.PReLU(), mask_conv)
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net4 = nn.Sequential(nn.PReLU(), mask_conv)
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net5 = nn.Sequential(nn.PReLU(), mask_conv)
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net6 = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        counter = 0
        for layer in self.TCN:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
            counter += 1
            # Use residual output when no skip connection
            mask_inp = skip_connection if self.skip_chan else output
            if counter == 1:
                score = self.mask_net1(mask_inp)
                score = score.view(batch, self.n_src, self.out_chan, n_frames)
                est_mask1 = self.output_act(score)
            elif counter == 2:
                score = self.mask_net2(mask_inp)
                score = score.view(batch, self.n_src, self.out_chan, n_frames)
                est_mask2 = self.output_act(score)
            elif counter == 3:
                score = self.mask_net3(mask_inp)
                score = score.view(batch, self.n_src, self.out_chan, n_frames)
                est_mask3 = self.output_act(score)
            elif counter == 4:
                score = self.mask_net4(mask_inp)
                score = score.view(batch, self.n_src, self.out_chan, n_frames)
                est_mask4 = self.output_act(score)
            elif counter == 5:
                score = self.mask_net5(mask_inp)
                score = score.view(batch, self.n_src, self.out_chan, n_frames)
                est_mask5 = self.output_act(score)
            elif counter == 6:
                score = self.mask_net6(mask_inp)
                score = score.view(batch, self.n_src, self.out_chan, n_frames)
                est_mask6 = self.output_act(score)
            
        return est_mask1, est_mask2, est_mask3, est_mask4, est_mask5, est_mask6

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "causal": self.causal,
        }
        return config
    
class BaseEncoderMaskerDecoder_comb(BaseModel):
    def __init__(self, encoder, masker, decoder1, decoder2, decoder3, decoder4, decoder5, decoder6, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.decoder3 = decoder3
        self.decoder4 = decoder4
        self.decoder5 = decoder5
        self.decoder6 = decoder6
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()

    def forward(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.enc_activation(self.encoder(wav))
        recons = []
        est_masks1, est_masks2, est_masks3, est_masks4, est_masks5, est_masks6 = self.masker(tf_rep)
        
        masked_tf_rep = est_masks1 * tf_rep.unsqueeze(1)
        decoded = self.decoder1(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        recons.append(_shape_reconstructed(reconstructed, shape).squeeze(1))
        
        masked_tf_rep = est_masks2 * tf_rep.unsqueeze(1)
        decoded = self.decoder2(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        recons.append(_shape_reconstructed(reconstructed, shape).squeeze(1))
        
        masked_tf_rep = est_masks3 * tf_rep.unsqueeze(1)
        decoded = self.decoder3(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        recons.append(_shape_reconstructed(reconstructed, shape).squeeze(1))
        
        masked_tf_rep = est_masks4 * tf_rep.unsqueeze(1)
        decoded = self.decoder4(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        recons.append(_shape_reconstructed(reconstructed, shape).squeeze(1))
        
        masked_tf_rep = est_masks5 * tf_rep.unsqueeze(1)
        decoded = self.decoder5(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        recons.append(_shape_reconstructed(reconstructed, shape).squeeze(1))
        
        masked_tf_rep = est_masks6 * tf_rep.unsqueeze(1)
        decoded = self.decoder6(masked_tf_rep)
        reconstructed = pad_x_to_y(decoded, wav)
        recons.append(_shape_reconstructed(reconstructed, shape).squeeze(1))
        return recons

    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        model_args = {**fb_config, **masknet_config, "encoder_activation": self.encoder_activation}
        return model_args
    
from asteroid.filterbanks import FreeFB, Encoder, Decoder
def BLOOM_make_enc_dec(fb_name,n_filters,kernel_size,stride=None,who_is_pinv=None,padding=0,output_padding=0,**kwargs):
    fb_class = FreeFB
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    enc = Encoder(fb, padding=padding)
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    dec1 = Decoder(fb, padding=padding, output_padding=output_padding)
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    dec2 = Decoder(fb, padding=padding, output_padding=output_padding)
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    dec3 = Decoder(fb, padding=padding, output_padding=output_padding)
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    dec4 = Decoder(fb, padding=padding, output_padding=output_padding)
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    dec5 = Decoder(fb, padding=padding, output_padding=output_padding)
    fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
    dec6 = Decoder(fb, padding=padding, output_padding=output_padding)
    return enc, dec1, dec2, dec3, dec4, dec5, dec6
    
class BLOOMNETFT2(BaseEncoderMaskerDecoder_comb):
    def __init__(
        self, n_src, out_chan=None, n_blocks=1, n_repeats=6, bn_chan=128, hid_chan=512, skip_chan=False, conv_kernel_size=3, 
        norm_type="gLN", mask_act="sigmoid", in_chan=None, causal=False, fb_name="free", kernel_size=16, n_filters=512, 
        stride=8, encoder_activation=None, sample_rate=8000, **fb_kwargs,
    ):
        encoder, decoder1, decoder2, decoder3, decoder4, decoder5, decoder6 = BLOOM_make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
#         my_decoder = 
        n_feats = encoder.n_feats_out

        # Update in_chan
        masker = BLOOMTDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )
        
        
        super().__init__(encoder, masker, decoder1, decoder2, decoder3, decoder4, decoder5, decoder6, encoder_activation=encoder_activation)
            
def edit_dict(ori_dict, idx):
    new_dict = copy.deepcopy(ori_dict)
    for k,v in ori_dict.items():
        if 'encoder' in k or 'bottleneck' in k:
            del new_dict[k]
        for j in range(idx):
            if 'TCN.{}'.format(j) in k:
                del new_dict[k]
                
    new_dict['masker.mask_net{}.0.weight'.format(idx+1)] = new_dict['masker.mask_net.0.weight']
    del new_dict['masker.mask_net.0.weight']
    new_dict['masker.mask_net{}.1.weight'.format(idx+1)] = new_dict['masker.mask_net.1.weight']
    del new_dict['masker.mask_net.1.weight']
    new_dict['masker.mask_net{}.1.bias'.format(idx+1)] = new_dict['masker.mask_net.1.bias']
    del new_dict['masker.mask_net.1.bias']
    new_dict['decoder{}.filterbank._filters'.format(idx+1)] = new_dict['decoder.filterbank._filters']
    del new_dict['decoder.filterbank._filters']

    return new_dict




def run_se_ctn_comb(args, model, speech_dataloader, noise_dataloader, optimizer=None, is_train=True):
    total_loss = np.zeros(len(speech_dataloader))

    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        mix_batch = mix_signals_batch(speech_batch, noise_batch, args.snr_ranges).to(torch.device("cuda"))
        est_batches = model(mix_batch)
        speech_batch = speech_batch.to(torch.device("cuda"))
        actual_sisdr = calculate_sisdr(speech_batch, mix_batch)
        total_loss[batch_idx] = 0
        losses = []
        tot_loss = 0
        for i in range(6):
            if is_train:
                optimizer.zero_grad()
            loss_i = loss_sisdr(speech_batch, est_batches[i])#, actual_sisdr)
#             if batch_idx == 0:
#                 print(batch_idx, calculate_sisdr(speech_batch, est_batches[i]))
            total_loss[batch_idx] += float(loss_i)
            tot_loss += loss_i
        if is_train:
            tot_loss.backward()        
            optimizer.step()
        del tot_loss
        torch.cuda.empty_cache()
        total_loss[batch_idx] /= 6
        if (batch_idx % args.print_every) == 0: 
            print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
                    batch_idx, total_loss[:batch_idx+1].mean())) # logging.info

    return total_loss[:batch_idx+1].mean()

def get_te_impr_comb(args, model, speech_dataloader, noise_dataloader, until_end=False):
    total_loss = np.zeros((6,len(speech_dataloader)))
    noise_iter = iter(noise_dataloader)
    for batch_idx, speech_batch in enumerate(speech_dataloader):
        try:
            noise_batch = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_dataloader)
            noise_batch = next(noise_iter)

        snr_db_batch = np.random.uniform(
            low=min(args.snr_ranges), high=max(args.snr_ranges), size=args.batch_size).astype(
            np.float32)

        s = speech_batch[0][0]
        n = noise_batch[0]
        slen = s.shape[-1]
        nlen = n.shape[-1]
        if slen > 160000:
            offset = np.random.randint(slen - 160000)
            s = s[:,offset:offset+160000]
            slen = s.shape[-1]
        while nlen <= slen:
            n = torch.cat(2*[n],1)
            nlen = n.shape[-1]
        offset = np.random.randint(nlen - slen)
        n = n[:,offset:offset+slen]
        s /= (s.std() + eps)
        n /= (n.std() + eps)

        x = mix_signals_batch(s, n, snr_db_batch)
        actual_sisdr = calculate_sisdr(s,x)
        e = model(x.cuda())
        for i in range(6):
            loss_i = loss_sisdr(s, e[i].detach().cpu(), actual_sisdr)
            total_loss[i][batch_idx] = float(loss_i)
            del loss_i
            torch.cuda.empty_cache()

            if ((batch_idx+1) % args.print_every) == 0:
                print ("Batch {}. i: {}. Loss: {:.3f} SI-SDRi".format(
                        batch_idx, i, total_loss[i][:batch_idx+1].mean())) # logging.info
        if batch_idx == 50 and not until_end:
            break

    return total_loss[:][:batch_idx+1]