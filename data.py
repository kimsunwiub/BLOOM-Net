import torchaudio
import torch.nn as nn
import torch
import torch.utils.data as data
import numpy as np
import os

from utils import all_seed
   
eps = 1e-6

# Data
class musan_train_prep_dataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir):
        self.ds_dir = ds_dir
        self.ds_files = [x for x in os.listdir(ds_dir) if "wav" in x]

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, idx):
        # Select sample
        filepath = self.ds_files[idx]
        filename = "{}/{}".format(self.ds_dir, filepath)
        waveform, _ = torchaudio.load(filename)
        return waveform

def signal_preprocessing(waveform, n_frames, dtype):
    if dtype=='speech':
        if waveform.shape[1] <= n_frames:
            diff = n_frames - waveform.shape[1] + 1
            waveform = torch.nn.functional.pad(waveform, (0,diff))
    else:
        while waveform.shape[1] <= n_frames:
            waveform = torch.cat(2*[waveform],1)
    offset = np.random.randint(waveform.shape[1] - n_frames)
    waveform = waveform[:, offset:n_frames + offset]
    waveform = waveform / (waveform.std() + eps)
    return waveform

def data_processing(data, n_frames, dtype="speech", spkr_id_exempt=-1):
    list_waveforms = torch.zeros(len(data), n_frames)
    for idx, elem in enumerate(data):
        if dtype=="speech":
            waveform, sr, uttr, spkr_id, chpt_id, uttr_id = elem
            if spkr_id == spkr_id_exempt:
                continue
            # print(spkr_id, chpt_id, uttr_id)
        else:
            waveform = elem
        waveform = signal_preprocessing(waveform, n_frames, dtype)
        list_waveforms[idx] = waveform
    return list_waveforms

class boost_weights_dataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir):
        self.ds_dir = ds_dir
        self.ds_files = [x for x in os.listdir(ds_dir) if "npy" in x]

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, idx):
        # Select sample
        filepath = self.ds_files[idx]
        filename = "{}/{}".format(self.ds_dir, filepath)
        w = np.load(filename)
        return w

def init_pers_set(args):
    ### Data
    eps = args.eps
    all_seed(args.seed)
    session_id = args.seed

    te_spkr_dir = '/home/kimsunw/data/LibriSpeech/test-clean'
#     spkr_ids = [x for x in os.listdir(te_spkr_dir) if 'txt' not in x]
    spkr_ids = [
        '2094',
        '4970',
        '1320',
        '4992',
        '3575',
        '237',
        '7729',
        '1221',
        '4507',
        '8463',
        '1188',
        '5639',
        '4077',
        '7127',
        '4446',
        '2300',
        '3570',
        '61',
        '8224',
        '8230',
        '8455',
        '2830',
        '5683',
        '6829',
        '121',
        '260',
        '5105',
        '672',
        '908',
        '5142',
        '1580',
        '1284',
        '6930',
        '7021',
        '7176',
        '3729',
        '8555',
        '1995',
        '1089',
        '2961']
    session_spkr = spkr_ids[session_id]
    te_spkr_id = "{}/{}".format(te_spkr_dir, session_spkr)
    spkr_utts = []
    for r,d,f in os.walk(te_spkr_id):
        for file in f:
            if 'flac' in file:
                spkr_utts.append(os.path.join(r,file))
    spkr_utts = np.array(spkr_utts)

    # Select noise for the session
    noise_dir = '/home/kimsunw/data/formatted_wham/'
#     all_noise_batch_list = [x for x in os.listdir(noise_dir) if '.sh' not in x]
    all_noise_batch_list = [
        'Tomatina',
        'CalafiaTaqueria',
        'CLinda',
        'DogBrew',
        'LinkedInLobby',
        'LakeshoreCafe',
        'Marsbar',
        'TheLocal',
        'Lucky13',
        'TheBeanery',
        'DoloresParkCafe',
        'TheMarket',
        'StarbucksOakland',
        'WesCafeAlameda',
        'TheGroveonMission',
        'Chipotle',
        'StarbucksParkSt',
        'inandoutalameda',
        'PeetsCoffee-NHFAlameda',
        'WholeFoodsCafe',
        'A15Linda',
        'Lapanca',
        'GoldCaneCocktailLounge',
        'MakeWesting',
        'LuckyT1North(17)',
        'StarbucksShothShore',
        'Luckyeast',
        'MCollins',
        'Almanac',
        'AtlasCafe',
        'Tertuliagalleryandcoffee',
        'VanKleef',
        'HC',
        'Peets-NHF',
        'Sidestreetpho',
        'Anotherpeetsalameda',
        'CoffeeBar',
        'StreetTaco',
        'StarbucksBridgeside',
        'FourBarrelsCoffee']

    noise_cls = all_noise_batch_list[session_id]
    noise_sigs = os.listdir("{}/{}".format(noise_dir, noise_cls))
    noise_sigs = np.array(["{}/{}/{}".format(noise_dir, noise_cls, x) for x in noise_sigs])
    
    tot_s = []
    for batch_idx in range(len(spkr_utts)):
        speech_w = spkr_utts[batch_idx]
        speech_w, sr = torchaudio.load(speech_w)
        assert sr == 16000
        tot_s.append(speech_w)
        
    tot_n = []
    for batch_idx in range(len(noise_sigs)):
        noise_w = noise_sigs[batch_idx]
        noise_w, sr = torchaudio.load(noise_w)
        assert sr == 16000
        assert len(noise_w) == 2
        tot_n.append(noise_w[0][None,:])

    print ("Session Spkr: ", session_spkr)
    print ("Noise Class: ", noise_cls)
    
    return tot_s, tot_n

def diviup(args, x, x_type):
    if x_type == 'noise':
        while x.shape[1] <= args.n_frames:
            x = torch.cat(2*[x],1)
    elif x_type == 'speech':
        # Pad speeches less than 2s
        if x.shape[1] <= args.n_frames:
            diff = args.n_frames - x.shape[1] + 1
            x = torch.nn.functional.pad(x, (0,diff))

    n_chunks = x.shape[1]//args.n_frames
    tot_frames = n_chunks*args.n_frames
    s_i = np.random.randint(x.shape[1]-tot_frames+1)
    tot_segs = []
    for i in range(n_chunks):
        seg = x[:,s_i:s_i + args.n_frames]
        if x_type == 'noise':
            seg = standardize(seg)
        tot_segs.append(seg)
        s_i = s_i + args.n_frames
    return tot_segs

def standardize(seg):
    if seg.std() > 1e-3:
        seg = seg / (seg.std() + eps)
    return seg

def shuffle_set(x):
    r = torch.randperm(len(x))
    return x[r]

def mixup(args, tot_s, tot_n):
    all_seed(args.seed)

    # Speech
    tot_s_segs = torch.cat([torch.stack(diviup(args, tot_s[i], 'speech')).squeeze(1) for i in range(len(tot_s))])

    slen = len(tot_s_segs)
    tr_s = tot_s_segs[:-120]
    va_s = tot_s_segs[-120:-60]
    te_s = tot_s_segs[-60:]
    te_s = te_s.reshape(len(te_s)//10, -1)

    # Noise
    lens = []
    for n  in tot_n:
        lens.append(len(n[0])/16000)
    lens = np.array(lens)

    te_n = []
    te_idxs = np.argsort(lens)[-6:]
    for idx in te_idxs:
        sig = tot_n[idx]
        tot_len = 16000*10
        sig_i = np.random.randint(sig.shape[1]-tot_len+1)
        te_n.append(sig[:,sig_i:sig_i+tot_len])
    te_n = torch.cat(te_n)
    for idx in te_idxs:
        del tot_n[idx]

    tot_n_segs = torch.cat([torch.stack(diviup(args, tot_n[i], 'noise')).squeeze(1) for i in range(len(tot_n))])
    n_i = shuffle_set(tot_n_segs)

    nsigs = len(va_s)
    tr_n = n_i[:-nsigs]
    va_n = n_i[-nsigs:]

    # Standardize
    tr_s = tr_s/(tr_s.std(1)[:,None] + eps)
    va_s = va_s/(va_s.std(1)[:,None] + eps)
    te_s = te_s/(te_s.std(1)[:,None] + eps)
    te_n = te_n/(te_n.std(1)[:,None] + eps)
        
    return tr_s, tr_n, va_s, va_n, te_s, te_n

def mix_signals_batch(s, n, snr_ranges):
    """
    Checked.
    """
    n = scale_amplitude(n, snr_ranges)
    x = s + n
    
    # Standardize
    x = x/(x.std(1)[:,None] + eps)
    return x

def prep_sig_ml(s,sr):
    """
    Checked. 
    """
    ml=np.minimum(s.shape[1], sr.shape[1])
    s=s[:,:ml]
    sr=sr[:,:ml]
    return ml, s, sr

def get_mixing_snr(x, snr_ranges):
    snr_batch = np.random.uniform(
        low=min(snr_ranges), high=max(snr_ranges), size=len(x)).astype(np.float32)
    return torch.Tensor(snr_batch).to(x.device)

def scale_amplitude(x, snr_ranges):
    """
    Scale signal x by values within snr_ranges
    
    e: est_batch. Located on GPU.
    g: speech_batch/noise_batch. 
    """    
    # Compute mixing SNR
    snr_batch = get_mixing_snr(x, snr_ranges)
    x = x * (10 ** (-snr_batch / 20.))[:,None]
    return x

def apply_scale_invariance(s, x):
    """
    Checked. 
    """
    alpha = s.mm(x.T).diag()
    alpha /= ((s ** 2).sum(dim=1) + eps)
    alpha = alpha.unsqueeze(1)
    s = s * alpha
    return s