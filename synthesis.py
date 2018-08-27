# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --symmetric-mels                  Symmetric mel.
    --max-abs-value=<N>               Max abs value [default: -1].
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import random
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams

from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, initial_value=None,
            fast=False, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    model.eval()
    if fast:
        model.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
            assert c.ndim == 2
        Tc = c.shape[0]
        upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not hparams.upsample_conditional_features:
            c = np.repeat(c, upsample_factor, axis=0)

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

    if initial_value is None:
        if is_mulaw_quantize(hparams.input_type):
            initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
        else:
            initial_value = 0.0

    if is_mulaw_quantize(hparams.input_type):
        assert initial_value >= 0 and initial_value < hparams.quantize_channels
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    initial_input = initial_input.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        # Batching Experiment
        Ti = 148
        initial_input= initial_input.repeat(Ti, 1, 1)
        print("c.size() : {}".format(str(c.size())))
        c = list(torch.chunk(c, Ti, dim=2))
        print("len(c) : {}".format(len(c)))
        for i, x in enumerate(c) :
            temp = torch.zeros_like(c[0])
            temp[:,:,:x.size(2)] = x[:,:,:]
            c[i] = temp
        c = torch.cat(c, dim=0)
        g = g.repeat(Ti)
        print("length : {}".format(length))
        length = c.size(-1) * audio.get_hop_size()
        print("c.size() : {}".format(str(c.size())))
        print("length : {}".format(length))
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat


def wavegen_fast(model, length=None, c=None, g=None, initial_value=None,
            fast=False, tqdm=tqdm, Ti=4):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    model.eval()
    if fast:
        model.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
            assert c.ndim == 2
        Tc = c.shape[0]
        upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not hparams.upsample_conditional_features:
            c = np.repeat(c, upsample_factor, axis=0)

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

    if initial_value is None:
        if is_mulaw_quantize(hparams.input_type):
            initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
        else:
            initial_value = 0.0

    if is_mulaw_quantize(hparams.input_type):
        assert initial_value >= 0 and initial_value < hparams.quantize_channels
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    initial_input = initial_input.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)
    c_non_upsampled = c
    with torch.no_grad():
        # Batching Experiment
        initial_input_first = initial_input.repeat(Ti, 1, 1)
        
        # Upsample local condition first
        # B x 1 x C x T
        c = c.unsqueeze(1)
        for f in model.upsample_conv:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)
        assert c.size(-1) == length

        
        c_first = list(torch.chunk(c, Ti, dim=2))
        for i, x in enumerate(c_first) :
            temp = torch.zeros_like(c_first[0])
            temp[:,:,:x.size(2)] = x[:,:,:]
            c_first[i] = temp
        c_first = torch.cat(c_first, dim=0)
        g_first = g.repeat(Ti)
        length_first = c_first.size(-1)
        y_hat, buffers = model.incremental_forward(
            initial_input_first,
            c=c_first,
            g=g_first,
            T=length_first,
            return_buffer=True,
            skip_upsample=True,
            tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

        if True : 
            def push_buffers(buffers) :
                if buffers is None :
                    return
                if type(buffers) is list :
                    new_buffers = []
                    for b in buffers :
                        new_buffers.append(push_buffers(b))
                    return new_buffers
                else :
                    b = buffers
                    z = torch.zeros_like(b)
                    z[1:,:,:] = b[:-1,:,:]
                    return z
                
            # Push buffers into next segment
            buffers = push_buffers(buffers)
                
            # Push initial inputs into next segment
            initial_input_first[1:,:,:] = y_hat[:-1,:,-1:]
                
            def selector(t, prev) :
                if t == 0 :
                    return initial_input_first
                elif t < length_first * 0.8:
                    return prev[-1]
                else :
                    if random.random() < 1 :
                        return y_hat[:, :, t-1].unsqueeze(2)
                    else :
                        return prev[-1]
                
            y_hat = model.incremental_forward(
                initial_input_first,
                c=c_first,
                g=g_first,
                T=length_first,
                buffers=buffers,
                return_buffer=False,
                skip_upsample=True,
                tqdm=tqdm, softmax=True, quantize=True,
                input_selector=None,
                log_scale_min=hparams.log_scale_min)
    
        if False:   
            y_hat = y_hat.view(1, 1, -1)[:,:,:length]
            mol = model.forward(
                y_hat,
                c = c_non_upsampled,
                g = g)
            y_hat = sample_from_discretized_mix_logistic(mol, log_scale_min=hparams.log_scale_min)
    
    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()
    
    return y_hat
    
if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]

    length = int(args["--length"])
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)
    conditional_path = args["--conditional"]
    # From https://github.com/Rayhane-mamah/Tacotron-2
    symmetric_mels = args["--symmetric-mels"]
    max_abs_value = float(args["--max-abs-value"])

    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]
    speaker_id = args["--speaker-id"]
    speaker_id = None if speaker_id is None else int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Load conditional features
    if conditional_path is not None:
        c = np.load(conditional_path)
        if c.shape[1] != hparams.num_mels:
            np.swapaxes(c, 0, 1)
        if max_abs_value > 0:
            min_, max_ = 0, max_abs_value
            if symmetric_mels:
                min_ = -max_
            print("Normalize features to desired range [0, 1] from [{}, {}]".format(min_, max_))
            c = np.interp(c, (min_, max_), (0, 1))
    else:
        c = None

    from train import build_model

    # Model
    model = build_model().to(device)

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    os.makedirs(dst_dir, exist_ok=True)
    

    if True :
        dst_wav_path = join(dst_dir, "{}{}.wav".format(checkpoint_name, file_name_suffix))
        # Faster Generation
        waveform = wavegen_fast(model, length, c=c, g=speaker_id, initial_value=initial_value, fast=True, Ti=16)
        
        # save
        librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)
        print("Finished! Check out {} for generated audio samples.".format(dst_dir))
        sys.exit(0)
    elif False :
        # Stochastic Model Experiment
        # 기본 합성 알고리즘으로 동일한 2개의 음성을 합성합니다. 모델이 stochatic하기 때문에 합성된 음성은 100% 동일하지 않을 것입니다.
        # 이 두 음성을 Faster Synthesis 알고리즘의 배율과 동일한 개수로 쪼개어, 서로 엮습니다.
        mult = 16
        waveform1 = wavegen_fast(model, length, c=c, g=speaker_id, initial_value=initial_value, fast=True, Ti=mult)
        waveform2 = wavegen_fast(model, length, c=c, g=speaker_id, initial_value=initial_value, fast=True, Ti=mult)
        Ti = 148
        segments1 = np.array_split(waveform1, Ti)
        segments2 = np.array_split(waveform2, Ti)
        woven = []
        for i in range(len(segments1)) :
            if i % 2 == 0 :
                woven.append(segments1[i])
            else :
                woven.append(segments2[i])
        waveform = np.concatenate(woven)
        dst_wav_path = join(dst_dir, "wave1-x{}-w{}.wav".format(mult, Ti))
        librosa.output.write_wav(dst_wav_path, waveform1, sr=hparams.sample_rate)
        dst_wav_path = join(dst_dir, "wave2-x{}-w{}.wav".format(mult, Ti))
        librosa.output.write_wav(dst_wav_path, waveform2, sr=hparams.sample_rate)
        dst_wav_path = join(dst_dir, "woven-x{}-w{}.wav".format(mult, Ti))
        librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)