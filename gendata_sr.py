from scipy.io.wavfile import read, write
import torchaudio
import torch
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import os
import soundfile as sf
import matplotlib.pyplot as plt

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


ac_wav_path = "/blob/v-yuancwang/audio_editing_data/audiocaps_refine/wav"
as_wav_path = "/blob/v-yuancwang/audio_editing_data/audioset96_refine/wav"
save_path = "/blob/v-yuancwang/audio_editing_data/sr"
ac_files = set(os.listdir(ac_wav_path))
as_files = os.listdir(as_wav_path)
wav_files = []
for file in as_files:
    if file not in ac_files:
        wav_files.append(file)
print(len(ac_files))
print(len(as_files))
print(len(wav_files))
wav_files = sorted(wav_files)

for file_name in tqdm(wav_files[:]):
    wav, sr = librosa.load(os.path.join(as_wav_path, file_name), sr=8000)
    wav = np.clip(wav, -1, 1)
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype('int16')
    write(os.path.join(save_path, "wav", file_name), 8000, wav)

for file_name in tqdm(wav_files[:]):
    wav, sr = librosa.load(os.path.join(save_path, "wav", file_name), sr=16000)
    x = torch.FloatTensor(wav)
    # print(len(x))
    x = mel_spectrogram(x.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=16000,
                    hop_size=256, win_size=1024, fmin=0, fmax=8000)
    # print(x.shape)
    spec = x.cpu().numpy()[0]
    # print(spec.shape)
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype('int16')
    write(os.path.join(save_path, "wav", file_name), 16000, wav)
    np.save(os.path.join(save_path, "mel", file_name.replace(".wav", ".npy")), spec) 