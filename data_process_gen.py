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
import json

MAX_WAV_VALUE = 32768.0
WAV_LENGTH = 16000 * 10 - 128

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


audioset_96_path = "/blob/v-yuancwang/speech/audioset/audioset_data"
as96_labels = os.listdir(audioset_96_path)

save_path = "/blob/v-yuancwang/audio_editing_data/audioset96"

for label in tqdm(as96_labels[64:]):
    file_dir = os.path.join(audioset_96_path, label) 
    for wav_file in tqdm(os.listdir(file_dir)[:]):
        wav, sr = librosa.load(os.path.join(file_dir, wav_file), sr=16000)
        if len(wav) < WAV_LENGTH:
            wav = np.pad(wav, (0, WAV_LENGTH - len(wav)), 'constant', constant_values=(0, 0))
        wav = wav[: WAV_LENGTH]
        x = torch.FloatTensor(wav)
        # print(len(x))
        x = mel_spectrogram(x.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=16000,
                        hop_size=256, win_size=1024, fmin=0, fmax=8000)
        # print(x.shape)
        spec = x.cpu().numpy()[0]
        # print(spec.shape)
        wav = wav * MAX_WAV_VALUE
        wav = wav.astype('int16')
        write(os.path.join(save_path, "wav", wav_file), 16000, wav)
        np.save(os.path.join(save_path, "mel", wav_file.replace(".wav", ".npy")), spec)

# dict = json.load(open("/home/v-yuancwang/AudioEditing/metadatas/audioset_ontology.json", "r"))
# dict = {d['id']: d['name'] for d in dict}
# print(dict)