# import numpy as np
# import os
# with open("/home/v-yuancwang/AudioEditing/metadatas/audiocaps_train_metadata.jsonl", "r") as f:
#     lines = f.readlines()
# lines = [eval(line) for line in lines]
# ac_path = "/blob/v-yuancwang/audio_editing_data/audiocaps/mel"
# as_path = "/blob/v-yuancwang/audio_editing_data/audioset96/mel"
# ac_set = set(os.listdir(ac_path))
# as_set = set(os.listdir(as_path))
# save_path = "/blob/v-yuancwang/audio_editing_data/sr/mel"

# ac_refine_path = "/blob/v-yuancwang/audio_editing_data/audiocaps_refine/mel"
# as_refine_path = "/blob/v-yuancwang/audio_editing_data/audioset96_refine/mel"
# ac_refine_set = set(os.listdir(ac_refine_path))
# as_refine_set = set(os.listdir(as_refine_path))

# with open("/home/v-yuancwang/AudioEditing/metadatas/audioset96_file_label.txt", "r") as f:
#     as_lines = f.readlines()
# as_lines = [{"file_name": line.replace("\n", "").split("   ")[0], "text": line.replace("\n", "").split("   ")[1].replace("_", " ", 5)} for line in as_lines]
# np.random.shuffle(as_lines)
# # print(as_lines[: 100])

# with open("/home/v-yuancwang/AudioEditing/metadata_infos/inpainting_refine.txt", "w") as f:
#     for line in lines:
#         wav_name = line["file_name"].replace(".wav", ".npy")
#         text = line["text"].replace("\n", "")
#         if wav_name not in ac_set:
#             continue
#         text = np.random.choice(["Inpainting:", "inpainting: ", "Inpainting: ", "Inpainting: "]) + text
#         if wav_name in ac_refine_set:
#             f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(ac_refine_path, wav_name) + "   " + text + "\n")
#         else:
#             f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(ac_path, wav_name) + "   " + text + "\n")
#     for line in as_lines[:150000]:
#         wav_name = line["file_name"].replace(".wav", ".npy")
#         text = line["text"].replace("\n", "")
#         if wav_name not in as_set:
#             continue
#         text = np.random.choice(["Inpainting:", "inpainting: ", "Inpainting: ", "Inpainting: "]) + text
#         if wav_name in as_refine_set:
#             f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(as_refine_path, wav_name) + "   " + text + "\n")
#         else:
#             f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(as_path, wav_name) + "   " + text + "\n")


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

import pandas as pd
fsd50k_root_data = []
with open("/home/v-yuancwang/AudioEditing/metadatas/fsd50k_short.txt","r") as f:
    lines = f.readlines()
for line in lines:
    fsd50k_root_data.append({"file_name": line.split("   ")[0], "text": line.split("   ")[1].replace("\n", "")})
print(len(fsd50k_root_data))
esc50_root_data = []
esc50_df = pd.read_csv("/home/v-yuancwang/AudioEditing/metadatas/esc50.csv")
esc50_df =  esc50_df[["filename", "category"]].to_numpy().tolist()
for file_name, text in esc50_df:
    esc50_root_data.append({"file_name": file_name, "text": text})
print(len(esc50_root_data))

fsd50k_path = "/blob/v-yuancwang/audio_editing_data/fsd50k"
fsd50k_refine_path = "/blob/v-yuancwang/audio_editing_data/fsd50k_refine/wav"
fsd50k_refine_set = set(os.listdir(fsd50k_refine_path))
print(len(fsd50k_refine_set))
esc50_path = "/blob/v-yuancwang/audio_editing_data/esc50"
audioset_root_data = []
audioset96_path = "/blob/v-yuancwang/audio_editing_data/audioset96"
audioset96_refine_path = "/blob/v-yuancwang/audio_editing_data/audioset96_refine/wav"
audioset96_refine_set = set(os.listdir(audioset96_refine_path))
with open("/home/v-yuancwang/AudioEditing/metadatas/audioset96_file_label.txt","r") as f:
    lines = f.readlines()
for line in lines:
    audioset_root_data.append({"file_name": line.split("   ")[0], "text": line.split("   ")[1].replace("\n", "")})
print(len(audioset_root_data))
print(audioset_root_data[:2])

save_path = "/blob/v-yuancwang/audio_editing_data/replacement_refine"

metadatas = []

for i in tqdm(range(25000, 50000)):
    id1 = np.random.randint(0, len(audioset_root_data))
    file_name1, text1 = audioset_root_data[id1]['file_name'], audioset_root_data[id1]['text']
    if file_name1 in audioset96_refine_set:
        y1, sr = librosa.load(os.path.join(audioset96_refine_path, file_name1), sr=16000)
    else:
        y1, sr = librosa.load(os.path.join(audioset96_path, "wav", file_name1), sr=16000)
    # print(file_name1, text1)

    id2 = np.random.randint(0, len(fsd50k_root_data) + len(esc50_root_data))
    if id2 >= len(fsd50k_root_data):
        file_name2, text2 = esc50_root_data[id2%len(fsd50k_root_data)]['file_name'], esc50_root_data[id2%len(fsd50k_root_data)]['text']
        y2, sr = librosa.load(os.path.join(esc50_path, "wav", file_name2), sr=16000)
    else:
        file_name2, text2 = fsd50k_root_data[id2]['file_name'], fsd50k_root_data[id2]['text']
        if file_name2 in fsd50k_refine_set:
            y2, sr = librosa.load(os.path.join(fsd50k_refine_path, file_name2), sr=16000)
        else:
            y2, sr = librosa.load(os.path.join(fsd50k_path, "wav", file_name2), sr=16000)
    # print(file_name2, text2)

    file_name3 = file_name2
    while(file_name3 == file_name2):
        id3 = np.random.randint(0, len(fsd50k_root_data) + len(esc50_root_data))
        if id3 >= len(fsd50k_root_data):
            file_name3, text3 = esc50_root_data[id3%len(fsd50k_root_data)]['file_name'], esc50_root_data[id3%len(fsd50k_root_data)]['text']
            y3, sr = librosa.load(os.path.join(esc50_path, "wav", file_name3), sr=16000)
        else:
            file_name3, text3 = fsd50k_root_data[id3]['file_name'], fsd50k_root_data[id3]['text']
            if file_name3 in fsd50k_refine_set:
                y3, sr = librosa.load(os.path.join(fsd50k_refine_path, file_name3), sr=16000)
            else:
                y3, sr = librosa.load(os.path.join(fsd50k_path, "wav", file_name3), sr=16000)
        # print(file_name3, text3)

    start = np.random.randint(int(len(y1) * 0.2), int(len(y1) * 0.65))

    y2 = np.pad(y2, (start, max(0, len(y1) - len(y2) - start)), 'constant', constant_values=(0, 0))
    y2 = y2[: len(y1)]
    y2 = y1 + y2
    y2 = np.clip(y2, -1, 1)

    x2 = torch.FloatTensor(y2)
    x2 = mel_spectrogram(x2.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=16000,
                    hop_size=256, win_size=1024, fmin=0, fmax=8000)
    spec2 = x2.cpu().numpy()[0]
    # print(spec3.shape)

    y2 = y2 * MAX_WAV_VALUE
    y2 = y2.astype('int16')
    
    y3 = np.pad(y3, (start, max(0, len(y1) - len(y3) - start)), 'constant', constant_values=(0, 0))
    y3 = y3[: len(y1)]
    y3 = y1 + y3
    y3 = np.clip(y3, -1, 1)

    x3 = torch.FloatTensor(y3)
    x3 = mel_spectrogram(x3.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=16000,
                    hop_size=256, win_size=1024, fmin=0, fmax=8000)
    spec3 = x3.cpu().numpy()[0]
    # print(spec3.shape)

    y3 = y3 * MAX_WAV_VALUE
    y3 = y3.astype('int16')

    wav_name2 = 'replacement'  + 'gen' + str(i) + '_0' + '.wav'
    spec_name2 = 'replacement'  + 'gen' + str(i) + '_0' + '.npy'
    wav_name3 = 'replacement'  + 'gen' + str(i) + '_1' + '.wav'
    spec_name3 = 'replacement'  + 'gen' + str(i) + '_1' + '.npy'

    write(os.path.join(save_path, "wav", wav_name2), 16000, y2)
    np.save(os.path.join(save_path, "mel", spec_name2), spec2)
    write(os.path.join(save_path, "wav", wav_name3), 16000, y3)
    np.save(os.path.join(save_path, "mel", spec_name3), spec3)

    metadatas.append((wav_name2, wav_name3, text2, text3, text1))

with open("/home/v-yuancwang/AudioEditing/metadatas/audioset_fsdesc_replacement_refine_1.txt", "w") as f:
    for wav_name2, wav_name3, text2, text3, text1 in metadatas:
        f.write(wav_name2 + "   " + wav_name3 + "   " + text2 + "   " + text3 + "   " + text1 + "\n")