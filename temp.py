import os
from tqdm import tqdm
import numpy as np
import shutil

gen_path = "/blob/v-yuancwang/audio_editing_test/sr/200000/wav/wav"
save_path = "/blob/v-yuancwang/audio_editing_test/random/wav"

gen_wavs = os.listdir(gen_path)
wav_dict = {}
for wav in gen_wavs:
    wav_name, _ = wav.split("_sample_")
    if wav_name not in wav_dict:
        wav_dict[wav_name] = []
    wav_dict[wav_name].append(wav)

for wav in tqdm(wav_dict):
    wav_name = np.random.choice(wav_dict[wav])
    wav_name = os.path.join(gen_path, wav_name)
    shutil.copyfile(wav_name, os.path.join(save_path, wav+".wav"))
