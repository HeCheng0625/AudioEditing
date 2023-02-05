import pandas as pd
import os

as96_path = "/blob/v-yuancwang/audio_editing_data/audioset96/wav"
as96_wavs = os.listdir(as96_path)
as96_wavs = set(as96_wavs)
print(len(as96_wavs))

txt_path = "/home/v-yuancwang/AudioEditing/metadatas/audioset96_file_label.txt"
with open(txt_path, "w") as f:
    path2 = "/blob/v-yuancwang/speech/audioset/audioset_data"
    labels = os.listdir(path2)
    print(len(labels))
    label_dict = {}
    for label in labels:
        label_dict[label] = 0
        wavs = os.listdir(os.path.join(path2, label))
        for wav in wavs:
            if wav in as96_wavs:
                label_dict[label] += 1
                f.write(wav + "   " + label + "\n")
# print(label_dict)
