import numpy as np
import os
with open("/home/v-yuancwang/AudioEditing/metadatas/audiocaps_train_metadata.jsonl", "r") as f:
    lines = f.readlines()
lines = [eval(line) for line in lines]
ac_path = "/blob/v-yuancwang/audio_editing_data/audiocaps"
as_path = "/blob/v-yuancwang/audio_editing_data/audioset96"
zero_path = "/blob/v-yuancwang/audio_editing_data/zeros.npy"
with open("/home/v-yuancwang/AudioEditing/metadatas/audioset96_file_label.txt", "r") as f:
    as_lines = f.readlines()
as_lines = [{"file_name": line.replace("\n", "").split("   ")[0], "text": line.replace("\n", "").split("   ")[1].replace("_", " ", 5)} for line in as_lines]
np.random.shuffle(as_lines)
print(as_lines[: 100])
with open("/home/v-yuancwang/AudioEditing/metadata_infos/gen.txt", "w") as f:
    for line in lines:
        wav_name = line["file_name"].replace(".wav", ".npy")
        text = line["text"].replace("\n", "")
        text = np.random.choice(["generate: ", "Generate: ", "Generate:", "Generate: "]) + text
        f.write(zero_path + "   " + os.path.join(ac_path, wav_name) + "   " + text + "\n")
    for line in as_lines[: 30000]:
        wav_name = line["file_name"].replace(".wav", ".npy")
        text = line["text"].replace("\n", "")
        text = np.random.choice(["generate: ", "Generate: ", "Generate:", "Generate: "]) + text
        f.write(zero_path + "   " + os.path.join(as_path, wav_name) + "   " + text + "\n")