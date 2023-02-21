import numpy as np
import os
with open("/home/v-yuancwang/AudioEditing/metadatas/audiocaps_train_metadata.jsonl", "r") as f:
    lines = f.readlines()
lines = [eval(line) for line in lines]
ac_path = "/blob/v-yuancwang/audio_editing_data/audiocaps/mel"
as_path = "/blob/v-yuancwang/audio_editing_data/audioset96/mel"
ac_set = set(os.listdir(ac_path))
as_set = set(os.listdir(as_path))
save_path = "/blob/v-yuancwang/audio_editing_data/sr/mel"

ac_refine_path = "/blob/v-yuancwang/audio_editing_data/audiocaps_refine/mel"
as_refine_path = "/blob/v-yuancwang/audio_editing_data/audioset96_refine/mel"
ac_refine_set = set(os.listdir(ac_refine_path))
as_refine_set = set(os.listdir(as_refine_path))

with open("/home/v-yuancwang/AudioEditing/metadatas/audioset96_file_label.txt", "r") as f:
    as_lines = f.readlines()
as_lines = [{"file_name": line.replace("\n", "").split("   ")[0], "text": line.replace("\n", "").split("   ")[1].replace("_", " ", 5)} for line in as_lines]
np.random.shuffle(as_lines)
# print(as_lines[: 100])

with open("/home/v-yuancwang/AudioEditing/metadata_infos/inpainting_refine.txt", "w") as f:
    for line in lines:
        wav_name = line["file_name"].replace(".wav", ".npy")
        text = line["text"].replace("\n", "")
        if wav_name not in ac_set:
            continue
        text = np.random.choice(["Inpainting:", "inpainting: ", "Inpainting: ", "Inpainting: "]) + text
        if wav_name in ac_refine_set:
            f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(ac_refine_path, wav_name) + "   " + text + "\n")
        else:
            f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(ac_path, wav_name) + "   " + text + "\n")
    for line in as_lines[:150000]:
        wav_name = line["file_name"].replace(".wav", ".npy")
        text = line["text"].replace("\n", "")
        if wav_name not in as_set:
            continue
        text = np.random.choice(["Inpainting:", "inpainting: ", "Inpainting: ", "Inpainting: "]) + text
        if wav_name in as_refine_set:
            f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(as_refine_path, wav_name) + "   " + text + "\n")
        else:
            f.write(os.path.join(save_path, wav_name) + "   " + os.path.join(as_path, wav_name) + "   " + text + "\n")