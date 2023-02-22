import os
import numpy as np

save_path = "/blob/v-yuancwang/audio_editing_data/replacement_refine/mel"

with open("/home/v-yuancwang/AudioEditing/metadata_infos/replacement_refine.txt", "w") as f:
    with open("/home/v-yuancwang/AudioEditing/metadatas/audioset_fsdesc_replacement_refine.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        f1, f2, t1, t2, _ = line.split("   ")
        f1 = os.path.join(save_path, f1.replace(".wav", ".npy"))
        f2 = os.path.join(save_path, f2.replace(".wav", ".npy"))
        t1 = t1.lower().replace("_", " ", 5)
        t2 = t2.lower().replace("_", " ", 5)
        t = np.random.choice(["Replace: ", "Replace: ", "Replace:", "replace: ", "Replacement: "]) + t1 + " to " + t2
        f.write(f1 + "   " + f2 + "   " + t + "\n")
    with open("/home/v-yuancwang/AudioEditing/metadatas/audioset_fsdesc_replacement_refine_1.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        f1, f2, t1, t2, _ = line.split("   ")
        f1 = os.path.join(save_path, f1.replace(".wav", ".npy"))
        f2 = os.path.join(save_path, f2.replace(".wav", ".npy"))
        t1 = t1.lower().replace("_", " ", 5)
        t2 = t2.lower().replace("_", " ", 5)
        t = np.random.choice(["Replace: ", "Replace: ", "Replace:", "replace: ", "Replacement: "]) + t1 + " to " + t2
        f.write(f1 + "   " + f2 + "   " + t + "\n")
        