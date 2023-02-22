import os
import numpy as np

save_path = "/blob/v-yuancwang/audio_editing_data/add_refine/mel"

with open("/home/v-yuancwang/AudioEditing/metadata_infos/add_refine.txt", "w") as f:
    with open("/home/v-yuancwang/AudioEditing/metadatas/audiocaps_add.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        tgt, src, _, _, t, _ = line.split("   ")
        tgt = os.path.join(save_path, tgt.replace(".wav", ".npy"))
        src = src.replace(".wav", ".npy")
        t = t.lower().replace("_", " ", 5)
        t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the background"
        f.write(src + "   " + tgt + "   " + t + "\n")
    with open("/home/v-yuancwang/AudioEditing/metadatas/audioset_fsdesc.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        tgt, src, _, _, t, type_c = line.split("   ")
        type_c  = type_c.replace("\n", "")
        tgt = os.path.join(save_path, tgt.replace(".wav", ".npy"))
        src = src.replace(".wav", ".npy")
        t = t.lower().replace("_", " ", 5)
        if type_c == "0":
            t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the " + np.random.choice(["front", "beginning", "front"])
        if type_c == "1":
            t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the middle"
        if type_c == "2":
            t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the end"
        f.write(src + "   " + tgt + "   " + t + "\n")

# import os
# import numpy as np

# save_path = "/blob/v-yuancwang/audio_editing_data/add_refine/mel"

# with open("/home/v-yuancwang/AudioEditing/metadata_infos/drop_refine.txt", "w") as f:
#     with open("/home/v-yuancwang/AudioEditing/metadatas/audiocaps_add.txt", "r") as ff:
#         lines = ff.readlines()
#     for line in lines:
#         src, tgt, _, _, t, _ = line.split("   ")
#         src = os.path.join(save_path, src.replace(".wav", ".npy"))
#         tgt = tgt.replace(".wav", ".npy")
#         t = t.lower().replace("_", " ", 5)
#         t = np.random.choice(["Drop: ", "Drop: ", "Drop:", "drop: "]) + t
#         f.write(src + "   " + tgt + "   " + t + "\n")
#     with open("/home/v-yuancwang/AudioEditing/metadatas/audioset_fsdesc.txt", "r") as ff:
#         lines = ff.readlines()
#     for line in lines:
#         src, tgt, _, _, t, type_c = line.split("   ")
#         src = os.path.join(save_path, src.replace(".wav", ".npy"))
#         tgt = tgt.replace(".wav", ".npy")
#         t = t.lower().replace("_", " ", 5)
#         # if type_c == "0":
#         #     t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the front"
#         # if type_c == "1":
#         #     t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the middle"
#         # if type_c == "2":
#         #     t = np.random.choice(["Add: ", "Add: ", "Add:", "add: "]) + t + " in the end"
#         t = np.random.choice(["Drop: ", "Drop: ", "Drop:", "drop: "]) + t
#         f.write(src + "   " + tgt + "   " + t + "\n")