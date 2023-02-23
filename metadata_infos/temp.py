import os

with open("/home/v-yuancwang/AudioEditing/metadata_infos/inpainting_refine.txt", "r") as f:
    lines = f.readlines()

with open("/home/v-yuancwang/AudioEditing/metadata_infos/inpainting_refine.txt", "w") as f:
    for line in lines:
        line = line.replace("/sr/", "/inpainting/")
        f.write(line)