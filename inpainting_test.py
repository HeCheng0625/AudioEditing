from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from transformers import T5EncoderModel, T5TokenizerFast
import torch
import os
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
import torchaudio

MODEL_PATH = "/blob/v-yuancwang/AudioEditingModel/Diffusion_SE/checkpoint-72000"
CFG = 6.0
TORCH_DEVICE = "cuda:4"
SAVE_MEL_PATH = "/blob/v-yuancwang/audio_editing_test/inpainting/72000/6.0/mel"

model_path = MODEL_PATH
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

vae.to(TORCH_DEVICE)
text_encoder.to(TORCH_DEVICE)
unet.to(TORCH_DEVICE)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)


with open("/home/v-yuancwang/AudioEditing/metadatas/audiocaps_test_metadata.jsonl", "r") as f:
    lines = f.readlines()
lines = [eval(line) for line in lines]
test_set = {}
for line in lines:
    file_name, text = line["file_name"], line["text"]
    file_name = file_name.replace(".wav", "")
    if file_name not in test_set:
        test_set[file_name] = []
    test_set[file_name].append(text)
print(test_set)

for file_name in tqdm(test_set.keys()):
    texts = test_set[file_name]

    text = np.random.choice(texts)

    text = "Inpainting: " + text
    prompt = [text]
    text_input = tokenizer(prompt, max_length=tokenizer.model_max_length, truncation=True, padding="do_not_pad", return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(TORCH_DEVICE))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(TORCH_DEVICE))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    mel_src = np.load(os.path.join("/blob/v-yuancwang/audio_editing_data/inpainting_test/mel", file_name+".npy"))
    latents_src = torch.Tensor(np.array([[mel_src]])).to(TORCH_DEVICE)
    latents_src = vae.encode(latents_src).latent_dist.sample()

    num_inference_steps = 100
    scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)

    guidance_scale = CFG
    scheduler.set_timesteps(num_inference_steps)

    latents = torch.randn((1, 4, 10, 78)).to(TORCH_DEVICE)
    latents_src_input = torch.cat([latents_src] * 2)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(torch.cat((latent_model_input, latents_src_input), dim=1), t, encoder_hidden_states=text_embeddings).sample
            # noise_pred = unet(torch.cat((latent_model_input, latent_model_input), dim=1), t, encoder_hidden_states=text_embeddings).sample
            # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents_out = latents

    with torch.no_grad():
        res = vae.decode(latents_out).sample
    res = res.cpu().numpy()[0,0,:,:]

    np.save(os.path.join(SAVE_MEL_PATH, file_name + ".npy"), res)