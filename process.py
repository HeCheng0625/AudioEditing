import os

mel_path = "/blob/v-yuancwang/audio_editing_data/audioset96/mel"
wav_path = "/blob/v-yuancwang/audio_editing_data/audioset96/wav"

mels = os.listdir(mel_path)
wavs = os.listdir(wav_path)
print(len(mels), len(wavs))
print(mels[: 5], wavs[: 5])