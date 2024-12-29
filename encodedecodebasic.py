import numpy as np
import torch
import torchaudio
from speech_tokenization_experiments import SpeechTokenizer
import numpy as np
import itertools
from pathlib import Path
import os
import soundfile as sf
import timeit

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device = "cpu"

tokenizer = SpeechTokenizer(device=device)

def time_this(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(f"{func.__name__} Execution Time: {execution_time} seconds")
        return result
    return wrapper

@time_this
def resample_waveform(waveform, sample_rate, new_sample_rate, chunk_size=10000000):
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resampled_waveform = []

    for start in range(0, waveform.size(1), chunk_size):
        end = min(start + chunk_size, waveform.size(1))
        chunk = waveform[:, start:end]
        resampled_chunk = resampler(chunk)
        resampled_waveform.append(resampled_chunk)

    return torch.cat(resampled_waveform, dim=1)

@time_this
def batch_list(lst, batch_size):
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, batch_size)), [])

sample_rate = 44000

def pad_waveform(waveform, multiple):
    length = waveform.size(1)
    remainder = length % multiple
    if remainder != 0:
        padding_size = multiple - remainder
        padding = torch.zeros((waveform.size(0), padding_size), device=waveform.device)
        waveform = torch.cat([waveform, padding], dim=1)
    return waveform

def batch_list(lst, batch_size):
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, batch_size)), [])


@time_this
def encode(waveform, sample_rate):
    if sample_rate != tokenizer.sample_rate:
        print(f"resampled to {tokenizer.sample_rate}")
        waveform = resample_waveform(waveform, sample_rate, tokenizer.sample_rate)
    if waveform.size(0) > 1:
        print(f"averaged audio to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    batch_element_size = 192*64
    # Reshape waveform into batches
    waveform = pad_waveform(waveform, batch_element_size)
    waveform_batches = waveform.view(-1, 1, batch_element_size)  # Shape: (num_batches, 1, batch_size)
    # Encode the batches
    encoded_batches = tokenizer.encode(waveform_batches)
    print("encoded batches", len(encoded_batches))

    # Concatenate encoded tokens from all batches
    #output_tokens = torch.cat(encoded_batches, dim=1)
    # Flatten the list of lists into a single list
    output_tokens = []
    for batch in encoded_batches:
        output_tokens.extend(batch)

    return [output_tokens]

@time_this
def decode(tokens):
    return tokenizer.decode(np.array([tokens[find_first_instance_of_seperator(tokens):find_last_instance_of_seperator(tokens)]]))

audio_path = "scarletheramireal.mp3"
audio_path = "./destin9s.mp3"
audio_path = "Krithik Puthalath (mp3cut.net).mp3"
wav, sample_rate = torchaudio.load(audio_path, backend='soundfile')
print("length of waveform in seconds", len(wav[1]) / sample_rate)

output_tokens = encode(wav, sample_rate)
print("output tokens", output_tokens)

print(len(output_tokens[0][:3168+1]))
wav = tokenizer.decode([output_tokens[0][:3168+1]]) #152 
print([x.shape for x in wav])
sf.write("output.wav", wav[0].cpu().numpy().squeeze().astype(np.float32), tokenizer.sample_rate)
# 192 samples / 1 token
