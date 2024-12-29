import pyaudio
import numpy as np
from speech_tokenizer import SpeechTokenizer
import torch
import torchaudio
from speech_tokenizer import SpeechTokenizer
import numpy as np
import itertools
from pathlib import Path
import os
import librosa
import soundfile as sf
import timeit

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

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
def find_last_instance_of_seperator(lst, element=4097):
    for i in range(len(lst)-1, -1, -1):
        if lst[i]==element:
            return i
    raise ValueError

def find_first_instance_of_seperator(lst, element=4097):
    for i in range(0, len(lst)):
        if lst[i]==element:
            return i
    raise ValueError

@time_this
def batch_list(lst, batch_size):
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, batch_size)), [])

sample_rate = 44000

@time_this
def encode(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend='soundfile')
    if sample_rate != tokenizer.sample_rate:
        print(f"resampled to {tokenizer.sample_rate}")
        waveform = resample_waveform(waveform, sample_rate, tokenizer.sample_rate)
    if waveform.size(0) > 1:
        print(f"averaged audio to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    single_doc = []
    encoded_batch = tokenizer.encode([waveform])
    for x in encoded_batch:
        single_doc.extend(x[:-1])
    return single_doc

@time_this
def decode(tokens):
    return tokenizer.decode(np.array([tokens[find_first_instance_of_seperator(tokens):find_last_instance_of_seperator(tokens)]]))

output_tokens = encode("./destin9s.mp3")
print(",".join([str(x) for x in list(output_tokens)]))