from transformers import MimiModel, AutoFeatureExtractor
import torchaudio

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

audio_sample, sr = torchaudio.load("./scarletheramireal.wav")
if sr != feature_extractor.sampling_rate:
    audio_sample = torchaudio.functional.resample(audio_sample, sr, feature_extractor.sampling_rate)

inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]

audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values