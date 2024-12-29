import torch
from snac import SNAC
import numpy as np

class SpeechTokenizer:
    def __init__(self, device='cpu') -> None:
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
        self.sample_rate = 44000
        self.device = device

    def flatten_tensor(self, code_arrays, separator=4097):
        """
        Flattens the codebooks into a single sequence according to the specified interleaving pattern.
        """

        # Remove batch dimension and convert to numpy arrays
        idxs = [0, 0, 0, 0]
        flattened = []

        while idxs[0] < len(code_arrays[0]):
            flattened.append(int(code_arrays[0][idxs[0]]))
            idxs[0] += 1

            for _ in range(2):
                if idxs[1] >= len(code_arrays[1]):
                    break
                flattened.append(int(code_arrays[1][idxs[1]]))
                idxs[1] += 1

                for _ in range(2):
                    if idxs[2] >= len(code_arrays[2]):
                        break
                    flattened.append(int(code_arrays[2][idxs[2]]))
                    idxs[2] += 1

                    for _ in range(2):
                        if idxs[3] >= len(code_arrays[3]):
                            break
                        flattened.append(int(code_arrays[3][idxs[3]]))
                        idxs[3] += 1
            flattened.append(separator)

        return flattened

    def reconstruct_single_tensors(self, flattened_output, separator=4097):
        """
        Reconstructs the codebooks from the flattened output according to the interleaving pattern.
        """
        code_arrays = [[] for _ in range(4)]
        i = 0
        N = len(flattened_output)

        while i < N:
            if flattened_output[i] == separator:
                i += 1
                continue

            code_arrays[0].append(flattened_output[i])
            i += 1

            for _ in range(2):
                if i >= N or flattened_output[i] == separator:
                    break
                code_arrays[1].append(flattened_output[i])
                i += 1

                for _ in range(2):
                    if i >= N or flattened_output[i] == separator:
                        break
                    code_arrays[2].append(flattened_output[i])
                    i += 1

                    for _ in range(2):
                        if i >= N or flattened_output[i] == separator:
                            break
                        code_arrays[3].append(flattened_output[i])
                        #code_arrays[3].append(50)
                        i += 1

            # Handle separator
            if i < N and flattened_output[i] == separator:
                i += 1

        # Convert code_arrays to tensors and add batch dimension
        reconstructed_codes = [torch.tensor(codes, dtype=torch.long).unsqueeze(0) for codes in code_arrays]
        return reconstructed_codes

    def encode(self, audio_batch, separator=4097):
        """
        Encodes a batch of audio waveforms into a list of flattened lists of integer codes.
        """
        audio_batch = audio_batch.to(self.device)
        flattened_codes_list = []
        with torch.inference_mode():
            codes_list = self.model.encode(audio_batch)  # Returns list of codebooks
        for batch_index in range(len(codes_list[0])):
            # Flatten the codes
            flattened_codes = self.flatten_tensor([codes[batch_index] for codes in codes_list], separator=separator)
            flattened_codes_list.append(flattened_codes)
        return flattened_codes_list

    def decode(self, flattened_codes_list, separator=4097):
        """
        Decodes a list of flattened lists of integer codes back into audio waveforms.
        """
        reconstructed_audios = []

        for flattened_codes in flattened_codes_list:
            # Reconstruct the codebooks from flattened codes
            codes = self.reconstruct_single_tensors(flattened_codes, separator=separator)
            # Move codes to device
            codes = [code.to(self.device) for code in codes]
            print([x.shape for x in codes])
            # Decode the codes to get audio
            with torch.inference_mode():
                audio_hat = self.model.decode(codes)
            reconstructed_audios.append(audio_hat)

        # Concatenate reconstructed audios into a batch
        reconstructed_audios = torch.cat(reconstructed_audios, dim=0)
        return reconstructed_audios


# Example Usage
if __name__ == "__main__":
    tokenizer = SpeechTokenizer(device='cpu')

    # Simulate a batch of waveforms (e.g., random noise for demonstration)
    batch_size = 2
    sequence_length = 32000  # 1-second audio at 32kHz
    audio_batch = torch.randn(batch_size, sequence_length)

    # Encode the audio batch into a flattened list of integer codes
    flattened_codes = tokenizer.encode(audio_batch)
    print("Flattened Codes with Separators:")
    print(flattened_codes)

    # Assuming code_lengths are known or can be inferred
    code_lengths = [12, 24, 48, 96]  # Replace with actual lengths from your model

    # Decode the flattened codes back into waveforms
    reconstructed_waves = tokenizer.decode(flattened_codes, code_lengths)
    print("\nDecoded Waveforms Shape:", reconstructed_waves.shape)
    # Should be [batch_size, sequence_length]
