import wave
import json
from pathlib import Path
import re


def pcm2wav(pcm_file, wav_file, channels=1, bit_depth=16, sampling_rate=16000):

    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth "+str(bit_depth) +
                         " must be a multiple of 8.")

    # Read the .pcm file as a binary file and store the data to pcm_data
    with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read()

        obj2write = wave.open(wav_file, 'wb')
        obj2write.setnchannels(channels)
        obj2write.setsampwidth(bit_depth // 8)
        obj2write.setframerate(sampling_rate)
        obj2write.writeframes(pcm_data)
        obj2write.close()


train_path = Path("KsponSpeech_01/KsponSpeech_0038/KsponSpeech_037840.pcm")
get = train_path.glob("**/*.pcm")
pcm = [x for x in get if x.is_file()]


for fpath in pcm:
    fname = fpath.name.split(".")[0]
    wav_name = fname + ".wav"
    print(fpath.parent/wav_name)
    pcm2wav(str(fpath), str(fpath.parent+"_wav"/wav_name), 1, 16, 16000)
