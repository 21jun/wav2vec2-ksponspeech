from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
import jiwer
import fastwer


tokenizer = Wav2Vec2Processor.from_pretrained(
    "/root/develop/wav2vec2-ko/processor_korean_base")
model = Wav2Vec2ForCTC.from_pretrained(
    "/root/develop/wav2vec2-ko/wav2vec2-korean-base-continue-new/checkpoint-700")

dataset = load_dataset('json', data_files={
    "validation": 'KsponSpeech_dev_sample_shorter.json',
    "test": 'KsponSpeech_eval_clean_sample_shorter.json'}, field="data")

val_dataset = dataset['test']


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


val_dataset = val_dataset.map(map_to_array)


def map_to_pred(batch):
    input_values = tokenizer(
        batch["speech"], return_tensors="pt", padding="longest", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    batch["transcription"] = transcription

    return batch


result = val_dataset.map(
    map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

# print(result['text'], result['transcription'])

texts = [x['text'] for x in result]
transcriptions = [x['transcription'] for x in result]


cer = fastwer.score(texts, transcriptions, char_level=True)
wer = fastwer.score(texts, transcriptions)
print(cer, wer)
print(jiwer.wer(texts, transcriptions))

print(texts[0])
print(transcriptions[0])
