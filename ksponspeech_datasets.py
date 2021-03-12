from datasets import load_dataset


dataset = load_dataset('json', data_files={
                       "train": 'KsponSpeech_train.json',
                       "validation": 'KsponSpeech_dev.json',
                       "test": 'KsponSpeech_eval_clean.json'}, field="data")

train = dataset['validation'][:100]

print(train['text'])