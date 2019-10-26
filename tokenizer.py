import json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Incorporate dicts as elements in a list to make it easier
# to read json as python dataframe

def parse_data(file):
    for line in open(file, 'r'):
        # use yield since we are interating to each row
        # yield produce a sequence of values into generator object
        yield json.loads(line)

# turn a generator into a list
data = list(parse_data('Sarcasm_Headlines_Dataset.json'))

sentences = []
labels = []
urls = []

for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# out of vocabulary words indexed as <OOV>
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
# unique words
len(word_index)

# Transform text to sequence of index
sequences = tokenizer.texts_to_sequences(sentences)

# Add  0s at the end of sequence to match the array length
padded = pad_sequences(sequences, padding='post')

padded.shape