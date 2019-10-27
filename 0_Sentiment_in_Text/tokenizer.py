import json
import re

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

#Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
             "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
             "during", "each", "few", "for", "from", "further", "had", "has", "have",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
             "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
             "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me",
             "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
             "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
             "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
             "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
             "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
             "we're", "we've", "were", "what", "what's", "when", "when's", "where",
             "where's", "which", "while", "who", "who's", "whom", "why", "why's",
             "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
             "yours", "yourself", "yourselves"]

for num, sentence in enumerate(sentences):
    sentences[num] = ' '.join([w for w in sentence.strip().split() if not w in stopwords])

# out of vocabulary words token <OOV>
# Tokenizer object used to tokenize sentences
tokenizer = Tokenizer(oov_token="<OOV>")
# Tokenize a list of sentences
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
# unique words
len(word_index)

# Encode a list os sentences to use the tokens
sequences = tokenizer.texts_to_sequences(sentences)

# Add  0s at the end of sequence to match the length of the
# longest seqeunce
padded = pad_sequences(sequences, padding='post')

padded.shape
