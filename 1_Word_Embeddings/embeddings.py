import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow_datasets as tfds

# print(tf.__version__)
# since version is 1.x
tf.enable_eager_execution()

# load the data: iterables containing 25000 sentences and labels as tensors
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

train_sentences = []
train_labels = []

test_sentences = []
test_labels = []

for s, l in train_data:
    # s, l are tensors. Calling numpy to extract their values
    train_sentences.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s, l in test_data:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

# For training labels needs to np.array
train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

# Tokenize the sentences
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
ovv_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token=ovv_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_sentences)

# Padding. Turn sequences into matrix
padded = pad_sequences(sequences, maxlen=max_length, truncating = trunc_type)
padded.shape

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(train_sentences[1])

# Define the neural network
model = tf.keras.Sequential([
    # 2D array with sentence's length and the embedding dimension
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Fit the model using 10 full interation over samples
num_epochs = 10
model.fit(padded, train_labels_final, epochs=num_epochs, 
          validation_data=(test_padded, test_labels_final))

# Get weights
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)