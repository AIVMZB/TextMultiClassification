import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


def load_glove_embeddings(embeddings_path, tokenizer, embedding_dim):
    embeddings_index = {}
    with open(embeddings_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_data(file_path):
    return pd.read_csv(file_path)


def save_object(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def tokenize_texts(texts, max_words, max_len):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=max_len)
    return tokenizer, x


def encode_labels(labels):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return label_encoder, y


def preprocess(config):
    data = load_data('bbc-text.csv')

    tokenizer, x = tokenize_texts(data['text'], config.max_words, config.max_len)
    save_object(tokenizer, 'trained_model/tokenizer.pickle')

    label_encoder, y = encode_labels(data['category'])
    save_object(label_encoder, 'trained_model/label_encoder.pickle')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return tokenizer, x_train, y_train, x_test, y_test
