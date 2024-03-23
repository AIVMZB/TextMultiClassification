import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_model():
    model = tf.keras.models.load_model("trained_model/bbc_trained_model.h5")
    return model


def preprocess_text(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    tokenized_text = pad_sequences(sequences, maxlen=max_len)
    return tokenized_text


def predict_class(text, model, tokenizer, max_len):
    tokenized_text = preprocess_text(text, tokenizer, max_len)
    prediction = model.predict(tokenized_text)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index


def main():
    model = load_model()

    with open('trained_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    max_len = 500

    with open('trained_model/label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)

    texts = [
        "wales  must learn health lessons  the new health minister for wales says there are lessons to learn from england in tackling",#pol
        "indonesia  declines debt freeze  indonesia no longer needs the debt freeze offered by the paris",
        "us regulator to rule on pain drug us food and drug regulators will decide on friday whether to recommend the sale of painkillers",
        "mobile gig aims to rock 3g forget about going to a crowded bar to enjoy a gig by the latest darlings of the music press.",
        "blair prepares to name poll date tony blair is likely to name 5 may as election day",
        "stock market eyes japan recovery japanese shares have ended the year at their highest level since 13 july amidst hopes of an economic recovery during 2005.",#pol
        "bates seals takeover ken bates has completed his takeover of leeds united.  the 73-year-old former chelsea chairman sealed the deal at 0227 gmt on friday",#sport
        "tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders",#tech
    ]

    for text in texts:
        predicted_class_index = predict_class(text, model, tokenizer, max_len)
        predicted_class = label_encoder.classes_[predicted_class_index]
        print(f"Predicted class for text '{text}': {predicted_class}")


if __name__ == "__main__":
    main()
