import tensorflow as tf
import keras

from keras.models import model_from_json

def load_dense_encoder_model():
    with open("sdn_encoder_architecture.json", 'r') as f:
        json_string = f.readlines()
    model_json = json_string[0]

    model = model_from_json(model_json)
    model.load_weights("sdn_encoder_weights.h5")
    return model

if __name__ == "__main__":
    model = load_dense_encoder_model()
    model.summary()

