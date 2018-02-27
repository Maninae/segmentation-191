import tensorflow as tf
import keras

from keras.models import model_from_json
from keras.models import load_model
from os.path import join

def load_densenet_encoder_model(path="encoder_model.h5"):
    model = load_model(path)
    return model
    
if __name__ == "__main__":
    model = load_densenet_encoder_model()
    model.summary()

