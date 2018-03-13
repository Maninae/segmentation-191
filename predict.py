import keras.backend as K
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from model.util import preprocess_input
from model.loss import per_pixel_softmax_cross_entropy_loss, IOU

custom_objects_dict = {
    'per_pixel_softmax_cross_entropy_loss': per_pixel_softmax_cross_entropy_loss,
    'IOU': IOU
}

def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
            help="Debug mode: more verbose, test things with less data, etc.",
            action='store_true')
    parser.add_argument("--load_path",
            help="optional path argument, if we want to load an existing model")
    args = parser.parse_args()
    return args

def get_generator():
    image_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    image_dir_wrapper = "backpack/people-val-wrapper"
    
    generator = image_datagen.flow_from_directory(
        image_dir_wrapper,
        class_mode=None,
        shuffle=False,
        batch_size=1,
        color_mode = 'rgb',
        target_size = (224,224))

    return generator


if __name__ == "__main__":
    args = parse_arguments_from_command()
    debug = args.debug
    stored_model_path = args.load_path

    if stored_model_path is None:
        stored_model_path = input("Load model from rel path: ")

    model = load_model(stored_model_path, custom_objects=custom_objects_dict)
    
    generator = get_generator()

    for x in generator: # x = input image
        y_pred = model.predict(x, batch_size=1, verbose=1) # Save memory by doing batch 1
        y_pred = K.argmax(y_pred, axis=-1) # 0 is nothing, 1 is person

        y_pred = y_pred.astype(np.uint8)



