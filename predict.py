from keras.models import load_model

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


def get_images():
    pass

def predict_on_images():
    pass


if __name__ == "__main__":
    args = parse_arguments_from_command()
    debug = args.debug
    stored_model_path = args.load_path

    if stored_model_path is None:
        stored_model_path = input("Load model from rel path: ")

    model = load_model(stored_model_path, custom_objects=custom_objects_dict)

    x = get_images()
    
    y_pred = model.predict(x, batch_size=1, verbose=1) # Save memory by batch 1
