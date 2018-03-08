from keras.regularizers import l2
from keras.models import load_model # also model.save(filepath)

from ..model.loss import per_pixel_softmax_cross_entropy_loss, IOU
from ..training import get_optimizer


def add_l2_reg_to_model(filepath):
    """
    Run this in the git repo root! Otherwise imports can't be found.
    """
    print("Loading the model from %s." % filepath)
    model = load_model(filepath)

    print("Updating layers with regularization.")
    for layer in model.layers:
        is_regularizable = "conv" in layer.name

    print("Recompiling model.")
    model.compile()

    print("Saving model to current dir.")
    model.save()

if __name__ == "__main__":
    filepath = input("Path to model, dem Sie regularization hinzufuegen moechten?: ")
    add_l2_reg_to_model(filepath)
