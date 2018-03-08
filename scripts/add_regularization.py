from keras.regularizers import l2
from keras.models import load_model # also model.save(filepath)
from keras.utils.generic_utils import get_custom_objects

from model.loss import per_pixel_softmax_cross_entropy_loss, IOU
custom_objects_dict = {
    'per_pixel_softmax_cross_entropy_loss': per_pixel_softmax_cross_entropy_loss,
    'IOU': IOU
}

#from training import get_optimizer, global_initial_learnrate


def add_l2_reg_to_model(filepath):
    """
    Run this in the git repo root! Otherwise imports can't be found.
    """
    assert filepath[-3:] == '.h5'

    print("Loading the model from %s." % filepath)
    model = load_model(filepath, custom_objects=custom_objects_dict)
    old_optimizer = model.optimizer # This is adam, so has saved weights associated with it

    print("Updating layers with regularization.")
    for layer in model.layers:
        is_regularizable = "conv" in layer.name
        if is_regularizable:
            layer.kernel_regularizer = l2(0.0001)
            print("kernel regularizer for %s is now: %s" % (layer.name, layer.kernel_regularizer))
        elif not is_regularizable:
            print("NOT", layer.name)

    #optimizer = get_optimizer(initial_learnrate=global_initial_learnrate)
    
    print("Recompiling model.")
    model.compile(loss=per_pixel_softmax_cross_entropy_loss,
                  optimizer=old_optimizer,
                  metrics=[IOU])

    print("Saving model...")
    savepath = filepath[:-3] + "_WITHREG.h5"
    model.save(savepath)
    print("Model saved to %s." % savepath)

if __name__ == "__main__":
    filepath = input("Path to model, dem Sie regularization hinzufuegen moechten?: ")
    add_l2_reg_to_model(filepath)
