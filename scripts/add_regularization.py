from keras.regularizers import l2
from keras.models import load_model # also model.save(filepath)
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Conv2D, Conv2DTranspose, Dropout

from model.loss import per_pixel_softmax_cross_entropy_loss, IOU
custom_objects_dict = {
    'per_pixel_softmax_cross_entropy_loss': per_pixel_softmax_cross_entropy_loss,
    'IOU': IOU
}


def add_l2_reg_to_model(model, l2reg_lambda):
    """
    Run this in the git repo root! Otherwise imports can't be found.
    """

    print("Updating layers with l2 regularization, lambda=%s." % str(l2reg_lambda))
    for layer in model.layers:
        
        is_regularizable = isinstance(layer, Conv2D) or isinstance(layer, Conv2DTranspose)
        double_check = "conv" in layer.name
        print("is_regularizable == double_check: %s" % str(is_regularizable == double_check))

        if is_regularizable:
            layer.kernel_regularizer = l2(l2reg_lambda)
            print("kernel regularizer for %s is now: %s" % (layer.name, layer.kernel_regularizer))
        else:
            print("NOT", layer.name)
    
    return model
    


def change_dropout_rate(model, dropout_rate):
    print("Changing dropout rate of dropout layers to %f." % dropout_rate)
    for layer in model.layers:
        is_dropout = isinstance(layer, Dropout)
        double_check = "dropout" in layer.name
        if is_dropout:
            layer.rate = dropout_rate
            print("Changed dropout on layer %s." % layer.name)
        else:
            print("NOT", layer.name)

    return model



if __name__ == "__main__":
    print("[db-add_regularization.py] Run this script from the git repo root, to find modules correctly.")
    filepath = input("Path to model, that you'd like to add regularization to: ")

    assert filepath[-3:] == '.h5'

    print("Loading the model from %s." % filepath)
    model = load_model(filepath, custom_objects=custom_objects_dict)
    old_optimizer = model.optimizer # This is adam, so has saved weights associated with it

    model = add_l2_reg_to_model(model, 1e-3)
    model = change_dropout_rate(model, 0.3)

    print("Recompiling model.")
    model.compile(loss=per_pixel_softmax_cross_entropy_loss,
                  optimizer=old_optimizer,
                  metrics=[IOU])

    print("Saving model...")
    savepath = filepath[:-3] + "_WITHREGS.h5"
    model.save(savepath)
    print("Model saved to %s." % savepath)


