import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from model.util import preprocess_input

__default_datagen_args = dict(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True
)

DEFAULT_BATCH_SIZE = 8

def preprocess_mask(y):
    """
    The mask will be yielded by the generator as a tensor of floats in [0, 255].
    We need a binary mask and not floats for softmax+cross_entropy, hence this fn.

    Args:
      y: (?, 224, 224, 1) ndarray of floats in [0, 255].
    Returns:
      a binary mask with values in {0, 1}.
    """
    y[y <= 255./2] = 0 # Needs to be in this order, otherwise 1 gets overwritten
    y[y > 255./2] = 1
    binary_mask = y.astype(np.uint8)

    return binary_mask


def get_generator(target_img_dir_wrapper,
                  target_mask_dir_wrapper,
                  datagen_args=__default_datagen_args,
                  batch_size=DEFAULT_BATCH_SIZE,
                  seed=191):
    """
    directory = e.g. val_img_dir_wrapper
    """

    image_datagen = ImageDataGenerator(**datagen_args, preprocessing_function=preprocess_input)
    mask_datagen  = ImageDataGenerator(**datagen_args, preprocessing_function=preprocess_mask)

    # Also use a batch_size = 128 in real code. shuffle=True by default
    image_generator = image_datagen.flow_from_directory(
        target_img_dir_wrapper,
        class_mode=None,
        seed=seed,
        shuffle=True,
        batch_size=batch_size,
        color_mode='rgb',
        target_size=(224,224))
    mask_generator  = mask_datagen.flow_from_directory(
        target_mask_dir_wrapper,
        class_mode=None,
        seed=seed,
        shuffle=True,
        batch_size=batch_size,
        color_mode = 'grayscale',
        target_size = (224,224))

    # This will yield (x images, y mask labels) upon iteration
    generator = zip(image_generator, mask_generator)
    
    return generator

