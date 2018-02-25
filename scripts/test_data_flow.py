import tensorflow as tf
import keras
import numpy as np
from os.path import join

from PIL import Image

# from util import cocoTrain, cocoVal
# Common pathnames
from util import train_img_dir_wrapper, val_img_dir_wrapper, \
                 train_mask_dir_wrapper, val_mask_dir_wrapper

from keras.preprocessing.image import ImageDataGenerator

def test_img_and_mask_datagen():
    datagen_args = dict(
#        rotation_range = 20,
#        width_shift_range = 0.1,
#        height_shift_range = 0.1,
#        zoom_range = 0.1,
#        horizontal_flip = True
    )

    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen  = ImageDataGenerator(**datagen_args)

    seed = 191
    # Also use a batch_size = 128 in real code. shuffle=True by default
    val_image_generator = image_datagen.flow_from_directory(
        val_img_dir_wrapper,
        class_mode=None,
        seed=seed,
        shuffle=True,
        batch_size=128,
        target_size=(224,224))
    val_mask_generator  = mask_datagen.flow_from_directory(
        val_mask_dir_wrapper,
        class_mode=None,
        seed=seed,
        shuffle=True,
        batch_size=128,
        color_mode = 'grayscale',
        target_size = (224,224))

    # This will yield (x images, y mask labels) upon iteration
    val_generator = zip(val_image_generator, val_mask_generator)

    print("val_generator is: %s" % str(val_generator))

    for x, y in val_generator:
        print("Info about x:")
        print(type(x))
        print(x.shape)

        print("Info about y:")
        print(type(y))
        print(y.shape)
        
        print("Taking #79 (arbitrary) out of this batch, and storing their images.")
        arbitrary = 79
        x = x[arbitrary]
        y = y[arbitrary][:,:,0] # remove the last dim
        print("x shape %s, y shape %s" % (str(x.shape), str(y.shape)))

        x_img = Image.fromarray(x, mode='RGB')
        y_img = Image.fromarray(np.uint8(y * 255), mode='L')

        print("Saving the image and mask in ../tmp.")
        tmp = "../tmp"
        x_img.save(join(tmp, "dataflow_x.png"), 'PNG')
        y_img.save(join(tmp, 'dataflow_y.png'), 'PNG')
        break

if __name__ == "__main__":
    test_img_and_mask_datagen()


