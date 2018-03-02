import numpy as np

from keras.preprocessing.image import ImageDataGenerator

__default_datagen_args = dict(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True
)


def preprocess_mask(y):
    """
    The mask will be yielded by the generator as a tensor of floats in [0, 255].
    We need a binary mask and not floats for softmax+cross_entropy, hence this fn.

    Args:
      y: (?, 224, 224, 1) ndarray of floats in [0, 255].
    Returns:
      a binary mask with values in {0, 1}.
    """

    y[y > 255./2] = 1
    y[y <= 255./2] = 0
    binary_mask = y.astype(np.uint8)
    return binary_mask


def get_generator(target_img_dir_wrapper,
                  target_mask_dir_wrapper,
                  datagen_args=__default_datagen_args,
                  batch_size=128,
                  seed=191):
    """
    directory = e.g. val_img_dir_wrapper
    """

    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen  = ImageDataGenerator(**datagen_args)

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






#     print("val_generator is: %s" % str(val_generator))

#     for x, y in val_generator:
#         print("Info about x:")
#         print(x.shape)

#         print("Info about y:")
#         print(y.shape)
        
#         print("Taking some out of this batch, and storing their images.")
#         arbitrary_nums  = [120]
#         for arbitrary in arbitrary_nums:
#             x_arr = x[arbitrary]
#             y_arr = y[arbitrary][:,:,0] # remove the last dim
            

#             # casting to uint8 is necessary for displaying photos.
#             # For the machine learning, don't need to do anything.
#             x_arr_int = x_arr.astype(np.uint8)
#             y_arr_int = y_arr.astype(np.uint8)
#             xdiff = np.sum(x_arr - x_arr_int)
#             ydiff = np.sum(y_arr - y_arr_int)
#             print("xdiff: %f, ydiff: %f" % (xdiff, ydiff))
#             #x_img = Image.fromarray(x, mode='RGB')
#             #y_img = Image.fromarray(np.uint8(y * 255), mode='L')
#             difference_array = x_arr - x_arr_int
#             np.save(join(tmp, "diff.npy"), difference_array)

#             print("Saving the image and mask in ../tmp.")
#             #x_img.save(join(tmp, "dataflow_x.png"), 'PNG')
#             #y_img.save(join(tmp, 'dataflow_y.png'), 'PNG')
#             plt.imshow(x_arr)
#             plt.savefig(join(tmp, "dataflow_%d_x.png" % arbitrary))
#             plt.imshow(y_arr)
#             plt.savefig(join(tmp, "dataflow_%d_y.png" % arbitrary))
#             plt.imshow(x_arr_int)
#             plt.savefig(join(tmp, "dataflow_%d_x_INT.png" % arbitrary))
#             plt.imshow(y_arr_int)
#             plt.savefig(join(tmp, "dataflow_%d_y_INT.png" % arbitrary))
#         break

# if __name__ == "__main__":
#     test_img_and_mask_datagen()


