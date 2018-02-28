import keras.backend as K


def swish(x):
    """
    Activation as described in https://arxiv.org/pdf/1710.05941.pdf
    Works better on deep networks, but slower.
    scaling: beta = 1

    This takes the place of relu in all layers when we use it.
    """
    return x * K.sigmoid(x)   

#def preprocess_input(x, data_format=None):
def preprocess_input(x):
        """
        Built off of https://github.com/titu1994/DenseNet/blob/master/densenet.py
        
        Preprocesses a tensor encoding a batch of images for DenseNet.
        We will assume data_format in Keras is 'channels_last'; don't want any
          extra overhead as this will be used during training time.

        # Args
            x: input Numpy tensor, 4D.

        # Returns
            Preprocessed tensor.
        """
        
        
        # if data_format == 'channels_first':
        #     if x.ndim == 3:
        #         # 'RGB'->'BGR'
        #         x = x[::-1, ...]
        #         # Zero-center by mean pixel
        #         x[0, :, :] -= 103.939
        #         x[1, :, :] -= 116.779
        #         x[2, :, :] -= 123.68
        #     else:
        #         x = x[:, ::-1, ...]
        #         x[:, 0, :, :] -= 103.939
        #         x[:, 1, :, :] -= 116.779
        #         x[:, 2, :, :] -= 123.68
        # else:
        
        # 'RGB'->'BGR'
        x = x[..., ::-1]

        # These values are likely the mean pixel and variance for ImageNet,
        #   that our DenseNet encoder has fitted to.
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
        x *= 0.017 # scale values

        return x