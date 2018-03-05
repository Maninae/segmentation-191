import keras.backend as K
import tensorflow as tf
from tf.bitwise import bitwise_or

def per_pixel_softmax_cross_entropy_loss(y_true, y_pred):
	return K.sum(K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


def IOU(y_true, y_pred):
	# argmax to predict, add axis to make (224, 224, 1), cast to uint8
	y_pred = K.argmax(y_pred)[..., np.newaxis].astype(tf.uint8)

	intersection = K.sum(y_true * y_pred)
	union = bitwise_or(y_true, y_pred)

	return intersection / union






# def colorization_loss(y_true, y_pred):
#   """ |y_true|: Our true colors. A tensor (batch_size, H, W) with entries 
#                 specifying one of the buckets that pixel's color is in.
#       |y_pred|: A (batch_size, H, W, NUM_VALID_BUCKETS) tensor with last dimension a
#                 softmax over bucket probabilities.
  
#       This loss involves computing the softmax cross-entropy over pixel's
#         predicted color bucket, over all images in the batch.
#   """

#   return Lambda(lambda (_yt, _yp): K.sum(K.sparse_categorical_crossentropy(_yt, _yp)),
#                   output_shape=lambda (_yt, _yp): _yt,
#                   name='loss')([y_true, y_pred])
