import keras.backend as K
import tensorflow as tf
import numpy as np

def per_pixel_softmax_cross_entropy_loss(y_true, y_pred):
        return K.sum(K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


def IOU(y_true, y_pred):
	# argmax to predict, add axis to make (224, 224, 1), cast to int32
        y_pred = tf.to_int32(K.argmax(y_pred)[..., np.newaxis])
        y_true = tf.to_int32(y_true)
        #print("ypred max min:", np.amax(y_pred), np.amin(y_pred))
        #print("ytrue max min:", np.amax(y_true), np.amin(y_true))
        
        #tf.assert_equal(tf.reduce_max(y_pred), 1)
        #tf.assert_equal(tf.reduce_min(y_pred), 0)
        #tf.assert_equal(tf.reduce_max(y_true), 1)
        #tf.assert_equal(tf.reduce_min(y_true), 0)

        intersection = K.sum(y_true * y_pred)
        union = K.sum(tf.bitwise.bitwise_or(y_true, y_pred))

        return intersection / union

