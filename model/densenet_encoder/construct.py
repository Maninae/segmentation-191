"""
This uses the implementation of DenseNet in Keras created
by titu1994 on Github:

    https://github.com/titu1994/DenseNet

The imported |densenet| library is from this repo.
This script was RUN IN THAT REPO, and the files moved over.
"""

import densenet
import keras
from keras.models import Model
import json

image_dim = (224, 224, 3)  # Tensorflow backend
print("Image dimensions set to %s." % str(image_dim))

print("Loading denseNet model...")
densenet_model = densenet.DenseNetImageNet161(input_shape=image_dim, include_top=False)

print("Extracting a subset of denseNet to use as an encoder.")
n_final_layer = 393
print("Final layer is %d." % n_final_layer)

inp = densenet_model.input

res56 = densenet_model.layers[49]
print("res56 layer is %s" % str(res56))
out_res56 = res56.output

res28 = densenet_model.layers[137]
print("res28 layer is %s" % str(res28))
out_res28 = res28.output

out_layer = densenet_model.layers[393]
print("out layer is %s" % str(out_layer))
out = out_layer.output

print("Input shape:  %s" % str(inp))
print("Res 28 shape: %s" % str(out_res28))
print("Res 56 shape: %s" % str(out_res56))
print("Output shape: %s" % str(out))


encoder = Model(inputs=inp, outputs=[out_res56, out_res28, out])
# encoder.summary()

print("Freezing encoder weights. All layers set trainable=False.")
for layer in encoder.layers:
    layer.trainable = False

print("Saving the encoder weights and architecture.")
encoder.save("encoder_model.h5")

print("Done.")

