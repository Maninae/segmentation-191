import numpy as np

from dataflow import preprocess_mask, get_generator
from model.diamondback import DiamondbackModelCreator
from model.loss import per_pixel_softmax_cross_entropy_loss

from scripts.util import train_img_dir_wrapper, train_mask_dir_wrapper, \
                         val_img_dir_wrapper, val_mask_dir_wrapper


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def get_callbacks_list():
    history = LossHistory()

    #savepath = "/output/model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    savepath = "model/weights/diamondback_ep{epoch:02d}-vloss={val_loss:.4f}-tloss={train_loss:.4f}.h5"
    checkpointer = ModelCheckpoint(savepath, monitor='val_loss', verbose=1, save_best_only=True)

    return [history, checkpointer]


def get_generators():
    train_generator = get_generator(train_img_dir_wrapper, train_mask_dir_wrapper)
    val_generator = get_generator(val_img_dir_wrapper, val_mask_dir_wrapper)

    return train_generator, val_generator


def get_model():
    creator = DiamondbackModelCreator(
                db_encoder_path="model/densenet_encoder/encoder_model.h5",
                nb_extra_sdn_units=1)

    model = creator.create_diamondback_model()


if __name__ == "__main__":

    model = get_model()
    model.compile(loss=per_pixel_softmax_cross_entropy_loss, optimizer='adam')

    callbacks_list = get_callbacks_list() # History, Checkpointer

    train_generator, val_generator = get_generators()

    model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=,
        validation_data=val_generator,
        validation_steps=1234,
        callbacks=callbacks_list)
