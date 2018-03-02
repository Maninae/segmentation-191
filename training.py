import numpy as np

from util.dataflow import preprocess_mask, get_generator
from util.pathutil import train_img_dir_wrapper, train_mask_dir_wrapper, \
                          val_img_dir_wrapper, val_mask_dir_wrapper

from model.diamondback import DiamondbackModelCreator
from model.loss import per_pixel_softmax_cross_entropy_loss

from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam


class IntraEpochHistory(Callback):
    """ Taken from the keras tutorial on callbacks.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def get_callbacks_list():
    print("[db-training] Getting our callbacks...")
    history = IntraEpochHistory()

    #savepath = "/output/model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    savepath = "model/weights/diamondback_ep{epoch:02d}-vloss={val_loss:.4f}-tloss={train_loss:.4f}.h5"
    checkpointer = ModelCheckpoint(savepath, monitor='val_loss', verbose=1, save_best_only=True)

    def step_decay(nb_epochs, lr):
        """ This needs to be in harmony with get_optimizer(initial_learnrate)!
            Also depends on the number epochs we are doing. Right now:
            90 epochs, /10 downscaling at 30, 60.
        """
        lr = 0.1 # Forget the parameter. Also, change this if Adam's initial lr changes
        if nb_epochs > 60:
            lr /= 100
        elif nb_epochs > 30:
            lr /= 10
        return lr
    lrate_scheduler = LearningRateScheduler(step_decay)
    
    return history, checkpointer, lrate_scheduler


def get_generators():
    print("[db-training] Getting the train data generator.")
    train_generator = get_generator(train_img_dir_wrapper, train_mask_dir_wrapper)
    
    print("[db-training] Getting the val data generator.")
    val_generator = get_generator(val_img_dir_wrapper, val_mask_dir_wrapper)

    return train_generator, val_generator


def get_model(nb_extra_sdn_units):
    print("[db-training] Getting the model...")
    creator = DiamondbackModelCreator(
                dn_encoder_path="model/densenet_encoder/encoder_model.h5",
                nb_extra_sdn_units=nb_extra_sdn_units)

    model = creator.create_diamondback_model()
    return model

def get_adam_optimizer(initial_learnrate):
    print("[db-training] Getting the optimizer...")
    optimizer = Adam(lr=0.1)
    return optimizer

if __name__ == "__main__":

    model = get_model(nb_extra_sdn_units=1)
    optimizer = get_adam_optimizer(initial_learnrate=0.1)
    
    print("[db-training] Compiling the model...")
    model.compile(loss=per_pixel_softmax_cross_entropy_loss, optimizer=optimizer)

    history, checkpointer, lrate_scheduler = get_callbacks_list() # History, Checkpointer
    callbacks_list = [history, checkpointer, lrate_scheduler]

    train_generator, val_generator = get_generators()

    print("[db-training] Beginning to fit diamondback model.")
    history_over_epochs = model.fit_generator(
        train_generator,
        steps_per_epoch=500, # 64115 / 128
        epochs=90,
        validation_data=val_generator,
        validation_steps=21, # 2693 / 128
        callbacks=callbacks_list)
