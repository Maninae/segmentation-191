import numpy as np

from util.dataflow import preprocess_mask, get_generator
from util.dataflow import DEFAULT_BATCH_SIZE

from util.pathutil import train_img_dir_wrapper, train_mask_dir_wrapper, \
                          val_img_dir_wrapper, val_mask_dir_wrapper, \
                          train_img_debug, train_mask_debug, \
                          val_img_debug, val_mask_debug

from model.diamondback import DiamondbackModelCreator
from model.loss import per_pixel_softmax_cross_entropy_loss, IOU

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, \
                            TensorBoard
from keras.optimizers import Adam, SGD
from keras.models import load_model

import datetime
import argparse

global_initial_learnrate = 1e-4
custom_objects_dict = {
        'per_pixel_softmax_cross_entropy_loss': per_pixel_softmax_cross_entropy_loss,
        'IOU': IOU
}


class IntraEpochHistory(Callback):
    """ Taken from the keras tutorial on callbacks.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, logs={}):
        with open("/output/history_ep{epoch:02d}-vloss={val_loss:.4f}" \
                  "-vIOU={val_IOU:.4f}-tloss={loss:.4f}-tIOU={IOU:.4f}.pkl", 'wb') as f:
            pickle.dump(self.losses, f)
        # clear the losses we have so far in this epoch, in prep for next epoch.
        self.losses = []


"""
This function was replaced by the ReduceLROnPlateau callback.
"""
# def step_decay(nb_epochs, lr=1e-4): # Needs default value for backward TF compatibility
#     """ This needs to be in harmony with get_optimizer(initial_learnrate)!
#         Also depends on the number epochs we are doing. Right now:
#         90 epochs, /10 downscaling at 30, 60.
#     """
#     if nb_epochs > 60:
#         lr /= 100
#     elif nb_epochs > 30:
#         lr /= 10
#     return lr
# lrate_scheduler = LearningRateScheduler(step_decay)

def get_callbacks_list():
    print("[db-training] Getting our callbacks...")
    history = IntraEpochHistory()

    # From AIH:
    #savepath = "/output/model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    #savepath = "model/weights/diamondback_ep{epoch:02d}-vloss={val_loss:.4f}-tloss={train_loss:.4f}.h5"
    savepath = "/output/diamondback_ep{epoch:02d}" \
               "-tloss={loss:.4f}-vloss={val_loss:.4f}" \
               "-tIOU={IOU:.4f}-vIOU={val_IOU:.4f}.h5"
    checkpointer = ModelCheckpoint(savepath,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False)

    # Reduce lr x10 if no val loss improvement after 3 epochs
    lrate_plateau_reducer = ReduceLROnPlateau(monitor='val_loss', 
                                              factor=0.25, min_lr=1e-6,
                                              patience=3, epsilon=100.,
                                              verbose=1, mode='min')

    # Won't need batch_size param unless histogram_freq > 0, but just in case
    tensorboard = TensorBoard(log_dir="/output/logs/{}"
                                .format(datetime.datetime.now()),
                              write_graph=False,
                              batch_size=DEFAULT_BATCH_SIZE)

    callbacks_list = [history, checkpointer, lrate_plateau_reducer, tensorboard]
    print("[db-training] We have the following callbacks:")
    for cb in callbacks_list:
        print("[db-training] %s" % str(cb))

    return callbacks_list


def get_generators(debug=False):
    if debug:
        train_gen_img_path = train_img_debug
        train_gen_mask_path = train_mask_debug
        val_gen_img_path = val_img_debug
        val_gen_mask_path = val_mask_debug
    else:
        train_gen_img_path = train_img_dir_wrapper
        train_gen_mask_path = train_mask_dir_wrapper
        val_gen_img_path = val_img_dir_wrapper
        val_gen_mask_path = val_mask_dir_wrapper

    print("[db-training] Getting the train data generator.")
    train_generator = get_generator(train_gen_img_path, train_gen_mask_path)
    
    print("[db-training] Getting the val data generator.")
    val_generator = get_generator(val_gen_img_path, val_gen_mask_path)

    return train_generator, val_generator


def get_model(nb_extra_sdn_units, dn_encoder_path):
    print("[db-training] Getting the model...")
    creator = DiamondbackModelCreator(
                dn_encoder_path=dn_encoder_path,
                nb_extra_sdn_units=nb_extra_sdn_units)

    model = creator.create_diamondback_model()
    return model


def get_optimizer(initial_learnrate):
    print("[db-training] Getting the optimizer...")
    optimizer = Adam(lr=initial_learnrate)
    #optimizer = SGD(lr=initial_learnrate)
    return optimizer


def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
        help="Debug mode: load instead the coco-debug directory as data",
        action='store_true')
    parser.add_argument("--load_path",
        help="optional path argument, if we want to load an existing model")
    args = parser.parse_args()
    return args

######################################################

if __name__ == "__main__":
    args = parse_arguments_from_command()    

    debug = args.debug
    stored_model_path = args.load_path

    if stored_model_path is not None:
        assert stored_model_path[-3:] == '.h5'
        model = load_model(stored_model_path, custom_objects=custom_objects_dict)
    else:
        model = get_model(nb_extra_sdn_units=1,
                          dn_encoder_path="model/densenet_encoder/encoder_model.h5")
    
    
    optimizer = get_optimizer(initial_learnrate=global_initial_learnrate)
    
    print("[db-training] Compiling the model...")
    model.compile(loss=per_pixel_softmax_cross_entropy_loss,
                  optimizer=optimizer,
                  metrics=[IOU])

    # History, Checkpointer, learning-rate scheduler
    callbacks_list = get_callbacks_list() 

    train_generator, val_generator = get_generators(debug=debug)

    print("[db-training] Beginning to fit diamondback model.")


    history_over_epochs = model.fit_generator(
        train_generator,
        steps_per_epoch = 64115 // DEFAULT_BATCH_SIZE,
        #steps_per_epoch=10, # 64115 / DEFAULT_BATCH_SIZE in util/dataflow.py
        epochs=30,
        validation_data=val_generator,
        validation_steps = 2693 // DEFAULT_BATCH_SIZE, # 2693 / DEFAULT_BATCH_SIZE in util/dataflow.py
        callbacks=callbacks_list)
    
    with open("/output/history_over_epochs.pkl", 'wb') as f:
        pickle.dump(history_over_epochs, f)
