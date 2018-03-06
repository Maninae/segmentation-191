import numpy as np

from util.dataflow import preprocess_mask, get_generator
from util.dataflow import DEFAULT_BATCH_SIZE

from util.pathutil import train_img_dir_wrapper, train_mask_dir_wrapper, \
                          val_img_dir_wrapper, val_mask_dir_wrapper, \
                          train_img_debug, train_mask_debug, \
                          val_img_debug, val_mask_debug

from model.diamondback import DiamondbackModelCreator
from model.loss import per_pixel_softmax_cross_entropy_loss, IOU

from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, SGD


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

    # From AIH:
    #savepath = "/output/model/%s/%s_ep{epoch:02d}-vloss={val_loss:.4f}-vbacc={val_binary_accuracy:.4f}.h5" % (sensor_id, model_base_name)
    
    #savepath = "model/weights/diamondback_ep{epoch:02d}-vloss={val_loss:.4f}-tloss={train_loss:.4f}.h5"
    savepath = "/output/diamondback_ep{epoch:02d}-vloss={val_loss:.4f}-IOU={IOU:.4f}.h5"
    checkpointer = ModelCheckpoint(savepath, monitor='val_loss', verbose=1, save_best_only=True)

    def step_decay(nb_epochs, lr=1e-4): # Needs default value for backward TF compatibility
        """ This needs to be in harmony with get_optimizer(initial_learnrate)!
            Also depends on the number epochs we are doing. Right now:
            90 epochs, /10 downscaling at 30, 60.
        """
        if nb_epochs > 60:
            lr /= 100
        elif nb_epochs > 30:
            lr /= 10
        return lr
    lrate_scheduler = LearningRateScheduler(step_decay)
    
    return history, checkpointer, lrate_scheduler


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


if __name__ == "__main__":
    #debug = True if input("Debug? [y/n lowercase]: ") == 'y' else False
    debug = False

    model = get_model(nb_extra_sdn_units=1, dn_encoder_path="model/densenet_encoder/encoder_model.h5")
    optimizer = get_optimizer(initial_learnrate=1e-4)
    
    print("[db-training] Compiling the model...")
    model.compile(loss=per_pixel_softmax_cross_entropy_loss,
                  optimizer=optimizer,
                  metrics=[IOU])

    history, checkpointer, lrate_scheduler = get_callbacks_list() # History, Checkpointer
    callbacks_list = [history, checkpointer, lrate_scheduler]

    train_generator, val_generator = get_generators(debug=debug)

    print("[db-training] Beginning to fit diamondback model.")


    history_over_epochs = model.fit_generator(
        train_generator,
        steps_per_epoch = 64115 // DEFAULT_BATCH_SIZE,
        #steps_per_epoch=10, # 64115 / DEFAULT_BATCH_SIZE in util/dataflow.py
        epochs=2,
        validation_data=val_generator,
        validation_steps = 2693 // DEFAULT_BATCH_SIZE, # 2693 / DEFAULT_BATCH_SIZE in util/dataflow.py
        callbacks=callbacks_list)
    
    with open("/output/history_over_epochs.pkl", 'wb') as f:
        pickle.dump(history_over_epochs, f)
