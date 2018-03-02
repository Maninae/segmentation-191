import keras.backend as K
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Concatenate

if __name__ == "__main__":
    from densenet_encoder.load import load_densenet_encoder_model
    from util import swish
else:
    from model.densenet_encoder.load import load_densenet_encoder_model
    from model.util import swish 

"""
TODO / Experiments List:
------------------------
1. Hierarchical supervision (SDN segmentation)
2. only pass on the new dense features to concat (100 Tiramisu)
3. Swish activation vs. relu (swish)
"""

class DiamondbackModelCreator(object):
    """
    """
    def __init__(self, dn_encoder_path, nb_extra_sdn_units):
        assert type(nb_extra_sdn_units) is int and nb_extra_sdn_units >= 0

        self.input_shape = (224, 224, 3) # Must be (224,224,3) to fit DenseNet encoder
        self.nb_extra_sdn_units = nb_extra_sdn_units

        self.dn_encoder_path = dn_encoder_path
        self.nb_encoder_output_filters = 1056

        self.nb_compression_after_down1 = 384
        self.nb_compression_after_down2 = 512
        self.nb_compression_after_up1   = 384
        self.nb_compression_after_up2   = 256

        self.nb_compression_before_output = self.nb_compression_after_up2 * (1 + nb_extra_sdn_units) // 2

        self.data_format = K.image_data_format()
        assert self.data_format in {'channels_last', 'channels_first'}
        self.concat_axis = -1 if self.data_format == 'channels_last' else 1

        # We can set this variable depending on whether we want to experiment with relu or swish.
        self.activation = 'relu' 
        # self.activation = swish
        # from keras.utils.generic_utils import get_custom_objects
        # get_custom_objects().update({'swish': Activation(swish)})


    def create_diamondback_model(self):
        """
        """
        input_tensor = Input(shape=self.input_shape)
        sdn_output_tensors = []

        x, small_F, big_F, small_H, big_H = self.__first_sdn_unit(input_tensor,
                                                                  self.dn_encoder_path,
                                                                  nb_conv_filters=32)
        sdn_output_tensors.append(x)

        x = self.__compression_layer(x, self.nb_compression_after_down2)

        if self.nb_extra_sdn_units > 0:
            small_H = self.__conv_layer(small_H, nb_filters=384, dropout_rate=0.2)
            big_H = self.__conv_layer(big_H, nb_filters=192, dropout_rate=0.2)
        
        # Add HierSup for 2 F's here

        
        for _ in range(self.nb_extra_sdn_units):
            x, small_F, big_F = self.__sdn_unit(x,
                                                down1_prev_feature_tensor=big_F,
                                                down2_prev_feature_tensor=small_F,
                                                up1_prev_feature_tensor=small_H,
                                                up2_prev_feature_tensor=big_H,
                                                nb_conv_filters=32)
            sdn_output_tensors.append(x)
            # Add HierSup for 2 F's here

        if len(sdn_output_tensors) > 1:
            x = Concatenate(axis=self.concat_axis)(sdn_output_tensors) # e.g. (56, 56, 1536,) for 3 units
        else:
            x = sdn_output_tensors[0]

        output_tensor = self.__last_upsampling_unit(x)

        diamondback = Model(inputs=input_tensor, outputs=output_tensor)

        return diamondback

        

    def __first_sdn_unit(self, x, path_to_dn_encoder, nb_conv_filters):
        """
        """

        # This model was created in densenet_encoder/construct.py
        encoder = load_densenet_encoder_model(path_to_dn_encoder) 

        # This encoder from DenseNet takes the place of 2 downsampling blocks.
        # big encoding tensor: shape (56, 56, 192)
        # small encoding tensor: shape (28, 28, 384)
        # x (before): shape (14, 14, 1056)
        big_enc_tensor, small_enc_tensor, x = encoder(x)

        small_feature_tensor = x
        
        x = self.__up_block(x, small_enc_tensor,
                            nb_convolutions=2,
                            nb_conv_filters=nb_conv_filters,
                            prev_nb_filters=self.nb_encoder_output_filters,
                            nb_comp_filters=self.nb_compression_after_up1)
      
        big_feature_tensor = x

        x = self.__up_block(x, big_enc_tensor,
                            nb_convolutions=2,
                            nb_conv_filters=nb_conv_filters,
                            prev_nb_filters=self.nb_compression_after_up1,
                            nb_comp_filters=self.nb_compression_after_up2)

        return x, small_feature_tensor, big_feature_tensor, small_enc_tensor, big_enc_tensor


    def __sdn_unit(self, x,
                   down1_prev_feature_tensor,
                   down2_prev_feature_tensor,
                   up1_prev_feature_tensor,
                   up2_prev_feature_tensor,
                   nb_conv_filters):
        """
        """

        x = self.__down_block(x, down1_prev_feature_tensor,
                              nb_convolutions=2,
                              nb_conv_filters=nb_conv_filters,
                              nb_comp_filters=self.nb_compression_after_down1)

        x = self.__down_block(x, down2_prev_feature_tensor,
                              nb_convolutions=4,      # To learn more detail at the lowest resolution
                              nb_conv_filters=nb_conv_filters,
                              nb_comp_filters=self.nb_compression_after_down2)
        
        small_feature_tensor = x

        x = self.__up_block(x, up1_prev_feature_tensor,
                            nb_convolutions=2,
                            nb_conv_filters=nb_conv_filters,
                            prev_nb_filters=self.nb_compression_after_down2,
                            nb_comp_filters=self.nb_compression_after_up1)
        
        big_feature_tensor = x

        x = self.__up_block(x, up2_prev_feature_tensor,
                            nb_convolutions=2,
                            nb_conv_filters=nb_conv_filters,
                            prev_nb_filters=self.nb_compression_after_up1,
                            nb_comp_filters=self.nb_compression_after_up2)

        return x, small_feature_tensor, big_feature_tensor


    def __last_upsampling_unit(self, x, dropout_rate=0.2):
        """
        Args:
            x: 
        """

        x = self.__deconv_layer(x, self.nb_compression_before_output, dropout_rate=dropout_rate)
        x = self.__conv_layer(x, self.nb_compression_before_output, dropout_rate=dropout_rate)

        x = self.__deconv_layer(x, self.nb_compression_before_output, dropout_rate=dropout_rate)
        x = self.__conv_layer(x, self.nb_compression_before_output, dropout_rate=dropout_rate)
        
        x = self.__conv_layer(x, 2) # No dropout on final convolution
        # final shape: (None, 224, 224, 2)
        # Ready for softmax + cross entropy loss

        return x


    def __down_block(self, x, prev_feature_tensor, nb_convolutions, 
                     nb_conv_filters, nb_comp_filters, dropout_rate=0.2):
        """
        asdf
        """
        return self.__block(x,
                            prev_feature_tensor,
                            nb_convolutions,
                            nb_conv_filters,
                            nb_comp_filters, 
                            dropout_rate=dropout_rate)


    def __up_block(self, x, prev_feature_tensor, nb_convolutions,
                   nb_conv_filters, nb_comp_filters, prev_nb_filters, dropout_rate=0.2):
        """
        """
        return self.__block(x,
                            prev_feature_tensor,
                            nb_convolutions,
                            nb_conv_filters,
                            nb_comp_filters, 
                            prev_nb_filters=prev_nb_filters, 
                            dropout_rate=dropout_rate)


    def __block(self, x, prev_feature_tensor, nb_convolutions,
                nb_conv_filters, nb_comp_filters, prev_nb_filters=None, dropout_rate=None):
        """
        """

        # This is an upsampling block
        if prev_nb_filters is not None: 
            x = self.__deconv_layer(x, prev_nb_filters // 2, dropout_rate=dropout_rate) # Twice resolution, half filters
        # This is a downsampling block
        else:
            x = MaxPooling2D()(x)

        x = Concatenate(axis=self.concat_axis)([x, prev_feature_tensor])

        #new_filters_list = []
        for _ in range(nb_convolutions):
            new_filters = self.__conv_layer(x, nb_conv_filters, dropout_rate=dropout_rate)
            #new_filters_list.append(new_filters)
            x = Concatenate(axis=self.concat_axis)([x, new_filters])

        #x = Concatenate(axis=self.concat_axis)(new_filters_list)

        x = self.__compression_layer(x, nb_comp_filters)

        return x


    def __conv_layer(self, x, nb_filters, dropout_rate=None):
        ''' 
        Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout.
        Built off of https://github.com/titu1994/DenseNet/blob/master/densenet.py
        
        We will assume data_format is 'channels_last'.

        Args:
            x: Input keras tensor
            nb_filter: number of filters
            dropout_rate: dropout rate
        Returns: keras tensor with batch_norm, swish and convolution2d added
        '''

        x = BatchNormalization(epsilon=1e-5)(x)
        x = Activation(self.activation)(x)
        x = Conv2D(nb_filters, (3,3), kernel_initializer='he_normal', padding='same')(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        
        return x


    def __deconv_layer(self, x, nb_filters, dropout_rate=None):
        """
        Same as the conv layer above, but deconvolution.
        """
        x = BatchNormalization(epsilon=1e-5)(x)
        x = Activation(self.activation)(x)
        x = Conv2DTranspose(nb_filters, (4,4), strides=2, kernel_initializer='he_normal', padding='same')(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
        
        return x


    def __compression_layer(self, x, nb_filters):
        """
        asdf
        """
        x = Activation(self.activation)(x)
        x = Conv2D(nb_filters, (3,3), kernel_initializer='he_normal', padding='same')(x)
        
        return x




if __name__ == "__main__":
    creator = DiamondbackModelCreator(
        dn_encoder_path="densenet_encoder/encoder_model.h5",
        nb_extra_sdn_units=1)
    db = creator.create_diamondback_model()
    db.summary()
