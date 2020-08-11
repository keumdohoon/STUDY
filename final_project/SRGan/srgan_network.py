from keras.layers import Dense, Input
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add

#residual blck
def res_block_gen(model, kernel_size, filters, strides):

    gen = model

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    #Using Parametric Relu
    model = PRelu(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding="same")(model)
    model = BatchNormalization(momentum = 0.5)(model)

    model = add([gen, model])

    return model

def up_sampling_block(model, kernel_size, filters, strides):

    #In place of Conv2D and Upsampling2D we can also use Conv2D transpose(both are used for deconvolution)
    #Even we can have our own function for deconvolution(ex, one made in utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size=kernel_size, strides = strides, padding= "same")(model)
    model - -Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model =UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model

def discriminator_block(model, filters, kernel_size, strides):

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momnetum = 0.5)(model)
    model - LeakyReLU(alpha = 0.2)(model)

    return model

#network Architecture is same as given in Paper  https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

        def generator(self):

            gen_input = Input(shape = self.noise_shape)

            model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding ="same")(gen_input)
            model - PRelu(alpha_initializer = "zeros", alpha_regularizer = None, alpha_constraint=None, shared_axes = [1, 2])(model)

            gen_model = model

            #using 16Residual blocks
            for index in range(16):
                model = res_block_gen(model, 3, 64, 1)

            model - Conv2D(filters = 64, kernel_size= 3, strides = 1, padding = "same")(model)
            model = BatchNormalization(momentum = 0.5)(model)
            model = add([gen_model, model])
        
            #using 2 Upsampling Blocks
            for index in range(2):
                model = up_sampling_block(model, 3, 256, 1)

            model  = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding ="same")(model)
            model  = Activation('tahn')(model)

            generator_model = Model(inputs = gen_input, outputs = model)

            return generator_model

#Network Architecture is same as given in Paper   https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):

        self.image_shape = image_shape

    def discriminator(self):

        dis_inpupt = Input(shape = self.image_shape)

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(di_input)
        model = LeakyRelu(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs = dis_input, outputs = model)

        return discriminator_model
    