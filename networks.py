
#%%

import tensorflow as tf
import numpy as np
import os
import tensorflow.keras.layers as layers
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import normalization_layers
import training_utils

#%%

class pix2pixGAN():

    def __init__(self, critic, generator, critic_optimizer, gen_optimizer, loss_obj, spec_norm, weight_ema = False):
        assert(loss_obj.is_composite()), 'Passed loss function must be GANLossWrapper'
        self.critic = critic
        self.generator = generator
        self.gen_optimizer = gen_optimizer
        self.critic_optimizer = critic_optimizer
        self.buffer = []
        self.loss_fn = loss_obj
        self.spec_norm = spec_norm
        if spec_norm:
            self.gen_spec_norm = training_utils.SpectralNormalization(generator)
            self.critic_spec_norm = training_utils.SpectralNormalization(critic)
        self.apply_ema = False
        if weight_ema:
            self.ema_obj = tf.train.ExponentialMovingAverage(0.5, zero_debias = True)
            self.apply_ema = True

    @staticmethod
    def preprocess_jpg(image):
        image = tf.dtypes.cast(image, 'float32')
        normalized = (image / 127.5) - 1
        return normalized

    @staticmethod
    def to_viewable_image(normalized_image):
        return normalized_image * 0.5 + 0.5

    def get_training_metrics(self):
        return self.loss_fn.get_metrics()

    def predict(self, X):
        return self.generator(X, training = True)
        
    def train_on_buffer(self, train_dataset, steps = 500, buffer_size = 50, critic_train_steps = 1, load_from_checkpoint = False):
        # first pre-populated the buffer
        if len(self.buffer) == 0:
            for (X,Y) in train_dataset.take(buffer_size):
                generated_y = self.generator(X)
                self.buffer.append((X,Y,generated_y))

        step = 0
        for (X,Y) in train_dataset.take(steps):
            for i in tf.range(critic_train_steps - 1):
                self.critic_train_step()
            self.fast_train_step(X, Y, buffer_size)
            step += 1
            tf.print('\rStep ',step,'/',steps, sep='',end ='')

        tf.print('\n',end='')


    def critic_train_step(self):

        if self.spec_norm:
            self.critic_spec_norm.normalize_weights()

        rand_sample = np.random.randint(0, len(self.buffer)) # sample from history
        (Xh, Yh, Yhath) = self.buffer[rand_sample]

        samples = [[Xh, Xh],[Yh,Yhath]]

        samples = tf.concat(samples, axis = -1)

        (n,m,h,w,c) = samples.get_shape()

        samples = tf.reshape(samples, (-1, h,w,c))

        with tf.GradientTape(persistent = True) as tape:
        
            critic_scores = self.critic(samples, training = True)

            scores = tf.split(critic_scores, 2, axis = 0)

            loss_kwargs = {
                'critic_real_score' : scores[0],
                'critic_generated_score' : scores[1],
            }

            critic_loss, critic_metrics = self.loss_fn.critic(**loss_kwargs)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))


    def fast_train_step(self,X,Y, buffer_size):

        if self.spec_norm:
            self.critic_spec_norm.normalize_weights()
            self.gen_spec_norm.normalize_weights()

        rand_sample = np.random.randint(0, len(self.buffer)) # sample from history
        (Xh, Yh, Yhath) = self.buffer[rand_sample]
        
        with tf.GradientTape(persistent = True) as tape:

            new_yhat = self.generator(X, training = True)
            
            samples = [[Xh, Xh, X],[Yh,Yhath, new_yhat]]

            samples = tf.concat(samples, axis = -1)

            (n,m,h,w,c) = samples.get_shape()

            samples = tf.reshape(samples, (-1, h,w,c))
            
            critic_scores = self.critic(samples, training = True)

            scores = tf.split(critic_scores, 3, axis = 0)

            loss_kwargs = {
                'critic_real_score' : scores[0],
                'critic_generated_score' : scores[1],
                'critic_score' : scores[2],
                'generated_y' : new_yhat,
                'real_y' : Y,
                'input_image' : X,
            }

            critic_loss, critic_metrics = self.loss_fn.critic(**loss_kwargs)
            generator_loss, generator_metrics = self.loss_fn.generator(**loss_kwargs)

        generator_grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
        
        del tape

        self.buffer.insert(0, (X,Y,new_yhat))
        if len(self.buffer) > buffer_size:
            self.buffer.pop()

        if self.apply_ema:
            self.ema_obj.apply(self.generator.trainable_weights)
            for weight in self.generator.trainable_weights:
                weight.assign(self.ema_obj.average(weight))
        
    def generate_image(self, X,Y):
    
        m = X.shape[0]

        prediction = self.predict(X)
        joined = np.concatenate([X,Y,prediction], axis = 2)
        if m == 1:
            joined = np.squeeze(joined)
        else:
            joined = np.concatenate(joined[:], axis = 0)
            joined = np.squeeze(joined)
        un_normalized = self.to_viewable_image(joined)
        return un_normalized

#%%

class SelfAttnLayer(tf.keras.layers.Layer):

    def __init__(self, k = 8, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        (m, h, w, nc) = input_shape
        assert(nc // self.k > 0)
        self.flattener = tf.keras.layers.Reshape((h*w, -1))
        self.deflattener = tf.keras.layers.Reshape((h, w, nc))
        self.convQ = tf.keras.layers.Conv2D(nc//self.k, (1,1), padding = 'SAME', use_bias = False)
        self.convK = tf.keras.layers.Conv2D(nc//self.k, (1,1), padding = 'SAME', use_bias = False)
        self.convV = tf.keras.layers.Conv2D(nc, (1,1), padding = 'SAME', use_bias = False)
        self.gamma = tf.Variable(0., dtype = 'float32', trainable = True, name = 'gamma')

    def call(self, X):

        Q = self.convQ(X)
        K = self.convQ(X)
        V = self.convV(X)

        Q = self.flattener(Q)
        K = self.flattener(K)
        V = self.flattener(V)

        energies = tf.matmul(Q, K, transpose_b = True)

        alphas = tf.nn.softmax(energies, axis = -1)

        o = tf.matmul(alphas, V)

        o_2D = self.deflattener(o)
        
        bypass = self.gamma * o_2D + X

        return bypass

## UNET Generator

def downsample_layer(num_filters, filter_size, kernel_init):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filters, filter_size, strides = 2, padding = 'same', 
                        use_bias = False, kernel_initializer = kernel_init),
        normalization_layers.InstanceNormalization(),
        layers.LeakyReLU(),
    ])

def upsample_layer(num_filters, filter_size, kernel_init, input_shape, apply_dropout = False):
    (m, h, w, nc) = input_shape
    X = tf.keras.Input(shape = (h, w, nc))
    resized = layers.UpSampling2D(2)(X)
    conv_X = layers.Conv2D(num_filters, filter_size, strides = 1, use_bias = False,
        padding = 'same', kernel_initializer = kernel_init)(resized)
    conv_X = normalization_layers.InstanceNormalization()(conv_X)
    if apply_dropout:
        conv_X = layers.Dropout(0.5)(conv_X)
    output = layers.LeakyReLU()(conv_X)

    return tf.keras.Model(X, output)

def unet_connection(X, num_filters, filter_size, layer_num, max_filters, kernel_init, sa_layers):

    downsample_filters = min(num_filters, max_filters)
    upsample_filters = min(num_filters//2, max_filters)
    
    if layer_num in sa_layers:
        X = SelfAttnLayer()(X)

    Y = downsample_layer(downsample_filters, filter_size, kernel_init)(X)

    if layer_num > 1:
        u_return = unet_connection(Y, num_filters * 2, filter_size, layer_num - 1,  max_filters, kernel_init, sa_layers)
    else:
        return upsample_layer(upsample_filters, filter_size, kernel_init, Y.get_shape(), True)(Y)

    cat = tf.keras.layers.Concatenate()([Y, u_return])
    
    output = upsample_layer(upsample_filters, filter_size, kernel_init, cat.get_shape(), apply_dropout = layer_num <= 3)(cat)

    if layer_num in sa_layers:
        output = SelfAttnLayer()(output)

    return output
    

def UNET(input_shape, num_layers, filter_size = 4, max_filters = 512, initial_filters = 64, output_channels = 3, use_SA = True, kernel_init = tf.random_normal_initializer(0., 0.02)):

    assert(num_layers > 2), 'Com\'mon this has to be a U-net'

    if use_SA:
        sa_layers = [num_layers - 3, num_layers - 4]
    else:
        sa_layers = []

    X = tf.keras.Input(shape = input_shape)

    u_output = unet_connection(X, initial_filters, filter_size, num_layers - 1, max_filters, kernel_init, sa_layers)

    to_pixel = tf.keras.layers.Conv2D(output_channels, filter_size, padding = 'same', 
                kernel_initializer = kernel_init,
                activation = 'tanh')(u_output)

    return tf.keras.Model(X, to_pixel, name = "UNET")


u = UNET((128,128,3),8)

tf.keras.utils.plot_model(u, 'generator.png', show_shapes = True)

#%%
# WCGAN-GP PatchGAN Critic

def PatchGAN(activation_type = 'sigmoid'):
    kernel_init = tf.random_normal_initializer(0., 0.02)
    conv_layer_kwargs = {
            'padding' : 'same',
            'use_bias' : False,
            'kernel_initializer' : kernel_init
    }
    return tf.keras.Sequential([
        layers.Conv2D(64, 4, 2, **conv_layer_kwargs),
        #normalization_layers.InstanceNormalization(),
        layers.LeakyReLU(),
        layers.Conv2D(128, 4, 2, **conv_layer_kwargs),
        normalization_layers.InstanceNormalization(),
        layers.LeakyReLU(),
        layers.Conv2D(256, 4, 2, **conv_layer_kwargs),
        normalization_layers.InstanceNormalization(),
        layers.LeakyReLU(),
        layers.ZeroPadding2D(),
        layers.Conv2D(512, 4, 1, use_bias=False,kernel_initializer= kernel_init),
        normalization_layers.InstanceNormalization(),
        layers.LeakyReLU(),
        layers.ZeroPadding2D(),
        layers.Conv2D(1, 4, 1, kernel_initializer = kernel_init),
        layers.Flatten(),
        layers.Activation(activation_type)
    ], name = 'PatchGAN')

def ConditionalPatchGAN(image_shape = (256,256,3), activation_type = 'sigmoid'):

    X = layers.Input(shape = image_shape, name = 'Conditioned_Image')

    critic_score = PatchGAN(activation_type)(X)

    ave_score = tf.reduce_mean(critic_score, axis = -1)

    critic_model = tf.keras.Model(inputs = [X], outputs = ave_score)

    return critic_model