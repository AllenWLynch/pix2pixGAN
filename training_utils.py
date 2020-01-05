
#%%

import tensorflow as tf


class SpectralNormalization():

    def __init__(self, model):
        #initialize U, filtering out bias layers and other 1-D vector layers
        self.normalize_layers = set([
            layer.name for layer in model.trainable_weights if len(layer.get_shape()) > 1
        ])
        self.U = [
            tf.random.normal((kernel.get_shape()[-1], 1))
            for kernel in model.trainable_weights if kernel.name in self.normalize_layers
        ]
        self.model = model

    def l2_normalize(self, x):
        return x/tf.linalg.norm(x)

    def normalize_weights(self):

        scalars = []
        i = 0
        for kernel in self.model.trainable_weights:
            if kernel.name in self.normalize_layers:
                Kf = tf.reshape(kernel, (kernel.get_shape()[-1], -1))
                vl = self.l2_normalize(tf.matmul(Kf, self.U[i], transpose_a = True))
                ul = self.l2_normalize(tf.matmul(Kf, vl))
                self.U[i] = ul
                norm_scalar = tf.matmul(tf.matmul(ul, Kf, transpose_a = True), vl)
                kernel.assign(tf.divide(kernel, norm_scalar))
                i += 1

'''
#%%
class SpectralNormalization():

    def __init__(self, model):
        #initialize U, filtering out bias layers and other 1-D vector layers
        self.normalize_layers = set([
            layer.name for layer in model.trainable_weights if len(layer.get_shape()) > 1
        ])
        self.U = [
            tf.random.normal((kernel.get_shape()[-1], 1))
            for kernel in model.trainable_weights if kernel.name in self.normalize_layers
        ]
        self.model = model

    @staticmethod
    def l2_normalize(x):
        return x/tf.linalg.norm(x)

    @staticmethod
    def calc_V(Kf, u):
        return SpectralNormalization.l2_normalize(tf.matmul(Kf, u, transpose_a = True))

    @staticmethod
    def calc_U(Kf, v):
        return SpectralNormalization.l2_normalize(tf.matmul(Kf, v))

    def calc_normScalar(Kf, u, v):
        return tf.matmul(tf.matmul(u, Kf, transpose_a = True), v)

    def normalize_weights(self):
        #print(list(zip(self.U, [kernel for kernel in model.trainable_weights if kernel.name in self.normalize_layers])))
        uws = [tf.concat([tf.reshape(kernel, (kernel.get_shape()[-1], -1)), u], axis = -1) for (u, kernel) in zip(self.U, [kernel for kernel in model.trainable_weights if kernel.name in self.normalize_layers])]
        print(len(uws))
        return tf.map_fn(
            lambda x : self.calc_V(x[:-1], x[-1]),
            uws
        )


#%%

model = tf.keras.Sequential([
    tf.keras.Input((30,30,3)),
    tf.keras.layers.Conv2D(15, 3),
    tf.keras.layers.Conv2D(30, 3),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)])

model.summary()

# %%

n = SpectralNormalization(model)
n.normalize_weights()

# %%
'''