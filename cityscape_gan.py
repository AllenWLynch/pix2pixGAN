#%%

import tensorflow as tf
import numpy as  np
import matplotlib.pyplot as mpl
import os
import imageio

import paired_image_lib
import networks
import importlib
import GAN_loss_functions as losses

#%%
importlib.reload(paired_image_lib)
importlib.reload(networks)
importlib.reload(losses)

#%%

#_________GET DATA GENERATORS___________

TRAIN_DIR = './cityscapes_data/train/'
TEST_DIR = './cityscapes_data/val/'

train_data = paired_image_lib.paired_images_training_datagen(TRAIN_DIR, 
    networks.pix2pixGAN.preprocess_jpg, batch_size=4)
test_data = paired_image_lib.paired_images_testing_datagen(TEST_DIR, 
    networks.pix2pixGAN.preprocess_jpg, batch_size=3)

#%%
#_________INSTANTIATE MODEL___________

INPUT_SHAPE = (256,256,3)
NUM_LAYERS = 8 # for the UNET

generator = networks.UNET(INPUT_SHAPE, NUM_LAYERS)
critic = networks.ConditionalPatchGAN(INPUT_SHAPE,'sigmoid')

GENERATOR_LEARNING_RATE = 0.0002
CRITIC_LEARNING_RATE = GENERATOR_LEARNING_RATE / 2.0


generator_optimizer = tf.keras.optimizers.Adam(GENERATOR_LEARNING_RATE)
critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LEARNING_RATE)

#%%
import datetime
log_writer = tf.summary.create_file_writer(
        os.path.join(
            'fixed_logs',
            "fit/",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ))

# Set up losses
leastsquares_loss = losses.Least_Square_Loss(label_smoothing = True)
l1_loss = losses.L1_Loss(1.)
loss_obj = losses.CompositeLossWrapper([leastsquares_loss, l1_loss], log_writer)

p2p_GAN = networks.pix2pixGAN(critic, generator, generator_optimizer, critic_optimizer, loss_obj)

#%%
#_________SET UP TRAINING LOGS___________

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 critic_optimizer=critic_optimizer,
                                 generator=generator,
                                 critic=critic)

def generate_image(X,Y):
    
    m = X.shape[0]

    prediction = GAN.predict(X)
    joined = np.concatenate([X,Y,prediction], axis = 2)
    if m == 1:
        joined = np.squeeze(joined)
    else:
        joined = np.concatenate(joined[:], axis = 0)
        joined = np.squeeze(joined)
    un_normalized = GAN.to_viewable_image(joined)
    return un_normalized

#%%
#______TRAINING PARAMETERS___________

EPOCHS = 100
STEPS_PER_EPOCH = 4000
NUM_CHECKPOINTS = 10
GENERATOR_OUTPUT_EXAMPLES_DIR = './generator_examples/'

checkpoint_every = int(np.ceil(EPOCHS/NUM_CHECKPOINTS))
checkpoint_epochs = set([EPOCHS] + list(range(checkpoint_every,EPOCHS-checkpoint_every + 2,checkpoint_every)))

print('Checkpoints at epochs:', ', '.join([str(ep) for ep in sorted(checkpoint_epochs)]))

#%%

#_______TRAINING LOOP_________

try:
    for epoch in range(1, EPOCHS + 1):
        print('EPOCH ', epoch)
        
        GAN.train_on_buffer(train_data, steps = STEPS_PER_EPOCH)

        if epoch in checkpoint_epochs:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print('Saved Checkpoint!')

        valX, valY = next(iter(test_data))
        example = generate_image(valX,valY)
        example = example * 255
        example = example.astype(np.uint8)
        imageio.imsave(os.path.join(GENERATOR_OUTPUT_EXAMPLES_DIR, 'epoch_{}.jpg'.format(str(epoch))),example)

except KeyboardInterrupt:
    print('Training interupted!')


# %%
