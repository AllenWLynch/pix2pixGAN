#%%
import tensorflow as tf
import numpy as np
import os

def splitXY(raw_image):

    pair = tf.stack((raw_image[:,:256,:], raw_image[:,256:,:]), axis = 0)

    return pair

def size_up(stack, upscale_size):
    return tf.image.resize(stack, [upscale_size, upscale_size], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def crop_images(stack, final_size):
    (m, _, _, c) = stack.get_shape()
    return tf.image.random_crop(stack, size = [2, final_size, final_size, 3])

def flip_images(stack):
    if tf.random.uniform(()) > 0.5:
        return tf.image.flip_left_right(stack)
    return stack

def unstack(stack):
    Y,X = tf.split(stack, 2, axis = 0)
    return tf.squeeze(X), tf.squeeze(Y)

@tf.function
def prepare_image(image_file, pre_processing_technique, final_size = 256, training = True):

    image = tf.io.read_file(image_file)
    
    sample = tf.image.decode_jpeg(image)

    sample = splitXY(sample)

    sample = size_up(sample, int(1.2 * final_size))

    sample = crop_images(sample, final_size)

    if training:
        sample = flip_images(sample)

    sample = pre_processing_technique(sample)

    (X,Y) = unstack(sample)

    return X,Y

def paired_images_training_datagen(train_dir, pre_processing_technique, image_size = 256, buffer_size = 400, batch_size = 1):
    train_data = tf.data.Dataset.list_files(os.path.join(train_dir,'*.jpg'))
    train_data = train_data.map(lambda x : prepare_image(x, pre_processing_technique, image_size, training = True), 
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size)
    train_data = train_data.repeat()
    train_data = train_data.batch(batch_size)
    return train_data


def paired_images_testing_datagen(test_dir, pre_processing_technique, image_size = 256, batch_size = 1):
    test_data = tf.data.Dataset.list_files(os.path.join(test_dir,'*.jpg'))
    test_data = test_data.map(lambda x : prepare_image(x, pre_processing_technique, image_size, training = False), 
                            num_parallel_calls = tf.data.experimental.AUTOTUNE)
    test_data = test_data.repeat()
    test_data = test_data.batch(batch_size)
    return test_data
    