


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as  np
import datetime

import paired_image_lib
import networks
import GAN_loss_functions as losses
import argparse

def extant_directory(x):
    if not os.path.isdir(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=extant_directory, help='Path to training data directory')
parser.add_argument('validation_dir', type=extant_directory, help='Path to testing data directory')
parser.add_argument('checkpoint_dir', type = extant_directory, help='Directory for depositing training checkpoints')
parser.add_argument('epochs', type=int, help='Number of epochs to train for.')
parser.add_argument('-s', '--steps_per_epoch',type=int,help='Number of iterations to run in each epoch.',default = 4000)
parser.add_argument('-lr','--learning_rate', type = float, default=0.0002,help='Generator learning rate')
parser.add_argument('--input_shape',type=int, nargs=3,help='size of input images')
parser.add_argument('--batch_size',type=int, default=1,help='Batch size, recommended: 1')
parser.add_argument('--generator_steps',type=int,default=1,help='Number of generator steps per training step, recommended: 1-2')
parser.add_argument('--critic_steps',type=int,default=1,help='Number of critic steps per training step, recommended: 1-2')
parser.add_argument('--checkpoint_every', type=int,default=10,help='Number of epochs between every checkpoint')
parser.add_argument('--load_from_checkpoint',action='store_true',help='Load models from most recent checkpoint in checkpoint directory.')
parser.add_argument('--MAE_lambda', type = float, default = 1.0, help='Lambda for L1-norm loss')
parser.add_argument('--label_smoothing',action='store_true')
parser.add_argument('--logdir',type=extant_directory,default='./logs/',help='Directory to deposit loss data')
parser.add_argument('--buffer_size', type = int, default=1, help='Buffer size for critic')
parser.add_argument('--critic_lr_factor', type = float, default=0.25, help='Multiplier for critic learning rate relative to generator\'s')
parser.add_argument('--spectral_norm', action = 'store_true', help = 'Apply spectral normalization to weights of generator and discriminator')
parser.add_argument('--generator_ema', action = 'store_true', help = 'Apply EMA to generator weights to stabilize training.')

if __name__ == "__main__": 

    args = parser.parse_args()

    if args.input_shape is None:
        INPUT_SHAPE = (256,256,3)
    else:
        INPUT_SHAPE = args.input_shape

    #load training/testing data
    train_data = paired_image_lib.paired_images_training_datagen(args.train_dir, 
        networks.pix2pixGAN.preprocess_jpg, batch_size=args.batch_size, image_size= INPUT_SHAPE[0])
    test_data = paired_image_lib.paired_images_testing_datagen(args.validation_dir, 
        networks.pix2pixGAN.preprocess_jpg, batch_size=3, image_size= INPUT_SHAPE[0])

    #allocate the models
    NUM_LAYERS = int(np.log(INPUT_SHAPE[0])/np.log(2)) + 1# for the UNET
    print('Generator is U-net with {} layers.'.format(str(NUM_LAYERS)))
    generator = networks.UNET(INPUT_SHAPE, NUM_LAYERS)
    critic = networks.ConditionalPatchGAN(INPUT_SHAPE,'sigmoid')

    # set up optimizers
    GENERATOR_LEARNING_RATE = args.learning_rate
    CRITIC_LEARNING_RATE = GENERATOR_LEARNING_RATE * args.critic_lr_factor

    print('Generator LR:',GENERATOR_LEARNING_RATE, '\nCritic LR:',CRITIC_LEARNING_RATE)
    
    generator_optimizer = tf.keras.optimizers.Adam(GENERATOR_LEARNING_RATE, beta_1=0.5)
    critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LEARNING_RATE, beta_1=0.5)

    #set up checkpointing
    CHECKPOINT_DIR = args.checkpoint_dir

    checkpoint = tf.train.Checkpoint(generator=generator,
                                    critic=critic)

    if args.load_from_checkpoint:
        try:
            checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).assert_consumed()
        except Exception as err:
            print('Failed to load models from checkpoint!')
            raise err

    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

    # Set up tensorboard support 

    log_writer = tf.summary.create_file_writer(
            os.path.join(
                args.logdir,
                "fit/",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ))

    # Set up losses
    leastsquares_loss = losses.Least_Square_Loss(label_smoothing = args.label_smoothing)
    l1_loss = losses.L1_Loss(args.MAE_lambda)
    loss_obj = losses.CompositeLossWrapper([leastsquares_loss, l1_loss], log_writer)

    p2p_GAN = networks.pix2pixGAN(critic, generator, critic_optimizer, generator_optimizer, loss_obj, args.spectral_norm, args.generator_ema)

    print('Successfully initialized model and training data.')
    print('Point tensorboard to {} to monitor training.'.format(args.logdir))

    arg_string = '\n'.join([str(argname) + ': ' + str(value) for argname, value in vars(args).items()])
    
    with log_writer.as_default():
        tf.summary.text('Run Parameters', arg_string, step = 0)
    
    try:
        for epoch in range(args.epochs):
            print('EPOCH ', epoch + 1)
            
            p2p_GAN.train_on_buffer(train_data, steps = args.steps_per_epoch, buffer_size= args.buffer_size,
                                critic_train_steps=args.critic_steps, load_from_checkpoint=args.load_from_checkpoint)

            if epoch % args.checkpoint_every == 0 and (epoch > 0 or not args.load_from_checkpoint):
                manager.save()
                print('Saved Checkpoint!')

            valX, valY = next(iter(test_data))
            example = p2p_GAN.generate_image(valX,valY)
            example = np.expand_dims(example, 0)
            with log_writer.as_default():
                tf.summary.image('generated_image.jpg'.format(str(epoch+1)), example, step=epoch)

    except KeyboardInterrupt:
        print('Training interupted!')
        user_input = ''
        while not (user_input == 'y' or user_input == 'n'):
            user_input = input('Save model\'s current state?: [y/n]')
        if user_input == 'y':
            manager.save()
            print('Saved checkpoint!')
        
    else:
        print('Training complete! Saving final model.')
        manager.save()