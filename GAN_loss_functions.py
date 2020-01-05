#%%
import tensorflow as tf
import numpy as np
import os

#%%
class GANLoss():

    def __init__(self):
        self.name = 'GAN loss'

    def critic(self, **kwargs):
        return np.float32(0.), []

    def generator(self, **kwargs):
        return np.float32(0.), []

    def get_metrics(self):
        return {}

    def is_composite(self):
        return False

class Wasserstein_GP(GANLoss):

    def __init__(self, grad_lambd = 10):
        self.lambd = grad_lambd
        self.name = 'Wasserstein loss'

    def critic(self, **kwargs):

        generated_y = kwargs['generated_y']
        real_y = kwargs['real_y']
        critic_generated_score = kwargs['critic_generated_score']
        critic_real_score = kwargs['critic_real_score']
        X = kwargs['input_image']
        critic = kwargs['critic']
        
        m = real_y.shape[0]

        epsilon = tf.random.uniform((m,1,1,1))
        y_hat = tf.multiply(epsilon, generated_y) + tf.multiply(1 - epsilon, real_y)

        with tf.GradientTape() as gradient_penalty_tape:
            gradient_penalty_tape.watch(y_hat)
            critic_prediction = critic((X, y_hat))
                
        dCritic_dyhat = gradient_penalty_tape.gradient(critic_prediction, y_hat)

        gradient_penalty = self.lambd * tf.square(tf.norm(dCritic_dyhat) - 1)

        EM_loss = tf.reduce_mean(critic_generated_score) - tf.reduce_mean(critic_real_score)
        
        critic_loss = EM_loss + gradient_penalty

        return critic_loss, [critic_loss, EM_loss, gradient_penalty]

    def generator(self, **kwargs):

        critic_scores = kwargs['critic_generated_score']
        adverserial_loss = -1 * tf.reduce_mean(critic_scores)
        return adverserial_loss, [adverserial_loss]

    def get_metrics(self):
        return {
            'generator':('Generator adverserial loss',),
            'critic' : ('Critic loss', 'Earth-mover loss','Gradient penalty',)
        }
    
class L1_Loss(GANLoss):

    def __init__(self, lambd):
        self.lambd = lambd
        self.name = 'L1-norm loss'

    def get_metrics(self):
        return {
            'generator':('L1-norm loss',)
        }

    def generator(self, **kwargs):

        real_y, generated_y = kwargs['real_y'], kwargs['generated_y']

        L1_Loss = tf.multiply(self.lambd,tf.reduce_mean(tf.abs(real_y - generated_y)))
        return L1_Loss, [L1_Loss]


class OLD_Least_Square_Loss(GANLoss):

    def __init__(self, label_smoothing = False):
        self.name = 'Least-square loss'
        self.correct_label = 0.9 if label_smoothing else 1.0

    def generator(self, **kwargs):
        critic_scores = kwargs['critic_score']
        #E[(D(G(X)) - 1)^2]
        least_square_loss = tf.reduce_mean(tf.square(critic_scores - 1.0))
        #ave_gen_rating = tf.reduce_mean(critic_scores)
        return least_square_loss, (least_square_loss,)
    
    def critic(self, **kwargs):
        critic_real_score, critic_generated_score = kwargs['critic_real_score'], kwargs['critic_generated_score']
        #E[(D(y) - 1)^2] + [(D(G(X))^2] 
        real_loss = tf.reduce_mean(tf.square(critic_real_score - self.correct_label))
        least_square_loss =  real_loss + tf.reduce_mean(tf.square(critic_generated_score - 0.))

        return least_square_loss, (least_square_loss, tf.reduce_mean(critic_real_score), tf.reduce_mean(critic_generated_score))
    
    def get_metrics(self):
        return {
            'generator' : ('Generator least-squares loss',),
            'critic' : ('Critic least-squares loss','Real sample rating','Generated sample rating')
        }

class Least_Square_Loss(GANLoss):

    def __init__(self, label_smoothing):
        self.name = 'Least-square loss'
        self.label_smoothing = label_smoothing

    def generator(self, **kwargs):
        loss = self.square_error(1.0, kwargs['critic_score'])
        return loss, (loss,)

    def critic(self, **kwargs):
        real_scores, fake_scores = kwargs['critic_real_score'], kwargs['critic_generated_score']
        loss = self.square_error(0.9 if self.label_smoothing else 1.0, real_scores) + self.square_error(0.0, fake_scores)
        return loss, (loss, tf.reduce_mean(real_scores), tf.reduce_mean(fake_scores))
    
    @staticmethod
    def square_error(target, score):
        return tf.reduce_mean(tf.square(score - target))

    def get_metrics(self):
        return {
            'generator' : ('Generator least-squares loss',),
            'critic' : ('Critic least-squares loss','Real sample rating','Generated sample rating')
        }

class CompositeLossWrapper(GANLoss):
    
    def __init__(self, losses, summary_writer):
        self.losses = losses
        self.num_losses = len(losses)
        self.summary_writer = summary_writer
        self.generator_steps = 0
        self.critic_steps = 0

    def generator(self, **kwargs):
        step_summary = []
        total_loss = 0.0
        for loss in self.losses:
            summary_loss, metric_vals = loss.generator(**kwargs)
            total_loss = tf.add(total_loss, summary_loss)

            metric_names = loss.get_metrics()
            
            if 'generator' in metric_names:
                metric_names = metric_names['generator']
                assert(len(metric_vals) == len(metric_names))
                step_summary.extend(zip(metric_names, metric_vals))

        with self.summary_writer.as_default():
            tf.summary.scalar('Generator total loss', total_loss, step=self.generator_steps)
            for name, val in step_summary:
                tf.summary.scalar(name, val, step=self.generator_steps)
            
        self.generator_steps += 1

        return total_loss, step_summary

    def critic(self, **kwargs):
        step_summary = []
        total_loss = 0.0
        for loss in self.losses:
            summary_loss, metric_vals = loss.critic(**kwargs)
            total_loss = tf.add(total_loss, summary_loss)

            metric_names = loss.get_metrics()
            if 'critic' in metric_names:
                metric_names = metric_names['critic']
                assert(len(metric_vals) == len(metric_names))
                step_summary.extend(zip(metric_names, metric_vals))

        with self.summary_writer.as_default():
            tf.summary.scalar('Critic total loss', total_loss, step=self.critic_steps)
            for name, val in step_summary:
                tf.summary.scalar(name, val, step=self.critic_steps)
            
        self.critic_steps += 1

        return total_loss, step_summary 

    def is_composite(self):
        return True


# %%
