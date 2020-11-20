# Customary Imports:
import tensorflow as tf
from tensorflow.keras import Model

###################################################################################################
'''
MODEL DEFINITION:
GAN MODEL
'''
class GAN(tf.keras.Model):
    def __init__(self,
                 discriminator,
                 generator,
                 discriminator_extra_steps=1,
                 generator_extra_steps=1):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps

    def compile(self,
                d_optimizer,
                g_optimizer,
                loss_fn,
                recon_loss_fn,
                loss_weights=[1,1,1],
                **kwargs):
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.recon_loss_fn = recon_loss_fn
        self.loss_weights = loss_weights

    def call(self, inputs):
        return self.generator(inputs)

    def train_step(self, data):
        input_images, real_images = data

        batch_size = tf.shape(real_images)[0]
        generated_images = self.generator(input_images)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)),
                            tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, predictions)
                d_loss = d_loss*self.loss_weights[0]
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        for i in range(self.g_steps):
            with tf.GradientTape() as tape:
                generated_images = self.generator(input_images)
                predictions = self.discriminator(generated_images)
                g_loss = self.loss_fn(tf.zeros((batch_size, 1)), predictions)
                recon_loss = self.recon_loss_fn(real_images, generated_images)
                loss = g_loss*self.loss_weights[1] + recon_loss*self.loss_weights[2]
            grads = tape.gradient(loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.compiled_metrics.update_state(real_images, generated_images)
        loss_dict = {'d_loss': d_loss, 'g_loss': g_loss, 'recon_loss': recon_loss}
        for k,v in loss_dict.items(): self.add_metric(v, name=k)
        return {m.name: m.result() for m in self.metrics}

###################################################################################################
'''
FUNCTION TO INSTANTIATE MODEL:
'''
def getModel(discriminator, generator,
             discriminator_extra_steps=1,
             generator_extra_steps=1):

    model = GAN(discriminator, generator,
                discriminator_extra_steps,
                generator_extra_steps)
    return model
getModel.__name__ = 'GAN_Model'
