import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import sys

# Define Contrastive VAE
class CVAE(keras.Model):
    '''
    Creates fully supervised CVAE Class 
    Training architecture: input -> latent space μ representation -> Proj(μ) -> contrastive loss 
    '''
    def __init__(self, encoder, contrastive_loss, temp = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projection_head = build_projection_head()
        self.temperature = temp
        self.contrastive_loss_fn = contrastive_loss
        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_tracker")

    @property
    def metrics(self):
        return [self.contrastive_loss_tracker]

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # Foward pass to create reconstruction + computes loss
            data, labels = inputs
            z_mean = self.encoder(data, training=True)
            projection = self.projection_head(z_mean, training=True)
            contrastive_loss = self.contrastive_loss_fn(projection, labels=labels, temperature=self.temperature)
            
        # Apply gradients and update losses
        grads = tape.gradient(contrastive_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {"contrastive_loss": self.contrastive_loss_tracker.result()}

class VAE(keras.Model):
    '''
    Creates VAE Class as defined/implemented in CMS github 
    '''
    def __init__(self, encoder, decoder, reco_loss, kl_loss, alpha: float = 1.0, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reco_loss_fn = reco_loss
        self.kl_loss_fn = kl_loss
        self.reco_scale = alpha * (1 - beta)
        self.kl_scale = beta

        # 3 losses: reconstruction, kl, and total loss
        self.reco_loss_tracker  = keras.metrics.Mean(name="reco_tracker")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_tracker")
        self.total_loss_tracker = keras.metrics.Mean(name="total_tracker")
        
    @property
    def metrics(self):
        return [self.reco_loss_tracker, self.kl_loss_tracker, self.total_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Foward pass to create reconstruction
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            # Computes loss from reconstruction vs original data
            reconstruction_loss = self.reco_scale * self.reco_loss_fn(data, reconstruction)
            kl_loss = self.kl_scale * self.kl_loss_fn(z_mean, z_log_var)
            total_loss = reconstruction_loss + kl_loss

        # Apply gradients and update losses
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reco_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reco_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result()}

def build_encoder(): 
    '''
    Encoder as defined in gitlab: https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae
    Only returns μ since anomoloy detected by (μ)**2 values 
    '''
    latent_dim = 8
    encoder_inputs = keras.Input(shape=(57,))

    x = layers.Dense(32)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    encoder = keras.Model(encoder_inputs, z_mean, name="encoder")
    return encoder

def build_projection_head(): 
    '''
    Build MLP projection head before computing loss as suggested by https://arxiv.org/pdf/2002.05709.pdf
    Disregarded after training
    '''
    projection_inputs = keras.Input(shape=(8,))
    x = layers.Dense(16)(projection_inputs)
    x = layers.LeakyReLU()(x)
    projection = layers.Dense(8)(x)
    projection_head = keras.Model(projection_inputs, projection, name="projection_head")
    return projection_head
    