import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import sys

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

def build_encoder(reconstruction=False): 
    '''
    Encoder as defined in gitlab: https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae
    '''
    latent_dim = 8
    enc_inputs = keras.Input(shape=(57,))
    x = layers.Dense(32)(enc_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    if reconstruction: 
        # If use for reconstruction returns μ, σ, and sampled point
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sample_Layer()([z_mean, z_log_var])
        encoder = keras.Model(enc_inputs, [z_mean, z_log_var, z], name="encoder")
        
    else: 
        # If no reconstruction, only returns μ as anomoloies are detected by (μ)**2 values 
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        encoder = keras.Model(enc_inputs, z_mean, name="encoder")
        return encoder

def build_decoder(): 
    '''
    Decoder as defined in gitlab: https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae 
    '''
    decoder_input = keras.Input(shape=(8,))
    x = layers.Dense(16)(latent_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)   
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(57)(x)
    x = layers.LeakyReLU()(x)
    
    decoder = keras.Model(latent_input, x, name="decoder")
    decoder.summary()

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

def build_classification_head(num_labels): 
    '''
    Build MLP classification head to map latent space represetnation to one of num_label events 
    '''
    latent_inputs = keras.Input(shape=(8,))
    classes = layers.Dense(num_labels)(latent_inputs)
    classification_head = keras.Model(latent_inputs, classes, name="classification_head")
    return classification_head

class Sample_Layer(layers.Layer):
    '''
    Builds custom sampling layer from gaussian distribution for VAE reconstruction 
    '''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dims = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dims))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
