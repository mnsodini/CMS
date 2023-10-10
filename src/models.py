print("Importing from 'models.py'")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

class CVAE(keras.Model):
    '''
    Creates fully supervised CVAE Class 
    Training architecture: input -> latent space μ representation -> Proj(μ) -> contrastive loss 
    '''
    def __init__(self, contrastive_loss, temp = 0.07, latent_dim=8, loss_type='SimCLR', **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = build_encoder(self.latent_dim)
        self.projection_head = build_projection_head(self.latent_dim)
        self.temperature = temp
        self.loss_type = loss_type
        self.contrastive_loss_fn = contrastive_loss
        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_tracker")

    @property
    def metrics(self):
        return [self.contrastive_loss_tracker]

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # Foward pass to create reconstruction + computes loss
            if self.loss_type == 'SimCLR': 
                data, labels = inputs
                z_mean = self.encoder(data, training=True)
                projection = self.projection_head(z_mean, training=True)
                contrastive_loss = self.contrastive_loss_fn(projection, labels=labels, temperature=self.temperature)
                
            if self.loss_type == 'VicReg': 
                z_mean = self.encoder(inputs, training=True)
                projection = self.projection_head(z_mean, training=True)
                contrastive_loss = self.contrastive_loss_fn(projection, projection)
                
        # Apply gradients and update losses
        grads = tape.gradient(contrastive_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {"contrastive_loss": self.contrastive_loss_tracker.result()}
    
    def test_step(self, data): 
        if self.loss_type == 'SimCLR':
            # Unpacks the data
            features, labels = data
            # Computes latent space representation and loss
            latent_rep = self.encoder(features, training=False)
            projection = self.projection_head(latent_rep, training=False)
            valid_loss = self.contrastive_loss_fn(projection, labels=labels, temperature=self.temperature)
            
        if self.loss_type == 'VicReg':
            latent_rep = self.encoder(data, training=False)
            projection = self.projection_head(latent_rep, training=False)
            valid_loss = self.contrastive_loss_fn(projection, projection)
            
        # Updates loss metrics 
        for metric in self.metrics: 
            if metric.name != "contrastive_loss": 
                metric.update_state(valid_loss)
        return {m.name: m.result() for m in self.metrics}
        
    def call(self, inputs): 
        encoded = self.encoder(inputs, training=False)
        projected = self.projection_head(encoded, training=False)
        return projected

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

def build_encoder(latent_dim=8, sampling=False): 
    '''
    Encoder as defined in gitlab: https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae
    '''
    enc_inputs = keras.Input(shape=(57,))
    x = layers.Dense(32)(enc_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    if sampling: 
        # If encoer is used for sampling, forms gaussian and returns μ, σ, and sampled point
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sample_Layer()([z_mean, z_log_var])
        encoder = keras.Model(enc_inputs, [z_mean, z_log_var, z], name="encoder")
        
    else: 
        # If no reconstruction, only returns μ as anomoloies are detected by (μ)**2 values 
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        encoder = keras.Model(enc_inputs, z_mean, name="encoder")
    return encoder

def build_decoder(latent_dim=8): 
    '''
    Decoder as defined in gitlab: https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae 
    '''
    decoder_input = keras.Input(shape=(latent_dim,))
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
    return decoder

def build_projection_head(latent_dim=8): 
    '''
    Build MLP projection head before computing loss as suggested by https://arxiv.org/pdf/2002.05709.pdf
    Disregarded after training
    '''
    projection_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16)(projection_inputs)
    x = layers.LeakyReLU()(x)
    projection = layers.Dense(latent_dim)(x)
    projection_head = keras.Model(projection_inputs, projection, name="projection_head")
    return projection_head

def build_classification_head(latent_dim=8, num_labels=2): 
    '''
    Build DNN classification head to map latent space represetnation to class
    '''
    classification_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64)(classification_inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1)(x)
    x = activations.sigmoid(x)
    
    classifier = keras.Model(classification_inputs, x, name="classifier")
    return classifier

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
        
    
    