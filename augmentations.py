import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations

# Contrastive Loss Function
def SimCLRLoss(features, labels, temperature = 0.07, denom_mode = 'ALL'):
    '''
    Computes SimCLRLoss as defined in https://arxiv.org/pdf/2004.11362.pdf
    '''
    batch_size = features.shape[0]
    if (features.shape[0] != labels.shape[0]):
        raise ValueError('Error in SIMCLRLOSS: Number of labels does not match number of features')

    # Generates mask indicating what samples are considered pos/neg
    positive_mask = tf.equal(labels, tf.transpose(labels))
    negative_mask = tf.logical_not(positive_mask)

    positive_mask = tf.cast(positive_mask, dtype=tf.float32) 
    negative_mask = tf.cast(negative_mask, dtype=tf.float32)
    print("positive, negative mask shape:", positive_mask, negative_mask)

    # Computes dp between pairs
    logits = tf.linalg.matmul(features, features, transpose_b=True)
    temperature = tf.cast(temperature, tf.float32)
    logits = logits / temperature
    print("logits shape:", logits)

    # Subtract largest |logits| elt for numerical stability
    # Simply for numerical precision -> stop gradient
    max_logit = tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
    logits = logits - max_logit

    exp_logits = tf.exp(logits)
    num_positives_per_row = tf.reduce_sum(positive_mask, axis=1)
    print("exp_logits shape:", exp_logits)
    

    if denom_mode == 'ALL':
        denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)
        denominator += tf.reduce_sum(exp_logits * positive_mask, axis = 1, keepdims=True)

    elif denom_mode == 'ONLY_NEGATIVES': # DEBUG !! 
        denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)
    
    print(denominator)
    
#     d_numpy = tf.make_ndarray(tf.make_tensor_proto(denominator))

#     if np.any(np.equal(d_numpy, 0)):
#         return 5 

    # Compute L OUTSIDE -> defined in eq 2 of paper
    log_probs = (logits - tf.math.log(denominator)) * positive_mask
    # log_probs = tf.math.log(exp_logits * positive_mask / denominator)
    log_probs = tf.reduce_sum(log_probs, axis=1)
    log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
        
    loss = -log_probs * temperature
    loss = tf.reduce_mean(loss, axis=0)
    return loss

# Contrastive Loss Function
def SimCLRLossComposite(features, labels=None, temperature = 0.07, denom_mode = 'ALL'):
    '''
    Computes SimCLRLoss as defined in https://arxiv.org/pdf/2004.11362.pdf
    If supervised -> features = (nxd), labels = (nx1) 
    If unsupervised -> features = [hidden1, hidden2], labels = None
    '''
    # Defined labels -> assumes fully supervised contrastive learning 
    if labels is not None: 
        if (features.shape[0] != labels.shape[0]):
            raise ValueError(f"IN SIMCLRLOSS: Num labels '{labels.shape[0]}' must match num features '{features.shape[0]}'")
        
        # Fully supervised -> no augmentations
        hidden1 = hidden2 = features 
        
        # Positive/Negative masks defined soley on labels
        positive_mask = tf.equal(labels, tf.transpose(labels))
        negative_mask = tf.logical_not(positive_mask)
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        negative_mask = tf.cast(negative_mask, dtype=tf.float32)

    # Undefined labels -> assumes unsupervised contrastive learning. 
    else:
        # Assumes can upack features to find hidden representations 
        hidden1, hidden2 = features 
        positive_mask = tf.eye(hidden1.shape[0])
        negative_mask = tf.subtract(1, positive_mask)
        
    # Computes dp between pairs
    logits = tf.linalg.matmul(hidden1, hidden2, transpose_b=True)
    temperature = tf.cast(temperature, tf.float32)
    logits = logits / temperature

    # Subtract largest |logits| elt for numerical stability
    # Simply for numerical precision -> stop gradient
    max_logit = tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
    logits = logits - max_logit

    exp_logits = tf.exp(logits)
    num_positives_per_row = tf.reduce_sum(positive_mask, axis=1)

    if denom_mode == 'ALL':
        denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)
        denominator += tf.reduce_sum(exp_logits * positive_mask, axis = 1, keepdims=True)
    elif denom_mode == 'ONE_POSITIVE': # DEBUG !! 
        denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)

    # Compute L OUTSIDE as defined in eq 2 of paper
    log_probs = (logits - tf.math.log(denominator)) * positive_mask
    log_probs = tf.reduce_sum(log_probs, axis=1)
    log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)

    loss = -log_probs * temperature
    loss = tf.reduce_mean(loss, axis=0)
    return loss


def mse_loss(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

def reco_loss(inputs, outputs):
    # reshape inputs and outputs to manipulate p_t, n, phi values
    inputs = tf.reshape(inputs, (-1,19,3,1))
    outputs = tf.reshape(outputs, (-1,19,3,1))

    # impose physical constraints on phi+eta for reconstruction
    tanh_outputs = tf.math.tanh(outputs)
    outputs_phi = math.pi*tanh_outputs
    outputs_eta_egamma = 3.0*tanh_outputs
    outputs_eta_muons = 2.1*tanh_outputs
    outputs_eta_jets = 4.0*tanh_outputs
    outputs_eta = tf.concat(
        [outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:],
         outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    outputs = tf.concat([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)

    # zero features -> no particles. Disgard loss @ those values through masking
    inputs = tf.squeeze(inputs, -1)
    mask = tf.math.not_equal(inputs, 0)
    mask = tf.cast(mask, tf.float32)
    outputs = outputs * mask

    # returns mse loss between reconstruction and real values
    reconstruction_loss = mse_loss(tf.reshape(inputs, (-1,57)), tf.reshape(outputs, (-1,57)))
    return tf.math.reduce_mean(reconstruction_loss)






