# Synch_AE.py
# 2026 Yuta Ushida and Kai Fukami (Tohoku University)

## Authors:
# Yuta Ushida, Daiki Beppu, Yuzuru Kato, and Kai Fukami
## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
#     Ref: Y. Ushida, D. Beppu, Y. Kato, and K. Fukami,
#     “Studying oscillation death in two-dimensional cylinder-airfoil interactions with synchronization-theoretic autoencoder,”
#     in Review
#
# The code is written for educational clarity and not for speed.
# -- version 1: Sep 29, 2026



from tensorflow.keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os, sys
sys.path.append(os.path.abspath('./util'))
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow import keras

ms = StandardScaler()

y_train_lat_cy = np.zeros([3498*5,4])
y_train_z = np.zeros([3498*5,4])

circle_index = np.concatenate([
    np.zeros((3498, 1)),
    np.ones((3498, 1)) * 1,
    np.ones((3498, 1)) * 2,
    np.ones((3498, 1)) * 3,
    np.ones((3498, 1)) * 4
], axis=0).astype(np.float32)

# training_data_cy: aerodynamic coefficients, y_train_lat_cy: dummy training data for latent space, y_train_z: dummy training data for latent space, circle_index: circle indices
X_train, X_test, X_train_lat, X_test_lat, X_train_z, X_test_z, circle_index_train, circle_index_test = train_test_split(
    training_data_cy, 
    y_train_lat_cy, 
    y_train_z,
    circle_index,
    test_size=0.3, 
    random_state=42
)


y_train_list = [X_train, X_train_lat, X_train_z, circle_index_train]
y_test_list  = [ X_test,  X_test_lat,  X_test_z, circle_index_test]

y_train_list = [np.asarray(y).astype(np.float32) for y in y_train_list]
y_test_list  = [np.asarray(y).astype(np.float32) for y in y_test_list]


X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)


dataset = tf.data.Dataset.from_tensor_slices((
    X_train,
    circle_index_train))

dataset_cy = dataset.shuffle(buffer_size=3498*5).batch(3498*5)


val_dataset_cy = tf.data.Dataset.from_tensor_slices(( X_test,circle_index_test))
val_dataset_cy = val_dataset_cy.batch(3498*5)












import tensorflow as tf
from tensorflow import keras

class PhysicsConstrainedModel(keras.Model):
    def __init__(self, encoder, decoder, alpha=1.0, beta=1.0,gamma=1.0, **kwargs):
        super(PhysicsConstrainedModel, self).__init__(**kwargs)
        # self.base_model = base_model
        self.encoder = encoder
        self.decoder = decoder

        
        self.alpha = alpha  # weight of the circle constraint
        self.beta = beta    # weight of the dynamics prediction loss in latent space
        self.gamma = gamma    # weight of the center-of-mass term in latent space



        
        #######################################
        ############wing#######################
        # 1. Circle radii for the first 3 groups (3 kinds)
        # The initial value 0.54 becomes approximately 1.0 after passing through softplus
        self.circle_radii = tf.Variable(
            initial_value=[[0.54]], 
            trainable=False, dtype=tf.float32, name='circle_radii'
        )
        self.circle_origin = tf.Variable(
            initial_value=[[0]], 
            trainable=False, dtype=tf.float32, name='circle_origin'
        )


        # 2. Ellipse coefficients for the next 3 groups (3 kinds x a, b)
        self.ellipse_axes = tf.Variable(
            initial_value=[[-0.2, -0.2], [-0.3, -0.5], [0.1, 0.2], [0.1, 0.2]], 
            trainable=True, dtype=tf.float32, name='ellipse_axes'
        )

        self.omega = tf.Variable(initial_value=[[0.10], [0.54], [0.70], [0.80], [0.90]], trainable=True,name='omega') # constant angular velocity
        self.theta = tf.Variable(initial_value=[[0.10], [0.54], [0.30], [0.15], [0.05]], trainable=True,name='theta') # ellipse rotation
        self.origin = tf.Variable(initial_value=[[0.54,0.30], [0.70,0.40], [0.80,0.80], [0.90,0.90]], trainable=True,name='origin') # constant angular velocity
        ############wing#######################
        #######################################


        
        # Tracker for monitoring the loss
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="recon_loss")
        self.circle_tracker = keras.metrics.Mean(name="circle_loss")
        self.dyn_tracker = keras.metrics.Mean(name="dyn_loss")
        self.mass_tracker = keras.metrics.Mean(name="mass_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.circle_tracker, self.dyn_tracker, self.mass_tracker]



    def get_config(self):
        # Just obtaining the base configuration of the parent class automatically includes alpha, beta, and gamma too
        config = super(PhysicsConstrainedModel, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # When restoring from config, first initialize the undefined encoder/decoder as None
        # (Safe here since at load time this is passed in via load_model's custom_objects)
        return cls(encoder=None, decoder=None, **config)

    
    def call(self, inputs, training=False):
        # At inference time, call the base model as is
        encoder=self.encoder(inputs, training=training)
        return self.decoder(encoder, training=training)

    def train_step(self, data):
        # Assumes the current time step (x_t) is received from the data loader
        x_t, circle_index = data
        # print('x_t.shape',x_t.shape)
        # print('circle_index.shape',circle_index.shape)
        with tf.GradientTape() as tape:
                
            # 1. Get the raw latent vector
            z_raw = self.encoder(x_t[:,:,0], training=True)
    
            # 2. Build the parameter table, made positive via softplus
            # For circles: expand (3, 1) -> (3, 2) (a=b=R)
            r = tf.math.softplus(self.circle_radii) + 1e-5
            circle_params = tf.concat([r, r], axis=1) 
            
            # For ellipses: (3, 2), made positive via softplus
            ellipse_params = tf.math.softplus(self.ellipse_axes) + 0.1
    
            
            # Lookup table for all 6 groups (index 0,1,2 = circle, index 3,4,5 = ellipse)
            all_params = tf.concat([circle_params, ellipse_params], axis=0)
    
            # Gather keisuu (coefficients)
            idx = tf.cast(tf.reshape(circle_index, [-1]), tf.int32)
            batch_radii = tf.gather(all_params, idx)
            a, b = batch_radii[:, 0]+ 1e-5, batch_radii[:, 1]+ 1e-5
    
    
            # Also gather omega and split it by case.
            # omega_val = tf.reshape(tf.math.softplus(self.omega), [-1])
            omega_val = tf.math.softplus(self.omega)
            batch_omega = tf.gather(omega_val, idx)
            omega1= batch_omega[:,0] + 1e-5#, batch_omega[:,1] + 1e-5
            
    
            # Origin setting
            circle_origin = self.circle_origin
            circle_origin_params = tf.concat([circle_origin, circle_origin], axis=1)
            ellipse_origin_params = tf.math.softplus(self.origin)
            origin_val = tf.concat([circle_origin_params, ellipse_origin_params], axis=0)
            batch_origin = tf.gather(origin_val, idx)
            origin1,origin2 = batch_origin[:,0] + 1e-5, batch_origin[:,1] + 1e-5
    
    
            # Theta setting
            theta_val = tf.math.softplus(self.theta)
            batch_theta = tf.gather(theta_val, idx)
            theta1 = batch_theta[:] + 1e-5
    
            # Apply the rotation matrix R
            cos_w1, sin_w1 = tf.cos(omega1), tf.sin(omega1)
            cos_t1, sin_t1 = tf.cos(theta1), tf.sin(theta1)
            
            zeros = tf.zeros_like(a)
            ones = tf.ones_like(a)
            
            # --- [583, 2, 2] ---
            
            # 1. Ellipse deformation matrix (a, b)
            ellipse_shape = tf.stack([
                tf.stack([a, zeros], axis=-1),
                tf.stack([zeros, b], axis=-1)
            ], axis=-2)
            # print('ellipse_shape.shape',ellipse_shape.shape)

            
            # 2. Ellipse inverse deformation matrix (1/a, 1/b)
            ellipse_shape_inv = tf.stack([
                tf.stack([1.0 / a, zeros], axis=-1),
                tf.stack([zeros, 1.0 / b], axis=-1)
            ], axis=-2)
            # print('ellipse_shape_inv.shape',ellipse_shape_inv.shape)
            
            
            # 3. Forward rotation matrix (cos_t1, sin_t1)
            rot_mat = tf.stack([
                tf.stack([cos_t1, -sin_t1], axis=-1),
                tf.stack([sin_t1, cos_t1], axis=-1)
            ], axis=-2)
            
            # 4. Inverse rotation matrix
            rot_mat_inv = tf.stack([
                tf.stack([cos_t1, sin_t1], axis=-1),
                tf.stack([-sin_t1, cos_t1], axis=-1)
            ], axis=-2)
            rot_mat = tf.squeeze(rot_mat, axis=1)
            rot_mat_inv = tf.squeeze(rot_mat_inv, axis=1)
            # print('rot_mat_inv.shape',rot_mat_inv.shape)
            # print('rot_mat.shape',rot_mat.shape)
            
            # 5. Dynamics rotation matrix (cos_w1, sin_w1)
            dyn_mat = tf.stack([
                tf.stack([cos_w1, -sin_w1], axis=-1),
                tf.stack([sin_w1, cos_w1], axis=-1)
            ], axis=-2)

            # print('dyn_mat.shape',dyn_mat.shape)

            
            ###############
    
            # --- A. Compute from the origin ---
            z_raw_origin = z_raw-batch_origin + 1e-5
    
            # --- B. Compute (soft) error minimization ---
            
            # Rotate, apply to the circle, apply each velocity to the ellipse, then shift to the origin via the inverse rotation
            M = tf.einsum('bik,bkj->bij', ellipse_shape_inv, rot_mat_inv)
            M = tf.einsum('bik,bkj->bij', dyn_mat, M)
            M = tf.einsum('bik,bkj->bij', ellipse_shape, M)
            # [583, 2, 2]
            total_transform_mat = tf.einsum('bik,bkj->bij', rot_mat, M)
            
            
            # --- 2. Multiply the completed "583 2x2 matrices" by the "583 2D vectors" all at once ---
            # * Using 'bij,bj->bi', compute the 2x2 matrix(ij) x 2D vector(j) -> 2D vector(i) for each sample
            z_final_rot = tf.einsum('bij,bj->bi', total_transform_mat, z_raw_origin)
            
            
            # --- 3. Finally shift back to the origin (add the 583x2 vectors) ---
            z_soft_pred = z_final_rot + batch_origin
    
            # z_soft_pred = tf.stack([
            #     z_raw_origin[:, 0] * cos_w1 - z_raw_origin[:, 1] * sin_w1*a/b,
            #     z_raw_origin[:, 0] * sin_w1*b/a + z_raw_origin[:, 1] * cos_w1
            # ], axis=1)
    
            # Merge via tf.where
            # Current latent variable
            # z_current = tf.where(is_strict[:, None], z_hard, z_raw)
            z_current = z_raw
            # Next-time-step predicted latent variable
            # z_predict = tf.where(is_strict[:, None], z_hard_pred, z_soft_pred)
            z_predict = z_soft_pred
    
            #################
    
            # decoding
            x_final = self.decoder(z_current, training=True)
            x_pred_out = self.decoder(z_predict, training=True)
    
    
            
    
    
    
    
   
            # (A) Reconstruction error (sum of squares of the 2D vector for each sample)
            loss_recon = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(x_t[:,:,0] - x_final),axis=1)))
            # print('loss_recon.shape',loss_recon.shape)
    
            # (B) Topology (circle/ellipse) error
            M = tf.einsum('bik,bkj->bij', ellipse_shape_inv, rot_mat_inv)
            centerized_ellipse = tf.einsum('bij,bj->bi', M , z_raw_origin)
            norm_err = tf.square(tf.square(centerized_ellipse[:, 0]) + tf.square(centerized_ellipse[:, 1])- 1.0)
            loss_topo = tf.sqrt(tf.reduce_mean(norm_err))
            # print('loss_topo.shape',loss_topo.shape)
            
            # (C) Dynamics (time evolution) error
            loss_dyn = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(x_t[:,:,19] - x_pred_out),axis=1)))
            # print('loss_dyn.shape',loss_dyn.shape)
            
            # (D) center of mass
            loss_mass = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(z_raw_origin),axis=1)))
            # print('loss_mass.shape',loss_mass.shape)
            
            # Compute the total loss
            total_loss = loss_recon + self.alpha * loss_topo + self.beta * loss_dyn + self.gamma * loss_mass
            # total_loss = loss_recon + self.alpha * loss_topo + self.beta * loss_dyn # + self.beta * loss_dyn + self.theta * loss_mass
                



            
            tf.debugging.check_numerics(total_loss, "Loss contains NaN")
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            
            # Safety handling to prevent the gradient from becoming None when an index is missing from the batch
            valid_grads_and_vars = [
                (grad, var) for grad, var in zip(gradients, trainable_vars) if grad is not None
            ]
            self.optimizer.apply_gradients(valid_grads_and_vars)
            
    
            # Record the loss
            self.loss_tracker.update_state(total_loss)
            self.recon_tracker.update_state(loss_recon)
            self.circle_tracker.update_state(loss_topo)
            self.dyn_tracker.update_state(loss_dyn)
            self.mass_tracker.update_state(loss_mass)
    
            return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_t, circle_index = data
        # 1. Get the raw latent vector
        z_raw = self.encoder(x_t[:,:,0], training=False)

        # 2. Build the parameter table, made positive via softplus
        # For circles: expand (3, 1) -> (3, 2) (a=b=R)
        r = tf.math.softplus(self.circle_radii) + 1e-5
        circle_params = tf.concat([r, r], axis=1) 
        
        # For ellipses: (3, 2), made positive via softplus
        ellipse_params = tf.math.softplus(self.ellipse_axes) + 0.1

        
        # Lookup table for all 6 groups (index 0,1,2 = circle, index 3,4,5 = ellipse)
        all_params = tf.concat([circle_params, ellipse_params], axis=0)

        # Gather keisuu (coefficients)
        idx = tf.cast(tf.reshape(circle_index, [-1]), tf.int32)
        batch_radii = tf.gather(all_params, idx)
        a, b = batch_radii[:, 0]+ 1e-5, batch_radii[:, 1]+ 1e-5


        # Also gather omega and split it by case.
        # omega_val = tf.reshape(tf.math.softplus(self.omega), [-1])
        omega_val = tf.math.softplus(self.omega)
        batch_omega = tf.gather(omega_val, idx)
        omega1= batch_omega[:,0] + 1e-5#, batch_omega[:,1] + 1e-5
        

        # Origin setting
        circle_origin = self.circle_origin
        circle_origin_params = tf.concat([circle_origin, circle_origin], axis=1)
        ellipse_origin_params = tf.math.softplus(self.origin)
        origin_val = tf.concat([circle_origin_params, ellipse_origin_params], axis=0)
        batch_origin = tf.gather(origin_val, idx)
        origin1,origin2 = batch_origin[:,0] + 1e-5, batch_origin[:,1] + 1e-5


        # Theta setting
        theta_val = tf.math.softplus(self.theta)
        batch_theta = tf.gather(theta_val, idx)
        theta1 = batch_theta[:] + 1e-5

        # Apply the rotation matrix R
        cos_w1, sin_w1 = tf.cos(omega1), tf.sin(omega1)
        cos_t1, sin_t1 = tf.cos(theta1), tf.sin(theta1)
        
        zeros = tf.zeros_like(a)
        ones = tf.ones_like(a)
        
        # --- [583, 2, 2] ---
        
        # 1. Ellipse deformation matrix (a, b)
        ellipse_shape = tf.stack([
            tf.stack([a, zeros], axis=-1),
            tf.stack([zeros, b], axis=-1)
        ], axis=-2)
        # print('ellipse_shape.shape',ellipse_shape.shape)

        
        # 2. Ellipse inverse deformation matrix (1/a, 1/b)
        ellipse_shape_inv = tf.stack([
            tf.stack([1.0 / a, zeros], axis=-1),
            tf.stack([zeros, 1.0 / b], axis=-1)
        ], axis=-2)
        # print('ellipse_shape_inv.shape',ellipse_shape_inv.shape)
        
        
        # 3. Forward rotation matrix (cos_t1, sin_t1)
        rot_mat = tf.stack([
            tf.stack([cos_t1, -sin_t1], axis=-1),
            tf.stack([sin_t1, cos_t1], axis=-1)
        ], axis=-2)
        
        # 4. Inverse rotation matrix
        rot_mat_inv = tf.stack([
            tf.stack([cos_t1, sin_t1], axis=-1),
            tf.stack([-sin_t1, cos_t1], axis=-1)
        ], axis=-2)
        rot_mat = tf.squeeze(rot_mat, axis=1)
        rot_mat_inv = tf.squeeze(rot_mat_inv, axis=1)
        # print('rot_mat_inv.shape',rot_mat_inv.shape)
        # print('rot_mat.shape',rot_mat.shape)
        
        # 5. Dynamics rotation matrix (cos_w1, sin_w1)
        dyn_mat = tf.stack([
            tf.stack([cos_w1, -sin_w1], axis=-1),
            tf.stack([sin_w1, cos_w1], axis=-1)
        ], axis=-2)

        # print('dyn_mat.shape',dyn_mat.shape)

        
        ###############

        # --- A. Compute from the origin ---
        z_raw_origin = z_raw-batch_origin + 1e-5

        # --- B. Compute (soft) error minimization ---
        
        # Rotate, apply to the circle, apply each velocity to the ellipse, then shift to the origin via the inverse rotation
        M = tf.einsum('bik,bkj->bij', ellipse_shape_inv, rot_mat_inv)
        M = tf.einsum('bik,bkj->bij', dyn_mat, M)
        M = tf.einsum('bik,bkj->bij', ellipse_shape, M)
        # [583, 2, 2]
        total_transform_mat = tf.einsum('bik,bkj->bij', rot_mat, M)
        
        
        # --- 2. Multiply the completed "583 2x2 matrices" by the "583 2D vectors" all at once ---
        # * Using 'bij,bj->bi', compute the 2x2 matrix(ij) x 2D vector(j) -> 2D vector(i) for each sample
        z_final_rot = tf.einsum('bij,bj->bi', total_transform_mat, z_raw_origin)
        
        
        # --- 3. Finally shift back to the origin (add the 583x2 vectors) ---
        z_soft_pred = z_final_rot + batch_origin

        # z_soft_pred = tf.stack([
        #     z_raw_origin[:, 0] * cos_w1 - z_raw_origin[:, 1] * sin_w1*a/b,
        #     z_raw_origin[:, 0] * sin_w1*b/a + z_raw_origin[:, 1] * cos_w1
        # ], axis=1)

        # Merge via tf.where
        # Current latent variable
        # z_current = tf.where(is_strict[:, None], z_hard, z_raw)
        z_current = z_raw
        # Next-time-step predicted latent variable
        # z_predict = tf.where(is_strict[:, None], z_hard_pred, z_soft_pred)
        z_predict = z_soft_pred

        #################

        # decoding
        x_final = self.decoder(z_current, training=False)
        x_pred_out = self.decoder(z_predict, training=False)


        





        # (A) Reconstruction error (sum of squares of the 2D vector for each sample)
        loss_recon = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(x_t[:,:,0] - x_final),axis=1)))
        # print('loss_recon.shape',loss_recon.shape)

        # (B) Topology (circle/ellipse) error
        M = tf.einsum('bik,bkj->bij', ellipse_shape_inv, rot_mat_inv)
        centerized_ellipse = tf.einsum('bij,bj->bi', M , z_raw_origin)
        norm_err = tf.square(tf.square(centerized_ellipse[:, 0]) + tf.square(centerized_ellipse[:, 1])- 1.0)
        loss_topo = tf.sqrt(tf.reduce_mean(norm_err))
        # print('loss_topo.shape',loss_topo.shape)
        
        # (C) Dynamics (time evolution) error
        loss_dyn = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(x_t[:,:,19] - x_pred_out),axis=1)))
        # print('loss_dyn.shape',loss_dyn.shape)
        
        # (D) center of mass
        loss_mass = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(z_raw_origin),axis=1)))
            
        # Compute the total loss
        total_loss = loss_recon + self.alpha * loss_topo + self.beta * loss_dyn + self.gamma * loss_mass
        
        # Update the metrics
        self.loss_tracker.update_state(total_loss)
        self.recon_tracker.update_state(loss_recon)
        self.circle_tracker.update_state(loss_topo)
        self.dyn_tracker.update_state(loss_dyn)
        self.mass_tracker.update_state(loss_mass)

        
        return {m.name: m.result() for m in self.metrics}



betas = 1e1
alphas = 1e-2
gammas = 1e-2
patience = 700
epoch = 100000

def hard_swish_custom(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0
act = hard_swish_custom

# --- 1. Encoder-only model ---
enc_input = Input(shape=(2,))
x = Dense(16, activation=act)(enc_input)
x = Dense(128, activation=act)(x)
x = Dense(64, activation=act)(x)
x = Dense(32, activation=act)(x)
z_raw = Dense(2, activation='linear', name='z_raw')(x) # coordinate hint
encoder_model = Model(enc_input, z_raw, name="encoder")

# --- 2. Decoder-only model ---
dec_input = Input(shape=(2,)) # receive the projected 2D vector
x = Dense(32, activation=act)(dec_input)
x = Dense(64, activation=act)(x)
x = Dense(128, activation=act)(x)
x = Dense(16, activation=act)(x)
x_out = Dense(2, activation='linear')(x)
decoder_model = Model(dec_input, x_out, name="decoder")


total_input = enc_input 

z = encoder_model(total_input)
total_output = decoder_model(z)




constrained_model = PhysicsConstrainedModel(encoder=encoder_model, decoder=decoder_model,  alpha=alpha, beta=beta,gamma=gamma)
constrained_model.build(input_shape=(None, 2))
constrained_model.compile(
    optimizer='adam',
    # loss=[orig_loss, MeanSquaredError(), MeanSquaredError()],
    # loss_weights=[1.0, 1.0, 1.0],
    weighted_metrics=[],
    run_eagerly=False
)



# --- Callback configuration ---
# When save_weights_only=False, save the whole model (uses the previous custom-compatible class)
model_cb = ModelCheckpoint(
f'./Model/model.keras', 
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False, # save the full model structure too
    verbose=0 # to avoid flooding the log, 0 or 2 is recommended when looping
)

early_cb = EarlyStopping(
    monitor='val_loss', 
    patience=patience, 
    verbose=0
)

cb = [model_cb, early_cb]

# --- Run training ---
history = constrained_model.fit(
    dataset_cy,
    epochs=epoch,
    verbose=0,
    callbacks=cb,
    validation_data=val_dataset_cy
)

# --- Save results ---
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch

df_results.to_csv(path_or_buf='./History/history.csv', index=False)

# Prevent memory leaks (release old graphs/models from the session)
tf.keras.backend.clear_session()