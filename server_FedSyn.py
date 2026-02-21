# server_FedSyn.py 

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
import matplotlib.pyplot as plt

final_parameters: Optional[List[np.ndarray]] = None
LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)


def build_generator():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2DTranspose, LeakyReLU, BatchNormalization

    model = Sequential()
 
    model.add(Dense(7 * 7 * 128, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128))) 


    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(1, (7, 7), padding='same', activation='tanh'))
    
    return model

class FedSynStrategy(fl.server.strategy.Strategy):
    def __init__(self, initial_generator):
        self.generator = initial_generator
        self.discriminator = None
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0004, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def initialize_parameters(self, client_manager):
        from flwr.common import ndarrays_to_parameters
        global final_parameters
        final_parameters = self.generator.get_weights()
        return ndarrays_to_parameters(self.generator.get_weights())

    def configure_fit(self, server_round, parameters, client_manager):
        from flwr.common import ndarrays_to_parameters, FitIns
        fit_ins = FitIns(parameters, {})
        clients = client_manager.sample(num_clients=client_manager.num_available(), min_num_clients=3)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}

        from flwr.common import parameters_to_ndarrays
        
        try:
            synthetic_images_list = [parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results]
            synthetic_images = np.concatenate(synthetic_images_list)
            print(f"\n[Server, Round {server_round}] Received {len(synthetic_images)} synthetic images.")

            if self.discriminator is None:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Flatten, Dense, LeakyReLU
                self.discriminator = Sequential([
                    Flatten(input_shape=IMG_SHAPE), Dense(512), LeakyReLU(alpha=0.2),
                    Dense(256), LeakyReLU(alpha=0.2), Dense(1, activation='sigmoid')
                ])

            dataset = tf.data.Dataset.from_tensor_slices(synthetic_images).shuffle(len(synthetic_images)).batch(64)
            
            print(f"[Server, Round {server_round}] Starting server-side training...")
            for epoch in range(1):
                for image_batch in dataset:
                    self._train_step(image_batch)
            print(f"[Server, Round {server_round}] Server-side training complete.")
            
            save_images(server_round, self.generator)
            global final_parameters
            final_parameters = self.generator.get_weights()
            
            from flwr.common import ndarrays_to_parameters
            return ndarrays_to_parameters(self.generator.get_weights()), {}

        except Exception as e:
            print(f"Error during aggregation: {e}")
            return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager): return []
    def aggregate_evaluate(self, server_round, results, failures): return None, {}
    def evaluate(self, server_round, parameters): return None, {}

    @tf.function
    def _train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, LATENT_DIM])
        
        # === CHANGE 2: Instance Noise ===
        # Add noise to the inputs so the discriminator can't be too precise
        img_noise = tf.random.normal(tf.shape(real_images), stddev=0.1)
        
        # Label smoothing
        smoothed_real_labels = tf.ones([batch_size, 1]) * 0.9

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            # Pass NOISY images to discriminator
            real_output = self.discriminator(real_images + img_noise, training=True)
            fake_output = self.discriminator(generated_images + img_noise, training=True)
            
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
            real_loss = self.loss_fn(smoothed_real_labels, real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
            
        grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

def save_images(server_round: int, generator_model):
    if server_round == 0 or server_round % 10 == 0 or server_round == 300:
        print(f"\n--- Saving images for evaluation round {server_round} ---")
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
        gen_imgs = generator_model.predict(noise, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"fedsyn_round_{server_round}_images.png")
        plt.close()
        print(f"Saved FedSyn images to fedsyn_round_{server_round}_images.png")

if __name__ == "__main__":
    initial_generator = build_generator()
    save_images(0, initial_generator)
    strategy = FedSynStrategy(initial_generator=initial_generator)
    print("Starting Flower server (FedSyn version)...")
    fl.server.start_server(server_address="0.0.0.0:8888", config=fl.server.ServerConfig(num_rounds=300), strategy=strategy, grpc_max_message_length=536_870_912)
    print("\n--- Federated Training Finished ---")
    if final_parameters:
        print("\n--- Saving final global generator model ---")
        final_generator = build_generator()
        final_generator.set_weights(fl.common.parameters_to_ndarrays(final_parameters))
        model_save_path = "final_fedsyn_generator.keras"
        final_generator.save(model_save_path)
        print(f"Final FedSyn generator model saved to: {model_save_path}")
        save_images(300, final_generator)
