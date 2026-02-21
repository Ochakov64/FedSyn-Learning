# server_GAN.py 

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional
import os
import matplotlib.pyplot as plt

final_parameters: Optional[fl.common.NDArrays] = None

LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)

def build_generator():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
    model = Sequential([
        Dense(256, input_dim=LATENT_DIM), LeakyReLU(alpha=0.2), BatchNormalization(momentum=0.8),
        Dense(512), LeakyReLU(alpha=0.2), BatchNormalization(momentum=0.8),
        Dense(1024), LeakyReLU(alpha=0.2), BatchNormalization(momentum=0.8),
        Dense(np.prod(IMG_SHAPE), activation='tanh'), Reshape(IMG_SHAPE)
    ], name="generator")
    return model

class SaveFinalParamsStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            global final_parameters
            final_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict):
    if server_round == 0 or server_round % 10 == 0 or server_round == 300:
        print(f"\n--- Saving images for evaluation round {server_round} ---")
        model = build_generator()
        model.set_weights(parameters)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
        gen_imgs = model.predict(noise, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"round_{server_round}_generated_images.png")
        plt.close()
        print(f"Saved generated images to round_{server_round}_generated_images.png")
    return 0.0, {}

strategy = SaveFinalParamsStrategy(fraction_fit=1.0, min_available_clients=3, min_fit_clients=3, evaluate_fn=evaluate_fn)

print("Starting Flower server (GAN version)...")
fl.server.start_server(
    server_address="0.0.0.0:8888",
    config=fl.server.ServerConfig(num_rounds=300),
    strategy=strategy
)

print("\n--- Federated Training Finished ---")
if final_parameters:
    print("\n--- Saving final global generator model ---")
    from flwr.common import parameters_to_ndarrays
    final_ndarrays = parameters_to_ndarrays(final_parameters)
    final_generator = build_generator()
    final_generator.set_weights(final_ndarrays)
    model_save_path = "final_federated_gan_generator.keras"
    final_generator.save(model_save_path)
    print(f"Final generator model saved to: {model_save_path}")
