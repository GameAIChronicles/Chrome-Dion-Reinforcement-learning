"""
This file or module train the model and save the model in data/models/Dino_model folder.
"""

# Import Library
from ENV_Chrome_dino import ENVChromeDino
import os
from stable_baselines3 import PPO


log_path = os.path.join('data', 'logs')  # Path of the log folder

env = ENVChromeDino()  # Costume environment
env.reset()

model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log=log_path)  # Setting up the algorithm

Time = 10_000
for i in range(1, 16):
    model.learn(total_timesteps=Time, reset_num_timesteps=False, tb_log_name='PPO')  # Training the model
    model.save(os.path.join('data', 'models', 'Dino_model', f'model_PPO_{Time*i}_new.zip'))  # Saving the model


"""
After the model is trained the model will be saved to the folder. It could be used for prediction later.
"""