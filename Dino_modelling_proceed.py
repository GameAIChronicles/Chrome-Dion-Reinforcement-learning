from stable_baselines3 import PPO
from ENV_Chrome_dino import ENVChromeDino
import os
env = ENVChromeDino()

model = PPO.load(os.path.join('data', 'models', 'Dino_model', 'model_PPO_60k'), env=env)
model.learn(total_timesteps=20000, reset_num_timesteps=False, tb_log_name='PPO', progress_bar=True)  # Training the model
model.save(os.path.join('data', 'models', 'Dino_model', 'model_S_PPO_80k.zip'))