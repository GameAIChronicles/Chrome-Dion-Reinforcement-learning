from stable_baselines3 import PPO
from ENV_Chrome_dino import ENVChromeDino
import os
env = ENVChromeDino()

model = PPO.load(os.path.join('data', 'models', 'Dino_model', 'model_PPO_7 0k'))

episodes = 15
for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(state)
        action = int(action)
        state, reward, done, _, _ = env.step(action)
        score += reward
    print('Episodes: {}, Score: {}'.format(episode, score))