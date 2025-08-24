from stable_baselines3.common.env_checker import check_env
from ENV_Chrome_dino import ENVChromeDino

env = ENVChromeDino()
check_env(env=env)