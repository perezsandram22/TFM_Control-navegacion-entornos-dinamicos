from stable_baselines3 import PPO
from webots_car_env import WebotsCarEnv

# Instancia el entorno
env = WebotsCarEnv()
# Carga el modelo entrenado
model_path = r"ppo_webots_car"
test_model = PPO.load(model_path, env=env)

obs, info = env.reset()
done = False
while not done:
    action, _ = test_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Reward: {reward}, Obs: {obs}")

env.close()
