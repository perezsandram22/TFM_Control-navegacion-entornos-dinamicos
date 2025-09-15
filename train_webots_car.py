import os
import time
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from webots_car_env import WebotsCarEnv
import numpy as np  # Asegúrate de importar numpy al inicio

# --- Callback personalizado para seguimiento de pasos ---
class PasoCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.paso_actual = 0

    def _on_step(self) -> bool:
        self.paso_actual += 1
        if self.paso_actual % self.check_freq == 0:
            print(f"[INFO] Paso de entrenamiento: {self.paso_actual}")
        return True

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=5, min_delta=0.5, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.rewards) >= 100:
                mean_reward = np.mean(self.rewards[-100:])
                print(f"[EARLY STOP] Reward promedio reciente: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    if self.no_improvement_count >= self.patience:
                        print(f"[EARLY STOP] No mejora en {self.patience} chequeos. Deteniendo entrenamiento.")
                        return False
        return True

    def _on_rollout_end(self) -> None:
        ep_info = self.locals.get("infos", [])
        for info in ep_info:
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])


# --- Inicializa entorno y modelo ---
def inicializar_modelo(model_path, env):
    
    print("[INFO] Creando nuevo modelo PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        gamma=0.99,
        learning_rate=0.0003,
        n_steps=2048,
        gae_lambda=0.98
    )     
    return model

# --- Entrenamiento con reconexión ---
def entrenar_modelo(model, env, total_timesteps=5000):
    callback = [PasoCallback(check_freq=1000), EarlyStoppingCallback(patience=7, check_freq=1000)]
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10, callback=callback)
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"[WARN] Conexión perdida: {e}. Intentando reconectar...")
        for intento in range(5):
            try:
                env.close()
                env = WebotsCarEnv()
                model.set_env(env)
                print(f"[INFO] Reconexión exitosa en el intento {intento+1}.")
                model.learn(total_timesteps=total_timesteps, log_interval=10, callback=callback)
                break
            except Exception as e2:
                print(f"[WARN] Falló el intento {intento+1}: {e2}")
                time.sleep(2)
        else:
            print("[ERROR] No se pudo reestablecer la conexión tras varios intentos.")
    return model

# --- Evaluación del modelo ---
def evaluar_modelo(model, env, num_episodios=10):
    logs_evaluacion = []
    for episode in range(1, num_episodios + 1):
        obs, info = env.reset()
        assert len(obs) >= 7, f"Esperaba al menos 7 dimensiones, recibí {len(obs)}"
        done = False
        step = 0
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs_, reward, terminated, truncated, info = env.step(action)
            logs_evaluacion.append({
                'episode': episode,
                'step': step,
                'reward': reward,
                'action_steer': action[0],
                'action_vel': action[1],
                'obs_steer': obs[0],
                'obs_min_lidar': obs[1],
                'obs_bumper': obs[2],
                'obs_velocidad': obs[3],
                'obs_obstacle_direction': obs[4],
                'obs_mpc_steering': obs[5],
                'obs_mpc_velocidad': obs[6],
            })
            obs = obs_
            total_reward += reward
            step += 1
            done = terminated or truncated
        #print(f"Episodio {episode} terminado. Reward total: {total_reward}")
    return logs_evaluacion

# --- Exportación y visualización ---
def exportar_resultados(logs, archivo_excel='entrenamiento_webots.xlsx'):
    df = pd.DataFrame(logs)
    reward_por_ep = df.groupby('episode')['reward'].sum()
    with pd.ExcelWriter(archivo_excel) as writer:
        df.to_excel(writer, sheet_name='Datos', index=False)
        reward_por_ep.to_frame(name='Reward Total').to_excel(writer, sheet_name='Estadísticas')
    print(f"[INFO] Datos exportados a {archivo_excel}")
    print(f"Reward promedio: {reward_por_ep.mean():.2f}")
    print(f"Reward máximo: {reward_por_ep.max():.2f}")
    print(f"Reward mínimo: {reward_por_ep.min():.2f}")
    #plt.plot(reward_por_ep)
    #plt.xlabel("Episodio")
    #plt.ylabel("Reward total")
    #plt.title("Desempeño del agente")
    #plt.grid()
    #plt.show()

#SIN MPC
# --- Ejecución principal ---
def main():
    model_path = "ppo_webots_car"
    env = WebotsCarEnv()
    check_env(env, warn=True)
    print(f"[DOT] Iniciazando modelo")
    model = inicializar_modelo(model_path, env)
    print(f"[DOT] Entrenando modelo...")
    model = entrenar_modelo(model, env, total_timesteps=500000)
    print(f"[DOT] Guardo el modelo en {model_path}")
    model.save(model_path)
    print(f"[DOT] Exportando resultados...")
    logs_evaluacion = evaluar_modelo(model, env, num_episodios=100)
    exportar_resultados(logs_evaluacion)
    env.close()
    print(f"[FIN]")

if __name__ == "__main__":
    main()
