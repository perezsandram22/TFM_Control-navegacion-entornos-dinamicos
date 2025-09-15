import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from webots_car_env import WebotsCarEnv

# CON MPC.
class MPCLoggingCallback(BaseCallback):
    """
    Un callback personalizado que registra los datos del entrenamiento
    y detiene el proceso después de un número específico de episodios.
    """
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.logs = []
        self.current_episode = 1
        self.max_episodes = max_episodes

    def _on_step(self) -> bool:
        # Imprime el progreso cada 1000 pasos de tiempo
        if self.num_timesteps % 1000 == 0:
            print(f"Paso actual: {self.num_timesteps}")

        # Obtiene la observación y la acción del paso actual
        obs = self.locals['new_obs']
        action = self.locals['actions']

        # Verifica si un episodio ha terminado
        if self.locals['dones'][0]:
            self.current_episode += 1
            # Si se ha alcanzado el número máximo de episodios, detiene el entrenamiento
            if self.current_episode > self.max_episodes:
                print(f"Alcanzado el límite de {self.max_episodes} episodios. Deteniendo el entrenamiento.")
                return False  # Retornar False detiene el bucle de entrenamiento

        # Desempaqueta la observación y la acción
        single_obs = obs[0]
        single_action = action[0]

        # Asegura que el número de observaciones sea el esperado antes de registrar
        if len(single_obs) >= 7:
            self.logs.append({
                'step': self.num_timesteps,
                'episode': self.current_episode,
                'reward': self.locals['rewards'][0],
                'action_steer': single_action[0],
                'action_vel': single_action[1],
                'obs_steer': single_obs[0],
                'obs_min_lidar': single_obs[1],
                'obs_bumper': single_obs[2],
                'obs_velocidad': single_obs[3],
                'obs_obstacle_direction': single_obs[4],
                'obs_mpc_steering': single_obs[5],
                'obs_mpc_velocidad': single_obs[6],
            })
        return True

    def exportar_resultados(self, archivo_excel='training_mpc_logs.xlsx'):
        """
        Procesa los datos de los logs para generar un archivo Excel con
        una hoja de 'Datos' y una de 'Estadísticas'.
        """
        if not self.logs:
            print("[INFO] No hay datos para exportar.")
            return

        df = pd.DataFrame(self.logs)
        reward_por_ep = df.groupby('episode')['reward'].sum()

        with pd.ExcelWriter(archivo_excel) as writer:
            df.to_excel(writer, sheet_name='Datos', index=False)
            reward_por_ep.to_frame(name='Reward Total').to_excel(writer, sheet_name='Estadísticas')
        
        print(f"[INFO] Datos exportados a {archivo_excel}")
        print(f"Reward promedio: {reward_por_ep.mean():.2f}")
        print(f"Reward máximo: {reward_por_ep.max():.2f}")
        print(f"Reward mínimo: {reward_por_ep.min():.2f}")

# ======================== CONFIGURACIÓN DEL ENTRENAMIENTO PRINCIPAL ========================

# Inicializa el entorno con la observación extendida (MPC)
env = WebotsCarEnv(mpc_in_obs=True)

# Crea una instancia del callback y especifica el número de episodios deseado
MAX_EPISODES = 100
callback = MPCLoggingCallback(max_episodes=MAX_EPISODES)

# Define el modelo PPO
model = PPO("MlpPolicy", env, verbose=1)

# Establece total_timesteps en un valor muy grande para que la detención
# sea controlada por el callback y no por el número de pasos de tiempo
model.learn(total_timesteps=500000, callback=callback)

# Exporta los datos a un archivo de Excel después de que el entrenamiento se detenga
callback.exportar_resultados('training_mpc_logs.xlsx')

# Guarda el modelo entrenado
model.save("ppo_webots_mpc")

# Cierra el entorno
env.close()