import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import pickle
import struct
from scipy.optimize import minimize
 
class WebotsCarEnv(gym.Env):
    def __init__(self, mpc_in_obs=True):
        super().__init__()
        self.mpc_in_obs = mpc_in_obs
        if self.mpc_in_obs:
            # Observación extendida: [steering, min_lidar, bumper, velocidad, obstacle_direction, mpc_steering, mpc_velocidad]
            self.observation_space = spaces.Box(
                low=np.array([-0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 22.0]),
                high=np.array([0.5, 100.0, 1.0, 250.0, 3.0, 0.5, 138.889]),
                dtype=np.float32
            )
        else:
            # Observación normal: [steering, min_lidar, bumper, velocidad, obstacle_direction]
            self.observation_space = spaces.Box(
                low=np.array([-0.5, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([0.5, 100.0, 1.0, 250.0, 3.0]),
                dtype=np.float32
            )
        self.action_space = spaces.Box(low=np.array([-0.5, 22.0]), high=np.array([0.5, 138.889]), dtype=np.float32)  # steering y velocidad mínima 22.0
        # Socket config
        self.host = 'localhost'
        self.port = 10000
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port))
 
    def recvall(self, n):
        data = b''
        while len(data) < n:
            packet = self.client.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
 
    def send_msg(self, msg):
        msg = pickle.dumps(msg)
        msg = struct.pack('>I', len(msg)) + msg
        self.client.sendall(msg)
 
    def recv_msg(self):
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        return pickle.loads(self.recvall(msglen))
 
    def mpc_action(self, obs):
        # Modelo cinemático simple y optimización rápida para sugerir acción MPC
        def car_dynamics(state, action, dt=0.1):
            x, y, theta = state
            v, delta = action
            L = 2.5
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = theta + v / L * np.tan(delta) * dt
            return np.array([x_new, y_new, theta_new])
        def mpc_cost(u_flat, state0, min_lidar, N=5):
            u = u_flat.reshape(N, 2)
            state = np.array(state0)
            cost = 0
            for i in range(N):
                v, delta = u[i]
                cost += 10 * (state[1]**2) + 0.1 * (delta**2) - 2 * v
                if min_lidar < 0.7:
                    cost += 100 * (0.7 - min_lidar)
                state = car_dynamics(state, u[i])
            return cost
        # Estado inicial ficticio (no tenemos GPS/yaw real aquí, así que usamos velocidad y steering)
        state0 = [0, 0, 0]
        min_lidar = obs[1]
        N = 5
        u0 = np.tile([float(obs[3]), float(obs[0])], N)
        bounds = [(-0.5, 0.5), (22.0, 138.889)] * N
        res = minimize(mpc_cost, u0, args=(state0, min_lidar, N), bounds=bounds, method='SLSQP', options={'maxiter': 10, 'disp': False})
        u_opt = res.x.reshape(N, 2)
        mpc_vel, mpc_steer = u_opt[0][0], u_opt[0][1]
        # Clipping por seguridad
        mpc_steer = float(np.clip(mpc_steer, -0.5, 0.5))
        mpc_vel = float(np.clip(mpc_vel, 22.0, 138.889))
        return mpc_steer, mpc_vel
 
    def reset(self, seed=None, options=None):
        for intento in range(10):
            try:
                self.send_msg({'cmd': 'reset'})
                obs_dict = self.recv_msg()
                obs = np.array([
                    obs_dict['steering'],
                    obs_dict['min_lidar'],
                    obs_dict['bumper'],
                    obs_dict['velocidad'],
                    obs_dict.get('obstacle_direction', 0)
                ], dtype=np.float32)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                if self.mpc_in_obs:
                    # --- Añadir acción MPC a la observación ---
                    mpc_steer, mpc_vel = self.mpc_action(obs)
                    obs = np.concatenate([obs, [mpc_steer, mpc_vel]]).astype(np.float32)
                info = {}
                return obs, info
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"[WARN] Conexión perdida en reset: {e}. Intentando reconectar ({intento+1}/10)...")
                import time; time.sleep(2)
                try:
                    self.client.close()
                except:
                    pass
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.host, self.port))
        raise RuntimeError("No se pudo reestablecer la conexión tras varios intentos en reset.")
 
    def step(self, action):
        # Acción: steering y velocidad, permitiendo velocidad mínima 22.0
        action_full = np.array([action[0], np.clip(action[1], 22.0, 138.889)], dtype=np.float32)
        for intento in range(10):
            try:
                self.send_msg({'cmd': 'step', 'action': action_full})
                obs_dict, reward, terminated, truncated, info = self.recv_msg()
                obs = np.array([
                    obs_dict['steering'],
                    obs_dict['min_lidar'],
                    obs_dict['bumper'],
                    obs_dict['velocidad'],
                    obs_dict.get('obstacle_direction', 0)
                ], dtype=np.float32)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                if self.mpc_in_obs:
                    # --- Añadir acción MPC a la observación ---
                    mpc_steer, mpc_vel = self.mpc_action(obs)
                    obs = np.concatenate([obs, [mpc_steer, mpc_vel]]).astype(np.float32)
                reward = float(np.clip(reward, -100, 100))
                return obs, reward, terminated, truncated, info
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"[WARN] Conexión perdida en step: {e}. Intentando reconectar ({intento+1}/10)...")
                import time; time.sleep(2)
                try:
                    self.client.close()
                except:
                    pass
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.host, self.port))
        raise RuntimeError("No se pudo reestablecer la conexión tras varios intentos en step.")
 
    def close(self):
        self.send_msg({'cmd': 'close'})
        self.client.close()
 
    def render(self):
        pass
 
    def test_drive_until_collision(self):
        # Este método no es usado por el entrenamiento, pero lo he corregido para que no falle.
        self.send_msg({'cmd': 'reset'})
        obs_dict = self.recv_msg()
        # Acceder a los datos de la observación de forma correcta
        obs = np.array([obs_dict['steering'], obs_dict['min_lidar'], obs_dict['bumper'], obs_dict['velocidad'], obs_dict.get('obstacle_direction', 0)], dtype=np.float32)
        done = False
        step = 0
        while not done:
            # Acción: steering=0 (recto), velocidad=250 (máxima)
            action = np.array([0.0, 250.0], dtype=np.float32)
            self.send_msg({'cmd': 'step', 'action': action})
            obs_dict, reward, terminated, truncated, info = self.recv_msg()
            # Acceder a los datos de la observación de forma correcta
            obs = np.array([obs_dict['steering'], obs_dict['min_lidar'], obs_dict['bumper'], obs_dict['velocidad'], obs_dict.get('obstacle_direction', 0)], dtype=np.float32)
            print(f"Step {step}: Velocidad={obs[3]:.2f}, Steering={obs[0]:.2f}, Reward={reward}, Terminado={terminated}, Info={info}")
            step += 1
            if terminated or truncated:
                print("Episodio terminado. Motivo: colisión o condición de parada.")
                break
 
if __name__ == "__main__":
    env = WebotsCarEnv()
    obs, info = env.reset()
    print("Conexión exitosa. Observación recibida:", np.shape(obs))
    # Acción para avanzar a una velocidad controlada durante 10 segundos
    velocidad_30kmh = 104.0
    action = np.array([0.0, velocidad_30kmh], dtype=np.float32)
    steps = 10
    import time
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        # Se accede a la velocidad en el índice 3 de la observación
        print(f"Paso {i+1}: Avanzando a {obs[3]:.2f} m/s. Recompensa: {reward}")
        time.sleep(1.0)
    # Acción para detenerse
    action = np.array([0.0, 22.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print("Detenido. Recompensa:", reward)
    env.close()