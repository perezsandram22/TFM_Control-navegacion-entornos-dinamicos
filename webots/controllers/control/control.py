from controller import Supervisor
import numpy as np
import socket
import pickle
import select
import struct
import traceback

TIME_STEP = 50

robot = Supervisor()

# Obtén el nodo del Tesla por DEF (ajusta el DEF en tu mundo a TESLA_MODEL3)
vehicle_node = robot.getFromDef("TESLA_MODEL3")
if vehicle_node is None:
    print("ERROR: No se encontró el nodo con DEF 'TESLA_MODEL3'. Verifica el nombre en el archivo .wbt.")
    exit(1)

# Dispositivos
left_steer = robot.getDevice("left_steer")
right_steer = robot.getDevice("right_steer")
left_rear_wheel = robot.getDevice("left_rear_wheel")
right_rear_wheel = robot.getDevice("right_rear_wheel")
bumper = robot.getDevice("bumper")
bumper.enable(TIME_STEP)

# Inicializar y habilitar la cámara para el overlay de Webots
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# Socket server
host = 'localhost'
port = 10000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen(1)
server.setblocking(False)
conn = None
print("Esperando conexión del cliente Gymnasium...")

def send_msg(sock, msg_bytes):
    msg_len = struct.pack('>I', len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)

# Constante para 30 km/h
VEL_30KMH_RAD_SEC = 30.0 / 3.6 / 0.1  # = 83.333... rad/s

# Contador de episodios
episode_count = 1
step_count = 0
MAX_STEPS_PER_EPISODE = 500

# Variables para métricas nuevas
traj_total_distance = 0.0  # Suma de desplazamientos GPS
traj_start_pos = None      # Posición inicial del episodio
traj_last_pos = None       # Última posición GPS
traj_efficiency = 0.0      # Eficiencia de la trayectoria
colisiones_amenazadas = 0  # Veces que se detecta obstáculo cerca
colisiones_ev = 0          # Veces que se evita colisión (obstáculo cerca pero sin bumper)
colisiones = 0             # Colisiones reales
respuesta_tiempos = []     # Lista de steps de respuesta ante obstáculos
respuesta_en_progreso = False
respuesta_step_inicio = 0
respuesta_min_lidar_antes = None

# Inicializar GPS
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

# Inicializar Lidar
lidar = robot.getDevice("Hokuyo URG-04LX")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# Variable para detectar reinicio de simulación manual
last_time = robot.getTime()

# Métricas para el episodio
if not hasattr(robot, 'episodio_metrica'):
    robot.episodio_metrica = {
        'distancia_total': 0.0,
        'gps_anterior': None,
        'colisiones_amenazadas': 0,
        'colisiones_ev': 0,
        'obstaculo_cerca': False,
        'pasos_respuesta': [],
        'pasos_desde_amenaza': 0,
        'respondiendo': False
    }

# Variable para rastrear el ángulo de giro anterior
last_steering_angle = 0.0

def get_lane_deviation_reward(image_array, road_color, off_road_color, tolerance=25):
    """
    Calcula una recompensa basada en la posición del coche respecto al centro del carril.
    Una recompensa positiva por estar centrado, y una penalización por desviación.
    """
    height, width, _ = image_array.shape
    reward = 0.0
    terminated = False
    
    # Define la región de interés (ROI) en la parte inferior de la imagen
    roi_height = height // 3
    roi_start = height - roi_height
    roi = image_array[roi_start:, :, :3]
    
    # Determina si el coche está completamente fuera de la carretera
    is_off_road = np.all(np.abs(roi - off_road_color) < tolerance, axis=2)
    if np.sum(is_off_road) > (roi_height * width * 0.5): # Si más del 50% del ROI es off-road
        print("[FUERA] ¡ATENCIÓN! Coche fuera de la carretera. Fuerte penalización.")
        reward -= 100.0
        terminated = True
        return reward, terminated

    # Identificar los píxeles de la carretera (color asfalto) en el ROI
    is_road = np.all(np.abs(roi - road_color) < tolerance, axis=2)
    road_pixels_x = np.where(is_road)[1]
    
    if len(road_pixels_x) > 0:
        # Calcular el centro de los píxeles de la carretera
        center_of_road_x = np.mean(road_pixels_x)
        center_of_image_x = width / 2
        
        # Calcular la desviación
        deviation = abs(center_of_road_x - center_of_image_x)
        max_deviation = width / 2
        
        # Recompensa: máxima si la desviación es 0, decrece cuadráticamente
        lane_reward = 1.0 - (deviation / max_deviation) ** 2
        reward += lane_reward * 20.0
        print(f"[CÁMARA] Desviación del carril: {deviation:.2f} px. Recompensa de carril: {lane_reward*20.0:.2f}")

    else:
        # Penalización si no se detecta la carretera (posiblemente fuera de ella o en una zona compleja)
        print("[FUERA1] No se detectan píxeles de carretera. Penalización.")
        reward -= 10.0
        
    return reward, terminated


while robot.step(TIME_STEP) != -1:
    # Detectar reinicio de simulación manual (Ctrl+Shift+V)
    current_time = robot.getTime()
    if current_time < last_time:  # El tiempo se reinició
        viewpoint_node = robot.getFromDef("VIEWPOINT")
        if viewpoint_node is not None:
            viewpoint_node.getField("position").setSFVec3f([-52.0, 42.0, 2.5])
            viewpoint_node.getField("orientation").setSFRotation([1, 0, 0, -1.5])
            try:
                viewpoint_node.setFollow(vehicle_node)
                print("[INFO] VIEWPOINT ahora sigue al vehículo.")
            except Exception as e:
                print(f"[WARN] No se pudo fijar el VIEWPOINT al vehículo: {e}")
            for _ in range(5):
                robot.step(TIME_STEP)
    last_time = current_time

    try:
        if conn is None:
            try:
                conn, addr = server.accept()
                print(f"Conectado a {addr}")
                conn.setblocking(False)
            except BlockingIOError:
                # Espera un ciclo de simulación para evitar bucle apretado
                robot.step(TIME_STEP)
                continue
            continue
        if conn and select.select([conn], [], [], 0)[0]:
            data = recv_msg(conn)
            if data is None:
                print("[WARN] Conexión perdida. Intentando reestablecer...")
                conn = None
                continue
            msg = pickle.loads(data)
            print(f"[INFO] Comando recibido: {msg['cmd']}")
            if msg['cmd'] == 'reset':
                # Reinicia la posición y orientación del coche
                initial_translation = [-52.3043, 39.7613, 0.472846]
                initial_rotation = [0, 1, 0, 0]
                vehicle_node.getField("translation").setSFVec3f(initial_translation)
                vehicle_node.getField("rotation").setSFRotation(initial_rotation)
                left_steer.setPosition(0.0)
                right_steer.setPosition(0.0)
                # Reinicia la vista de la simulación y fuerza actualización
                viewpoint_node = robot.getFromDef("VIEWPOINT")
                if viewpoint_node is not None:
                    viewpoint_node.getField("position").setSFVec3f([-52.0, 42.0, 2.5])
                    viewpoint_node.getField("orientation").setSFRotation([1, 0, 0, -1.5])
                    try:
                        viewpoint_node.setFollow(vehicle_node)
                        print("[INFO] VIEWPOINT ahora sigue al vehículo.")
                    except Exception as e:
                        print(f"[WARN] No se pudo fijar el VIEWPOINT al vehículo: {e}")
                    for _ in range(5):
                        robot.step(TIME_STEP)
                # Reinicia la física
                robot.simulationResetPhysics()
                # Asegura modo velocidad después de resetear la física
                left_rear_wheel.setPosition(float('inf'))
                right_rear_wheel.setPosition(float('inf'))
                left_rear_wheel.setVelocity(VEL_30KMH_RAD_SEC)
                right_rear_wheel.setVelocity(VEL_30KMH_RAD_SEC)
                robot.step(TIME_STEP)
                robot.step(TIME_STEP)  # Paso extra para estabilizar la simulación
                print(f"[INFO] --- Episodio {episode_count} iniciado ---")
                # Limpiar consola cada 10 episodios
                if episode_count % 10 == 0:
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                # Mostrar métricas del episodio anterior (si no es el primero)
                if episode_count > 1:
                    met = robot.episodio_metrica
                    pasos = step_count if step_count > 0 else 1
                    eficiencia = met['distancia_total'] / pasos
                    colisiones_ev = met['colisiones_ev']
                    tiempo_resp = np.mean(met['pasos_respuesta']) if met['pasos_respuesta'] else 0
                    print(f"[MÉTRICAS] Eficiencia trayectoria: {eficiencia:.3f} m/step | Colisiones evitadas: {colisiones_ev} | Tiempo resp. medio: {tiempo_resp:.2f} steps")
                # Reiniciar métricas
                robot.episodio_metrica = {
                    'distancia_total': 0.0,
                    'gps_anterior': None,
                    'colisiones_amenazadas': 0,
                    'colisiones_ev': 0,
                    'obstaculo_cerca': False,
                    'pasos_respuesta': [],
                    'pasos_desde_amenaza': 0,
                    'respondiendo': False
                }
                episode_count += 1
                step_count = 0  # Reinicia el contador de steps
                # Reinicia métricas de trayectoria y colisiones
                traj_total_distance = 0.0
                traj_start_pos = None
                traj_last_pos = None
                traj_efficiency = 0.0
                colisiones_amenazadas = 0
                colisiones_ev = 0
                colisiones = 0
                respuesta_tiempos = []
                respuesta_en_progreso = False
                respuesta_step_inicio = 0
                respuesta_min_lidar_antes = None
                # REINICIA EL ÁNGULO DE GIRO PARA EL NUEVO EPISODIO
                last_steering_angle = 0.0
                # Sensores
                current_steering = left_steer.getTargetPosition() if hasattr(left_steer, 'getTargetPosition') else 0.0
                current_speed = (left_rear_wheel.getVelocity() + right_rear_wheel.getVelocity()) / 2.0
                obs_velocidad = min(max(current_speed, 0), 250)
                lidar_values = lidar.getRangeImage()
                min_lidar = min(lidar_values) if lidar_values else 0.0
                # Mostrar distancia y localización del obstáculo más cercano (sin ángulo)
                if lidar_values:
                    min_index = np.argmin(lidar_values)
                    lidar_count = len(lidar_values)
                    angle_per_ray = 360.0 / lidar_count
                    min_angle = min_index * angle_per_ray  # 0° = frente
                    # Determinar localización
                    if (min_angle <= 45 or min_angle >= 315):
                        loc = "frente"
                    elif 45 < min_angle <= 135:
                        loc = "izquierda"
                    elif 135 < min_angle <= 225:
                        loc = "atrás"
                    else:
                        loc = "derecha"
                    print(f"[LIDAR] Obstáculo más cercano: {min_lidar:.2f} m | Localización: {loc}")
                bumper_value = bumper.getValue()
                gps_position = gps.getValues()  # [x, y, z]
                print(f"[GPS] Posición: x={gps_position[0]:.2f}, y={gps_position[1]:.2f}, z={gps_position[2]:.2f}")
                # Determinar dirección del obstáculo más cercano
                obstacle_direction = 0  # 0: ninguno, 1: frente, 2: izquierda, 3: derecha
                if lidar_values:
                    min_index = np.argmin(lidar_values)
                    lidar_count = len(lidar_values)
                    angle_per_ray = 360.0 / lidar_count
                    min_angle = min_index * angle_per_ray
                    if (min_angle <= 45 or min_angle >= 315):
                        obstacle_direction = 1  # frente
                    elif 45 < min_angle <= 135:
                        obstacle_direction = 2  # izquierda
                    elif 225 < min_angle < 315:
                        obstacle_direction = 3  # derecha
                    else:
                        obstacle_direction = 0  # otro/no relevante
                obs_dict = {'velocidad': obs_velocidad, 'steering': current_steering, 'min_lidar': min_lidar, 'bumper': bumper_value, 'gps': gps_position, 'obstacle_direction': obstacle_direction}
                # Envía la observación inicial al cliente
                send_msg(conn, pickle.dumps(obs_dict))
            elif msg['cmd'] == 'step':
                step_count += 1
                
                # Inicializa la recompensa para este paso
                reward = 0.0
                terminated = False
                truncated = False
                info = {}

                steering = msg['action'][0]
                # Si la acción incluye velocidad, úsala; si no, 30 km/h
                if len(msg['action']) > 1:
                    velocidad_objetivo = float(msg['action'][1])
                else:
                    velocidad_objetivo = VEL_30KMH_RAD_SEC
                print(f"[DEBUG] Acción recibida: steering={steering}, velocidad objetivo={velocidad_objetivo:.1f} rad/s")
                max_steering = 0.5
                steering = np.clip(steering, -max_steering, max_steering)
                left_steer.setPosition(steering)
                right_steer.setPosition(steering)
                
                # Por seguridad, limita la velocidad objetivo al máximo permitido por el motor
                velocidad_objetivo = np.clip(velocidad_objetivo, 0, 138.889)
                left_rear_wheel.setPosition(float('inf'))
                right_rear_wheel.setPosition(float('inf'))
                left_rear_wheel.setVelocity(velocidad_objetivo)
                right_rear_wheel.setVelocity(velocidad_objetivo)
                # Sensores
                current_steering = left_steer.getTargetPosition() if hasattr(left_steer, 'getTargetPosition') else 0.0
                right_steering = right_steer.getTargetPosition() if hasattr(right_steer, 'getTargetPosition') else 0.0
                # Velocidad angular real (rad/s)
                current_speed = (left_rear_wheel.getVelocity() + right_rear_wheel.getVelocity()) / 2.0
                lidar_values = lidar.getRangeImage()
                min_lidar = min(lidar_values) if lidar_values else 0.0
                bumper_value = bumper.getValue()
                gps_position = gps.getValues()  # [x, y, z]
                
                # Limita la velocidad reportada en la observación al rango [0, 250]
                obs_velocidad = min(max(current_speed, 0), 250)
                # Determinar dirección del obstáculo más cercano
                obstacle_direction = 0  # 0: ninguno, 1: frente, 2: izquierda, 3: derecha
                if lidar_values:
                    min_index = np.argmin(lidar_values)
                    lidar_count = len(lidar_values)
                    angle_per_ray = 360.0 / lidar_count
                    min_angle = min_index * angle_per_ray
                    if (min_angle <= 45 or min_angle >= 315):
                        obstacle_direction = 1  # frente
                    elif 45 < min_angle <= 135:
                        obstacle_direction = 2  # izquierda
                    elif 225 < min_angle < 315:
                        obstacle_direction = 3  # derecha
                    else:
                        obstacle_direction = 0  # otro/no relevante

                # --- LÓGICA DE RECOMPENSA UNIFICADA Y CORREGIDA ---
                # Penalización por cambio brusco de dirección
                delta_steering = abs(steering - last_steering_angle)
                abrupt_change_threshold = 0.2
                if delta_steering > abrupt_change_threshold:
                    penalty = -20.0 * (delta_steering - abrupt_change_threshold)
                    reward += penalty
                    print(f"[PENALIZACIÓN] Giro brusco detectado. Penalización: {penalty:.2f}")

                # Penalizaciones por obstáculos cercanos
                if lidar_values:
                    lidar_count = len(lidar_values)
                    angle_per_ray = 360.0 / lidar_count
                    front_indices = [i for i in range(lidar_count) if (i * angle_per_ray <= 45 or i * angle_per_ray >= 315)]
                    left_indices = [i for i in range(lidar_count) if 45 < i * angle_per_ray <= 135]
                    right_indices = [i for i in range(lidar_count) if 225 < i * angle_per_ray < 315]
                    
                    penalty = 0.0
                    if any(lidar_values[i] <= 0.3 for i in front_indices):
                        penalty -= 5.0
                        print("[LIDAR] Penalización: obstáculo muy cerca al frente")
                    if any(lidar_values[i] <= 0.3 for i in left_indices):
                        penalty -= 3.0
                        print("[LIDAR] Penalización: obstáculo muy cerca a la izquierda")
                    if any(lidar_values[i] <= 0.3 for i in right_indices):
                        penalty -= 3.0
                        print("[LIDAR] Penalización: obstáculo muy cerca a la derecha")
                    reward += penalty

                # Recompensa por avanzar rápido y penalización por estar quieto
                min_speed_kmh = 2.0
                current_speed_kmh = current_speed * 0.18
                reward += 4.0 * current_speed_kmh # Recompensa por avanzar
                if current_speed_kmh < min_speed_kmh:
                    reward -= 10.0 # Penaliza mucho más estar parado
                
                # Penalización por girar demasiado
                reward -= 2.0 * abs(current_steering)

                # Finaliza el episodio si está demasiado cerca de un objeto
                min_distancia_segura = 0.5  # metros
                if min_lidar < min_distancia_segura:
                    reward -= 5.0  # Penalización por acercarse demasiado
                    terminated = True
                
                # Penalización moderada por colisión
                if bumper_value > 0:
                    reward -= 100.0
                    terminated = True
                
                # Detecta si el coche está atascado (la posición GPS no cambia mucho)
                if 'gps' in locals():
                    if not hasattr(robot, 'prev_gps_position'):
                        robot.prev_gps_position = gps_position
                        robot.stuck_counter = 0
                    else:
                        delta_gps = [abs(gps_position[i] - robot.prev_gps_position[i]) for i in range(3)]
                        if sum(delta_gps) < 0.05:
                            robot.stuck_counter += 1
                        else:
                            robot.stuck_counter = 0
                        robot.prev_gps_position = gps_position
                        if robot.stuck_counter > 10:
                            print("[INFO] El coche parece atascado. Episodio terminado por atasco.")
                            reward -= 100.0
                            terminated = True
                
                # Lógica de recompensa por carril
                cam_image = camera.getImage()
                if cam_image:
                    image_array = np.frombuffer(cam_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
                    road_color = np.array([85, 85, 85])
                    off_road_color = np.array([140, 131, 120])
                    lane_reward, off_road_terminated = get_lane_deviation_reward(image_array, road_color, off_road_color)
                    reward += lane_reward
                    if off_road_terminated:
                        terminated = True

                # Actualiza el ángulo de dirección para el próximo paso
                last_steering_angle = steering

                # Lógica de métricas personalizadas
                met = robot.episodio_metrica
                if met['gps_anterior'] is not None:
                    dist = np.linalg.norm(np.array(gps_position) - np.array(met['gps_anterior']))
                    met['distancia_total'] += dist
                met['gps_anterior'] = gps_position
                
                amenaza = min_lidar < 0.5
                if amenaza and bumper_value == 0:
                    if not met['obstaculo_cerca']:
                        met['colisiones_amenazadas'] += 1
                        met['obstaculo_cerca'] = True
                        met['respondiendo'] = True
                        met['pasos_desde_amenaza'] = 0
                if met['obstaculo_cerca']:
                    met['pasos_desde_amenaza'] += 1
                    if min_lidar > 0.6:
                        met['colisiones_ev'] += 1
                        met['pasos_respuesta'].append(met['pasos_desde_amenaza'])
                        met['obstaculo_cerca'] = False
                        met['respondiendo'] = False
                        met['pasos_desde_amenaza'] = 0
                if not amenaza:
                    met['obstaculo_cerca'] = False
                    met['respondiendo'] = False
                    met['pasos_desde_amenaza'] = 0
                
                # Logging solicitado
                print(f"Ep:{episode_count-1} | Step:{step_count} | Reward:{reward:.2f} | GPS:({gps_position[0]:.2f},{gps_position[1]:.2f},{gps_position[2]:.2f}) | Bumper:{bumper_value}")
                
                # Envío de la observación, recompensa, etc.
                obs_dict = {'velocidad': obs_velocidad, 'steering': current_steering, 'min_lidar': min_lidar, 'bumper': bumper_value, 'gps': gps_position, 'obstacle_direction': obstacle_direction}
                send_msg(conn, pickle.dumps((obs_dict, reward, terminated, truncated, info)))
            
            elif msg['cmd'] == 'stop_simulation':
                print("[INFO] Comando 'stop_simulation' recibido. Deteniendo la simulación de Webots.")
                robot.simulationQuit()
                break
            
            elif msg['cmd'] == 'close':
                print("[INFO] Comando 'close' recibido. Cerrando conexión y esperando nueva conexión.")
                conn.close()
                conn = None
                continue
    except Exception as e:
        print(f"[ERROR] Excepción en el bucle principal: {e}")
        traceback.print_exc()
        if conn is not None:
            try:
                conn.close()
            except:
                pass
            conn = None
        continue