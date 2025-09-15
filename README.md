Entrenamiento de un Agente de Refuerzo para Control de Vehículo en Webots

Este proyecto utiliza el aprendizaje por refuerzo para entrenar un agente de IA que controla un vehículo en el simulador de robots Webots. El objetivo principal es que el agente aprenda a navegar de manera efectiva en un entorno virtual. El código se basa en las bibliotecas Gymnasium y Stable Baselines3, y se comunica con el simulador Webots a través de sockets.

Componentes del Proyecto
El proyecto consta de los siguientes archivos principales:

webots_car_env.py: Define el entorno de Gymnasium que actúa como puente de comunicación entre el agente de Python y el simulador Webots. Se encarga de enviar acciones y recibir observaciones, recompensas y la condición de finalización del episodio. Incluye una funcionalidad opcional para incorporar una acción sugerida por un Control Predictivo de Modelo (MPC) en la observación del agente, lo que puede guiar el aprendizaje.

train_webots_car.py: El script principal para entrenar el modelo. Aquí se configura el entorno y el modelo de aprendizaje por refuerzo PPO (Proximal Policy Optimization). Incluye un callback para detener el entrenamiento anticipadamente si el rendimiento del agente deja de mejorar. Después del entrenamiento, evalúa el modelo y exporta los resultados a un archivo de Excel llamado entrenamiento_webots.xlsx.

train_car_MPC.py: Una versión alternativa para el entrenamiento que se enfoca en el uso del MPC. Utiliza un callback de registro de episodios personalizado (MPCLoggingCallback) para registrar los datos del entrenamiento y detener el proceso automáticamente después de un número fijo de episodios, exportando los resultados a un archivo training_mpc_logs.xlsx.

test_webots_car.py: Un script simple para cargar y probar un modelo PPO previamente entrenado en el entorno de Webots.

requirements.txt: Enumera las dependencias de Python necesarias para ejecutar el proyecto.

Requisitos Previos
Asegúrate de tener instalado el simulador Webots y configurado un mundo con un vehículo y un controlador que se comunique con el script de Python a través de sockets en el puerto 10000.

Para las dependencias de Python, puedes instalarlas con el siguiente comando:

pip install -r requirements.txt

1. Configurar y Ejecutar Webots
Inicia el simulador Webots.

Abre el mundo virtual donde está el vehículo que será controlado.

Asegúrate de que el controlador del vehículo en Webots esté configurado para establecer una conexión de socket con el script de Python.

2. Entrenar al Agente
Para comenzar el entrenamiento, ejecuta el script principal:

python train_webots_car.py

O, si prefieres usar la versión con el callback de MPC:

python train_car_MPC.py

El script imprimirá el progreso en la consola. Al finalizar, el modelo entrenado se guardará como ppo_webots_car.zip (o ppo_webots_mpc.zip si usas el segundo script), y se generará un archivo de Excel con los registros del entrenamiento.

3. Probar el Agente Entrenado
Una vez que tengas un modelo guardado, puedes probar su rendimiento ejecutando:

python test_webots_car.py

Este script cargará el modelo y lo ejecutará en el entorno de Webots de manera determinista, lo que te permitirá ver cómo el agente navega.

Estructura de la Observación y la Acción

Espacio de Observación: El agente recibe un vector que puede variar en tamaño dependiendo de la configuración del entorno (mpc_in_obs).

Observación normal: [steering, min_lidar, bumper, velocidad, obstacle_direction]

Observación extendida (con MPC): [steering, min_lidar, bumper, velocidad, obstacle_direction, mpc_steering, mpc_velocidad]

Espacio de Acción: El agente controla dos parámetros del vehículo.

Acción: [steering_value, velocity_value]