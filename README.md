# 🚗 Aprendizaje por Refuerzo para Control de Vehículo en Webots

Este proyecto implementa un agente de **Inteligencia Artificial** que aprende a controlar un vehículo en el simulador **Webots** mediante técnicas de **aprendizaje por refuerzo**. Utiliza las bibliotecas **Gymnasium** y **Stable Baselines3**, comunicándose con Webots a través de **sockets TCP**.

---

## 📚 Tabla de Contenido

- [🔧 Componentes del Proyecto](#-componentes-del-proyecto)
- [⚙️ Requisitos Previos](#️-requisitos-previos)
- [📦 Instalación](#-instalación)
- [🧪 Configuración de Webots](#-configuración-de-webots)
- [🏋️‍♂️ Entrenamiento del Agente](#️-entrenamiento-del-agente)
- [🎯 Evaluación del Modelo](#-evaluación-del-modelo)
- [🧠 Estructura de Observación y Acción](#-estructura-de-observación-y-acción)
- [📌 Notas Finales](#-notas-finales)

---

## 🔧 Componentes del Proyecto

| Archivo                | Descripción |
|------------------------|-------------|
| `webots_car_env.py`    | Define el entorno Gymnasium que conecta el agente con Webots. Incluye soporte opcional para observaciones extendidas con MPC. |
| `train_webots_car.py`  | Script principal para entrenar el modelo PPO. Exporta resultados a `entrenamiento_webots.xlsx`. |
| `train_car_MPC.py`     | Variante con MPC. Usa `MPCLoggingCallback` y exporta a `training_mpc_logs.xlsx`. |
| `test_webots_car.py`   | Carga y prueba un modelo PPO entrenado en Webots. |
| `requirements.txt`     | Lista de dependencias necesarias para ejecutar el proyecto. |

---

## ⚙️ Requisitos Previos

- Tener instalado **Webots**.
- Configurar un mundo con vehículo y controlador que se comunique por **socket TCP** en el **puerto 10000**.

---

## 📦 Instalación

Instala las dependencias necesarias con:

> [!NOTE]
> pip install -r requirements.txt

🧪 Configurar y Ejecutar Webots

- Inicia el simulador Webots.
- Abre el mundo virtual donde está el vehículo que será controlado.
- Asegúrate de que el controlador del vehículo en Webots esté configurado para establecer una conexión de socket con el script de Python.

🏋️‍♂️ Entrenar al Agente

Para comenzar el entrenamiento, ejecuta el script principal:

> [!NOTE]
> python train_webots_car.py

O, si prefieres usar la versión con el callback de MPC:

> [!NOTE]
> python train_car_MPC.py

📌 El script imprimirá el progreso en la consola. Al finalizar:

El modelo entrenado se guardará como ppo_webots_car.zip o ppo_webots_mpc.zip.

Se generará un archivo de Excel con los registros del entrenamiento.

🎯 Probar el Agente Entrenado

Una vez que tengas un modelo guardado, puedes probar su rendimiento ejecutando:

> [!NOTE]
> python test_webots_car.py

Este script cargará el modelo y lo ejecutará en el entorno de Webots de manera determinista, lo que te permitirá ver cómo el agente navega.

🧠 Estructura de la Observación y la Acción
🔍 Espacio de Observación
El agente recibe un vector que puede variar en tamaño dependiendo de la configuración del entorno (mpc_in_obs):

Observación normal: [steering, min_lidar, bumper, velocidad, obstacle_direction]

Observación extendida (con MPC): [steering, min_lidar, bumper, velocidad, obstacle_direction, mpc_steering, mpc_velocidad]

🎮 Espacio de Acción
El agente controla dos parámetros del vehículo:

steering_value

velocity_value

🧠 control.py — Controlador Webots
Este script se ejecuta dentro del simulador Webots y actúa como puente entre el entorno simulado y el agente de aprendizaje por refuerzo. Sus principales funciones incluyen:

-Inicialización de sensores y actuadores del vehículo (GPS, Lidar, cámara, ruedas, dirección, bumper).
-Servidor TCP que espera comandos desde el entorno Gymnasium.
-Gestión de episodios: reinicio de posición, métricas de desempeño, detección de colisiones y obstáculos.

📌 Cálculo de recompensas basadas en:

-Desviación del carril (usando visión por cámara).
-Proximidad a obstáculos (Lidar).
-Colisiones reales (bumper).
-Cambios bruscos de dirección.
-Velocidad y eficiencia de trayectoria.

Comunicación continua con el agente: recibe acciones y envía observaciones, recompensas y estados de terminación.

Este archivo debe estar configurado como controlador del vehículo en Webots, y debe estar vinculado al nodo TESLA_MODEL3 en el mundo .wbt.

El proyecto desarrollado en webots está organizado de la siguiente manera:

<img width="256" height="618" alt="Captura de pantalla 2025-09-15 a la(s) 5 04 11 p m" src="https://github.com/user-attachments/assets/a2aa20de-b3b5-4902-beb6-f8c9b6e09e98" />

Esta estructura incluye:

* Controladores para vehículos autónomos y semáforos.
* Plugins personalizados para visualización y control del automóvil.
* Mundos simulados en Webots, como city.wbt.
* Archivos fuente y ejecutables para compilar y ejecutar los controladores.

📌 Notas Finales

Este proyecto es una excelente base para explorar el uso del aprendizaje por refuerzo en robótica simulada. Puedes extenderlo para incluir otros algoritmos, sensores o entornos más complejos.
