import os
import time
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Control de Calidad: Huevos",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🥚 Sistema de Inspección Automática en Banda Transportadora")
st.markdown(
    "**MLOps Pipeline** - Detección de Huevos Dañados en Tiempo Real"
)

# 1. Cargar el Modelo
MODEL_PATH = os.path.join("models", "best.pt")


@st.cache_resource
def load_model():
  return YOLO(MODEL_PATH)


try:
  model = load_model()
  st.sidebar.success("Modelo YOLOv8 cargado correctamente.")
except Exception as e:
  st.sidebar.error(
      f"Error al cargar el modelo desde '{MODEL_PATH}'. Revisa la ruta."
  )
  st.stop()

# 2. Sidebar / Parámetros
st.sidebar.header("Configuración de Banda")
conf_threshold = st.sidebar.slider(
    "Umbral de Confianza (Confidence)", 0.1, 1.0, 0.40, 0.05
)

# 3. Métricas Principales
col_video, col_stats = st.columns([3, 2])

with col_stats:
  st.subheader("📊 Métricas de Producción")
  m_total = st.metric("Total Procesados", 0)
  m_sanos = st.metric("Huevos Sanos", 0)
  m_danados = st.metric("Huevos Dañados (Rechazo)", 0)

with col_video:
  st.subheader("🎥 Simulación de Cámara en Banda")
  uploaded_video = st.file_uploader(
      "Cargar video de simulación (.mp4, .avi)", type=["mp4", "avi", "mov"]
  )

if uploaded_video is not None:
  # Guardar temporalmente el video para procesarlo con OpenCV
  temp_video_path = "temp_input_video.mp4"
  with open(temp_video_path, "wb") as f:
    f.write(uploaded_video.read())

  cap = cv2.VideoCapture(temp_video_path)
  st_frame = st.empty()

  total_count = 0
  sanos_count = 0
  danados_count = 0

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Inferencia con YOLOv8
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    # Extraer clases detectadas
    boxes = results[0].boxes
    if boxes is not None:
      for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()

        # Ajusta estas etiquetas según las clases de tu dataset en Roboflow
        if "dan" in class_name or "crack" in class_name or "defect" in class_name:
          danados_count += 1
        else:
          sanos_count += 1
        total_count += 1

    # Actualizar Dashboard
    st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
    m_total.metric("Total Procesados", total_count)
    m_sanos.metric("Huevos Sanos", sanos_count)
    m_danados.metric("Huevos Dañados (Rechazo)", danados_count)

    time.sleep(0.01)  # Controlar la velocidad de simulación

  cap.release()
  if os.path.exists(temp_video_path):
    os.remove(temp_video_path)