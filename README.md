# 🐾 ResNet-101: Clasificación de Gato vs Perro con Interpretabilidad

Repositorio de una aplicación completa (API + App) que implementa **ResNet-101** en PyTorch para clasificar imágenes de **gatos** y **perros**, e incorpora un módulo de **interpretabilidad** para entender “qué ve” la red en cada predicción. El foco es proporcionar una herramienta avanzada de clasificacion y entendimiento con codigo altamente reproducible: código modular, configuración por YAML y utilidades de evaluación/visualización.

---

## 🎯 Objetivos

- **Clasificación binaria** (cat vs dog) sobre imágenes del usuario (archivo local o URL).
- **Interpretabilidad avanzada** para inspeccionar la decisión del modelo en cada imagen:
  - **Occlusion Sensitivity**: relevancia por parches.
  - **Integrated Gradients**: atribuciones acumuladas desde una referencia.
  - **Grad-CAM**: mapa de calor clase-específico.
  - **Feature Maps (depth)**: activaciones intermedias por capa.
  - **Kernels (depth)**: filtros aprendidos (p. ej., Conv1).

---

## 🧩 Componentes principales

- **`resnet101/`**: implementación del modelo (desde cero) y artefactos de experimentos.
  - `src/` – arquitectura, bloques residuales, utilidades de guardado.
  - `model_trained/` – pesos entrenados.
  - `experiments/` – resultados y paneles generados.

- **`src/`**: API de inferencia (FastAPI) y utilidades.
  - `api/` – routers (`/health`, `/predict`, `/predict/advanced`), errores, middleware, deps.
  - `inference/` – pipeline de preprocesamiento → forward → postprocesado → validación.
  - `schemas/` – contratos Pydantic v2 para requests/responses (incluye metadatos e imágenes base64).
  - `utils/` – configuración, lectura de variables de entorno, paths, etc.
  - `tests/` – pruebas de contrato y validaciones de entrada (pytest).

- **`app/`**: interfaz de usuario (Streamlit) para subir imágenes/URLs y explorar explicaciones.

- **`data/`**: preparación de datos y estadísticas.
  - `processed/` – carpeta de trabajo (no versionar datos brutos).
  - `pet_stats.json` – medias/desviaciones para normalización reproducible.

- **`notebooks/`**: verificación de flujo de datos y sanity checks del modelo.

- **`oxford_pets_binary_resnet101.yaml`**: configuración del experimento (datos, modelo, optimizador, scheduler, device).

---

## 🖼️ Showcase 

**Vista general de la App (Home / Subida de imagen):**  

<div align="center">
  <img src="app/showcase/Showcase app.png" alt="showcase app" width="620">
</div>

---

**Predicción avanzada (Interpretabilidad por método):**

<!-- 1) Grad-CAM centrado solo -->
<div align="center">
  <strong>Grad-CAM</strong><br>
  <img src="advance_visualization/samples/Grad Cam.png" alt="Grad-CAM" width="620"><br>
  <sub>Mapa de calor clase-específico.</sub>
</div>

---

<!-- 2) Fila: Occlusion | Integrated, ambos centrados -->
<div align="center">
  <table>
    <tr>
      <td align="center" style="padding:12px;">
        <strong>Occlusion Sensitivity</strong><br>
        <img src="advance_visualization/samples/oclusion v2.png" alt="Occlusion Sensitivity" width="360"><br>
        <sub>Importancia local al ocultar parches.</sub>
      </td>
      <td align="center" style="padding:12px;">
        <strong>Integrated Gradients (overlay)</strong><br>
        <img src="advance_visualization/samples/Integrated Gradients.png" alt="Integrated Gradients overlay" width="360"><br>
        <sub>Atribuciones acumuladas superpuestas.</sub>
      </td>
    </tr>
  </table>
</div>

---

<!-- 3) Feature Maps centrado -->
<div align="center">
  <strong>Feature Maps (profundidad/capas)</strong><br>
  <img src="advance_visualization/samples/Feature Maps.png" alt="Feature Maps" width="620"><br>
</div>

<!-- 4) Kernels centrado -->
<div align="center" style="padding-top:8px;">
  <strong>Kernels (filtros aprendidos)</strong><br>
  <img src="advance_visualization/samples/Filters.png" alt="Kernels aprendidos" width="620"><br>
  <sub>Filtros de capas tempranas (bordes, texturas, orientaciones).</sub>
</div>


---

## 🔍 Flujo de inferencia

1. **Entrada**: archivo o URL → validación (MIME/shape).
2. **Preprocesamiento**: resize/center-crop → tensor normalizado (estadísticos cacheados).
3. **Modelo (ResNet-101)**: forward → logits → softmax.
4. **Salida básica**: `label` + `scores` + `meta`.
5. **Salida avanzada**: añade `artifacts` con paneles (PNG base64) para:
   - `gradcam_panel`
   - `integrated_gradients_overlay`
   - `occlusion_overlay`
   - `feature_maps_panel`
   - `kernels_panel`
   (e indicadores de error por panel si aplica.)

---

## 🧠 Interpretabilidad (resumen conceptual)

- **Grad-CAM**: resalta zonas espacialmente relevantes para la clase objetivo a partir de gradientes de capas profundas.
- **Integrated Gradients**: atribuye importancia a cada píxel integrando el gradiente a lo largo de un camino desde una referencia (baseline) hasta la imagen.
- **Occlusion Sensitivity**: mide cómo cambia la confianza al “ocultar” parches de la imagen (búsqueda local de relevancia).
- **Feature Maps**: muestran qué patrones intermedios activan las capas (texturas, bordes, partes).
- **Kernels**: visualiza filtros aprendidos en convoluciones (especialmente capas tempranas).

---

## 📊 Métricas de referencia (validación)

- Pérdida (Val Loss)
- ROC-AUC
- Reporte de clasificación (Precisión/Recall/F1 por clase)
- Matriz de confusión

[ESPACIO RESERVADO PARA TABLA/FIGURA DE MÉTRICAS]

---

## 🔐 Pesos y datos

- Los **pesos del modelo** pueden distribuirse por **Git LFS** o alojarse externamente (HuggingFace/Drive) y enlazarse en el README.
- Los **datos** deben respetar sus licencias; este proyecto usa **Oxford-IIIT Pet** con fines educativos.

---

## 🗺️ Rutas expuestas (API)

- `GET /health` → Estado del servicio.
- `POST /predict` → Predicción base: `label`, `scores`, `meta`.
- `POST /predict/advanced` → Predicción + `artifacts` de interpretabilidad.

[ESPACIO RESERVADO PARA EJEMPLOS DE REQUEST/RESPONSE]

---

## ⚙️ Reproducibilidad

- Configuración centralizada en **YAML** (dataset, normalización, arquitectura, optimizador, scheduler).
- Estadísticos de normalización cacheados en `data/pet_stats.json`.
- Semillas y dispositivos controlados (CPU/CUDA).

---

## 🧭 Roadmap

- Exportación ONNX/TorchScript.
- Batch inference por carpeta/URL list.
- Panel de interpretabilidad adicional (SmoothGrad, Grad-CAM++).  
- Tests más exhaustivos de contrato y regresión visual.

---

## 📄 Licencia y créditos

- Uso **educativo y de investigación**.
- Cita sugerida a **He et al. (2015)** por ResNet y a los trabajos originales de interpretabilidad correspondientes.
- Agradecimientos a la comunidad PyTorch y al dataset Oxford-IIIT Pet.

---

## 📬 Contacto

- Issues y mejoras: usar el **Issue Tracker** del repositorio.
- Preguntas técnicas: abrir un **Discussion** con el tag `help-wanted`.
