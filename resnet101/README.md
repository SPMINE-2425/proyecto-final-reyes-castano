# Implementación de ResNet-101

## Introducción

**ResNet (Residual Network)**, introducida por Kaiming He et al. en 2015, revolucionó el aprendizaje profundo al permitir el entrenamiento de **redes neuronales convolucionales muy profundas**.  
La idea clave es el uso de **conexiones residuales (skip connections)**, que alivian el problema del desvanecimiento del gradiente y permiten la optimización efectiva de arquitecturas que superan las 100 capas.

Este repositorio implementa **ResNet-101 desde cero en PyTorch**, aplicada al **Oxford-IIIT Pet dataset** (clasificación binaria: gatos vs perros).  
Aspectos destacados:

- Arquitectura basada en **bloques residuales** con atajos de identidad y convolución.  
- Diseño modular para **carga de datos, entrenamiento y evaluación**.  
- Integración de **utilidades de visualización** para filtros, activaciones, mapas de características y métricas de desempeño.  
- Configurable mediante **archivo YAML** para garantizar reproducibilidad.

---

## Estructura del Proyecto

El repositorio está organizado en componentes modulares:

### 1. `src/data/`

Utilidades para manejo de datos.

- **`load_data.py`** – define los cargadores para el dataset Oxford Pets.  
- **`utils_data.py`** – funciones auxiliares (cálculo de media/desviación estándar, preprocesamiento).  

### 2. `src/model/`

Implementación del modelo ResNet-101.

- **`resnet_bloks.py`** – definición de los bloques residuales.  
- **`restnet.py`** – arquitectura principal de ResNet (profundidad personalizable).  
- **`save_model.py`** – utilidad para guardar modelos entrenados.  

### 3. `src/training/`

Pipeline de entrenamiento.

- **`train_loop.py`** – ciclo principal de entrenamiento.  
- **`train_utils.py`** – funciones auxiliares de optimización y logging.  

### 4. `src/testing_utils/`

Evaluación y visualización.

- **`evaluate_model.py`** – ciclo de evaluación (pérdida, ROC-AUC, etc.).  
- **`evaluate_plots.py`** – generación de gráficas (ROC, matrices de confusión, curvas).  
- **`test_utils.py`** – visualización de mapas de características, Grad-CAM, filtros.  

### 5. `test/`

Scripts de verificación básica.

- **`data_sanity_cheks.py`** – valida la integridad del dataset.  
- **`model_sanity_cheks.py`** – prueba el paso forward del modelo.  

### 6. `experiments/`

Resultados visuales generados durante el entrenamiento/evaluación:

- Mapas de características (visualización por capa).  
- Filtros aprendidos (Conv1, Conv2, Layer4).  
- Curva ROC.  
- Predicciones en el conjunto de prueba.  

### 7. Configuración y Notebooks

- **`oxford_pets_binary_resnet101.yaml`** – configuración del experimento (datos, modelo, optimizador, scheduler, resultados).  
- **`Resnet101.ipynb`** – notebook de Jupyter con el flujo completo de entrenamiento.  

---

## Configuración del Entrenamiento

El entrenamiento se configura a través de `oxford_pets_binary_resnet101.yaml`:

- **Dataset**: Oxford-IIIT Pet (tarea binaria: gato vs perro).  
- **Modelo**: tipo ResNet-101 (`blocks_per_stage=(3,4,23,3)`).  
- **Dispositivo**: CUDA si está disponible, en su defecto CPU.  
- **Optimizador**: Adam (`lr=1e-3`, `betas=(0.9,0.999)`, `weight_decay=1e-4`).  
- **Scheduler**: StepLR (`step_size=30, gamma=0.1`).  
- **Épocas**: 30.  
- **Normalización**: mean = `[0.4829, 0.4449, 0.3957]`, std = `[0.2592, 0.2532, 0.2598]`.  

---

## Resultados

Rendimiento final en validación:

- **Pérdida de validación (Val Loss)**: 0.4084  
- **ROC-AUC**: 0.9108  

**Reporte de Clasificación**:

| Clase        | Precisión | Recall | F1-Score   | Soporte |
| ------------ | --------- | ------ | ---------- | ------- |
| 0 (gato)     | 0.6840    | 0.8750 | 0.7678     | 240     |
| 1 (perro)    | 0.9301    | 0.8044 | 0.8627     | 496     |
| **Exactitud**|           |        | **0.8274** | 736     |

- **Promedio Macro**: Precisión 0.8071, Recall 0.8397, F1 0.8153  
- **Promedio Ponderado**: Precisión 0.8498, Recall 0.8274, F1 0.8318  

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). _Deep Residual Learning for Image Recognition_. CVPR.
- Oxford-IIIT Pet Dataset: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
