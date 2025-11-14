# Proyecto TFM – Evaluación de modelos LLM para el examen MIR

Este repositorio recopila el material desarrollado para el Trabajo Fin de Máster (TFM) de Lorena Ariceta García. El objetivo principal es evaluar modelos de lenguaje generativos (LLM) con preguntas reales del examen MIR y experimentar con técnicas de *Retrieval-Augmented Generation* (RAG) apoyadas en literatura médica.

## Contenido del repositorio

La carpeta principal `Lorena/` agrupa los datos y el código empleados durante el proyecto:

- `MEDICINA/`
  - `code/`
    - `1_data_preparation/`: automatiza la descarga de exámenes oficiales y los transforma a estructuras JSON listas para su análisis.
    - `2_models/`: orquesta la inferencia de distintos LLM a través de Ollama y calcula métricas de rendimiento.
    - `3_RAG/` y `3_RAG_biomed/`: ejecutan pipelines de RAG que consultan Wikipedia, PubMed y otras fuentes médicas para enriquecer las respuestas.
  - `examenes_mir_v_0/`: copia local de los cuadernos oficiales descargados.
  - `results/`: resultados intermedios y finales (respuestas en JSON, métricas en CSV/XLSX, etc.).
- `results_prompt.xlsx`, `results_rag.xlsx`: resúmenes globales del rendimiento de los modelos.
- `TFM_Lorena_Ariceta.pptx`, `INDEX.docx`: documentación de apoyo al trabajo académico.

## Requisitos

El proyecto se ejecutó con Python 3.8+ y hace uso de las siguientes dependencias principales:

- `pandas`, `numpy`
- `requests`
- `selenium`, `webdriver-manager`
- `beautifulsoup4`, `lxml`
- `transformers`, `torch`
- Servidor [Ollama](https://ollama.com/) en `http://localhost:11434` con modelos como `llama3`, `mistral`, `gemma`, `deepseek-coder`, `phi3`, entre otros.

> **Nota:** Algunos scripts contienen rutas absolutas (`/home/xs1/Desktop/...`). Sustitúyelas por rutas válidas en tu entorno antes de ejecutar los pipelines.

## Puesta en marcha

1. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instala las dependencias principales:
   ```bash
   pip install -U pandas numpy requests selenium webdriver-manager beautifulsoup4 lxml transformers torch
   ```
3. Descarga e inicia Ollama e importa los modelos necesarios, por ejemplo:
   ```bash
   ollama pull llama3
   ollama pull mistral
   # ...
   ```
4. Ajusta en los scripts de `code/` las rutas de entrada y salida para tu máquina.

## Flujo de trabajo recomendado

1. **Descarga de exámenes:** `1_data_preparation/1_extract_pdf_from_ministerio.py` utiliza Selenium para descargar los cuadernos oficiales desde la web del Ministerio y los guarda en `examenes_mir_v_0/`.
2. **Conversión a JSON:** los scripts numerados del mismo directorio limpian y estructuran las preguntas, añadiendo metadatos como respuesta correcta, tipo de pregunta o año.
3. **Inferencia con LLM:** los scripts de `2_models/` generan respuestas para cada examen llamando a la API local de Ollama y almacenan resúmenes por modelo en `results/2_models/`.
4. **RAG:** los orquestadores `3_RAG/RAG_main.py` y `3_RAG_biomed/RAG_main.py` recuperan contexto (Wikipedia, PubMed, libros médicos) antes de consultar al modelo. Las métricas consolidadas quedan en `results/summary/`.
5. **Análisis final:** los ficheros Excel del directorio `results/` ofrecen comparativas globales de aciertos, errores y cobertura.

## Métricas y salidas

- Cada ejecución de modelos genera un JSON con las respuestas propuestas y logs con la trazabilidad de la inferencia.
- Las métricas agregadas (exactitud, preguntas respondidas, ejemplos de error) se encuentran en CSV/XLSX bajo `results/2_models/` y `results/summary/`.
- Los notebooks y presentaciones adjuntas ayudan a contextualizar los resultados y conclusiones del TFM.
