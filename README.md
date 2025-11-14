 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..12c747744182530b59239fdb3ad05c5a5057b165
--- /dev/null
+++ b/README.md
@@ -0,0 +1,73 @@
+# Proyecto TFM – Evaluación de modelos LLM para el examen MIR
+
+Este repositorio recopila el material desarrollado para el Trabajo Fin de Máster de Lorena Ariceta García. El objetivo es evaluar modelos de lenguaje (LLM) generativos en preguntas del examen MIR y explorar técnicas de recuperación aumentada con conocimiento médico.
+
+## Contenido del repositorio
+
+La carpeta principal `Lorena/` contiene todos los datos y código utilizados durante el proyecto:
+
+- `MEDICINA/`
+  - `code/`
+    - `1_data_preparation/`: automatiza la descarga de exámenes oficiales y la conversión a estructuras JSON listas para su análisis.
+    - `2_models/`: orquesta la inferencia de distintos LLM a través de Ollama y calcula métricas de rendimiento.
+    - `3_RAG/` y `3_RAG_biomed/`: pipelines de *Retrieval-Augmented Generation* que consultan Wikipedia, PubMed y literatura médica para mejorar las respuestas.
+  - `examenes_mir_v_0/`: copia local de los cuadernos oficiales descargados.
+  - `results/`: salidas intermedias y finales (JSON con respuestas, métricas en CSV/XLSX, etc.).
+- `results_prompt.xlsx`, `results_rag.xlsx`: resúmenes globales del rendimiento de los modelos.
+- `TFM_Lorena_Ariceta.pptx`, `INDEX.docx`: documentación de apoyo al trabajo académico.
+
+## Requisitos
+
+El proyecto se ha ejecutado con Python 3.8+ e incluye scripts que necesitan:
+
+- `pandas`, `numpy`
+- `requests`
+- `selenium`, `webdriver-manager`
+- `beautifulsoup4`, `lxml` (para extracción puntual de contenido web)
+- `transformers`, `torch` (para carga de modelos Hugging Face en los pipelines RAG)
+- Servidor [Ollama](https://ollama.com/) ejecutándose en `http://localhost:11434` con los modelos `llama3`, `mistral`, `gemma`, `deepseek-coder`, `phi3`, entre otros.
+
+> **Nota:** Algunos scripts usan rutas absolutas (`/home/xs1/Desktop/...`). Sustitúyelas por rutas válidas en tu entorno antes de ejecutarlos.
+
+## Puesta en marcha
+
+1. Crea y activa un entorno virtual:
+   ```bash
+   python3 -m venv .venv
+   source .venv/bin/activate
+   ```
+2. Instala las dependencias principales:
+   ```bash
+   pip install -U pandas numpy requests selenium webdriver-manager beautifulsoup4 lxml transformers torch
+   ```
+3. Descarga y pon en marcha Ollama; importa los modelos necesarios, por ejemplo:
+   ```bash
+   ollama pull llama3
+   ollama pull mistral
+   # ...
+   ```
+4. Ajusta en los scripts de `code/` las rutas de entrada/salida para tu máquina.
+
+## Flujo de trabajo recomendado
+
+1. **Descarga de exámenes:** `1_data_preparation/1_extract_pdf_from_ministerio.py` usa Selenium para descargar automáticamente los cuadernos oficiales desde la web del Ministerio y los guarda en `examenes_mir_v_0/`.
+2. **Conversión a JSON:** scripts numerados del mismo directorio limpian y estructuran las preguntas, añadiendo metadatos como respuesta correcta, tipo de pregunta o año.
+3. **Inferencia con LLM:** los scripts de `2_models/` generan respuestas para cada examen llamando a la API local de Ollama y generan resúmenes por modelo (`results/2_models/...`).
+4. **RAG:** los orquestadores `3_RAG/RAG_main.py` y `3_RAG_biomed/RAG_main.py` ejecutan pipelines que recuperan contexto de Wikipedia, PubMed o libros médicos antes de preguntar al modelo. Las métricas se consolidan en `results/summary/`.
+5. **Análisis final:** los ficheros Excel del directorio `results/` ofrecen comparativas globales de aciertos, errores y cobertura.
+
+## Métricas y salidas
+
+- Cada ejecución de modelos produce un JSON con las respuestas propuestas y logs con la trazabilidad de la inferencia.
+- Las métricas agregadas (exactitud, preguntas respondidas, ejemplos de error) se encuentran en CSV/XLSX bajo `results/2_models/` y `results/summary/`.
+- Los notebooks y presentaciones incluidos ayudan a contextualizar los resultados y conclusiones del TFM.
+
+## Contribuir o reutilizar
+
+Si deseas adaptar el proyecto:
+
+1. Revisa los prompts definidos en `code/2_models` y `code/3_RAG/prompt_config.py` para ajustarlos a la titulación o idioma deseado.
+2. Añade nuevas fuentes de conocimiento creando módulos en `code/3_RAG/utils/`.
+3. Estandariza rutas relativas para facilitar la ejecución en otros equipos.
+
+¡Disfruta explorando los resultados y extendiendo el análisis de modelos LLM en el contexto del examen MIR!
 
EOF
)
