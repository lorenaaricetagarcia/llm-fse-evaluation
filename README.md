# AI-Based Diagnosis: Ally or Risk? — Master’s Thesis Project

This repository contains the full research pipeline for a Master’s thesis on
**language-model performance in Spanish medical residency examinations (FSE)**.
It covers data collection, preprocessing, model evaluation (prompting), and
retrieval-augmented generation (RAG) experiments. The work is centered on
**Version 0** FSE exam booklets from the Spanish Ministry of Health and evaluates
how different LLM prompting strategies and RAG pipelines affect accuracy on
text-only questions.

## Thesis scope and objectives

**Research theme:** AI-based clinical decision support and the reliability of
large language models when confronted with standardized medical exams.

**Primary objectives:**
- Build a reproducible dataset from official FSE exam booklets.
- Normalize the data into machine-readable JSON with correct-answer labels.
- Benchmark multiple LLMs under different prompting schemes.
- Evaluate RAG pipelines (Wikipedia and PubMed) for knowledge grounding.
- Report per-exam and aggregate accuracy metrics.

## Repository structure

```
.
├── FSE_exams_v0/              # Downloaded FSE Version 0 PDFs (created by scripts)
├── code/
│   ├── 1_data_preparation/    # Data collection + transformation scripts
│   ├── 2_models/              # Prompting experiments and metrics
│   ├── 3_rag/                 # RAG pipelines (Wikipedia/PubMed)
│   └── requirements.txt       # Python dependencies
├── results/
│   ├── 1_data_preparation/    # JSON outputs per preprocessing step
│   ├── 2_models/              # Model predictions + prompt metrics
│   ├── 3_rag/                 # RAG outputs + metrics
│   └── summary_files/         # Aggregated summaries for reporting
└── README.md
```

## Data pipeline (code/1_data_preparation)

The dataset is built in multiple steps:

1. **Download official booklets**
   - `1_extract_pdf_from_ministerio.py` uses Selenium to download **Version 0**
     text and image booklets per specialization and year from the official
     Ministry of Health portal.
2. **Convert PDFs to JSON**
   - `2_convert_all_fse_pdf_to_json.py` extracts raw text from PDFs and converts
     them to JSON format under `results/1_data_preparation/1_json_specialization/`
     using the base keys `titulacion`, `preguntas`, `numero`, `enunciado`,
     `opciones`, and `archivo_origen`.
3. **Extract correct answers**
   - `3_extract_correct_answer.py` parses the answer keys from exam documents.
4. **Merge correct answers into JSON**
   - `4_add_correct_answer_to_json.py` injects `respuesta_correcta` plus
     per-question `titulacion` and `convocatoria` metadata.
5. **Remove administrative instructions**
   - `5_remove_instructions.py` cleans non-question boilerplate.
6. **Classify question modality**
   - `6_add_type_text_or_image.py` tags each question as `tipo: "texto"` or
     `tipo: "imagen"`.
7. **Add metadata**
   - `7_add_titulation_year.py` adds dataset-level `etiqueta_titulacion` and
     `etiqueta_convocatoria` fields.

Outputs for each step are stored under `results/1_data_preparation/` to preserve
intermediate artifacts for traceability and auditability.

## Prompting experiments (code/2_models)

The model-evaluation pipeline compares multiple prompting strategies:

- `1_no_prompt.py` — baseline with no instruction prompt.
- `2_prompt_es.py` — Spanish instruction prompt.
- `3_prompt_en.py` — English instruction prompt.
- `metrics.py` — computes accuracy per exam and produces CSV/Excel summaries.

**Key metric:** accuracy on **text-only questions** (image-based questions are
excluded), computed as `correct / answered` with `None` counted as unanswered.

The metrics script now supports configurable paths via CLI flags or
environment variables (defaults are repo-relative):

```bash
python code/2_models/metrics.py \
  --base-dir results/2_models/1_prompt \
  --ground-truth-dir results/1_data_preparation/6_json_final \
  --output-dir results/2_models/1_prompt/metrics
```

Environment variable equivalents:

```bash
FSE_BASE_DIR=results/2_models/1_prompt \
FSE_GROUND_TRUTH_DIR=results/1_data_preparation/6_json_final \
FSE_OUTPUT_DIR=results/2_models/1_prompt/metrics \
python code/2_models/metrics.py
```

## RAG experiments (code/3_rag)

RAG pipelines retrieve evidence from external knowledge sources and feed it into
LLM prompts. Two sources are used:

- **Wikipedia** pipelines (`code/3_rag/1_wikipedia/`)
- **PubMed** pipelines (`code/3_rag/2_pubmed/`)

`RAG_main_selective.py` provides an interactive launcher to select models and
pipelines (v1/v2/v3/final) and runs the experiments while saving metrics.

## Results

All outputs are stored under `results/`, including:

- Cleaned JSON question sets with correct answers.
- Model predictions grouped by prompt type and model.
- RAG outputs and per-exam metrics.
- Summary tables used in the thesis report.

## Environment setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r code/requirements.txt
   ```

## Running the pipeline

> ⚠️ Some scripts assume repo-relative defaults (for example `results/...` or
> `FSE_exams_v0/`). If you run the pipeline from a different working directory,
> set the provided environment variables or update the base paths accordingly.

### Data preparation
Run scripts in order from `code/1_data_preparation/`.

### Prompting experiments
Run any of the prompt scripts in `code/2_models/`, then compute metrics:

```bash
python code/2_models/1_no_prompt.py
python code/2_models/2_prompt_es.py
python code/2_models/3_prompt_en.py
python code/2_models/metrics.py
```

### RAG experiments
Use the interactive runner:

```bash
python code/3_rag/RAG_main_selective.py
```

## Reproducibility notes

- The dataset is derived from official FSE PDFs and is **not bundled** in this
  repository. It must be downloaded via the provided scripts.
- Some RAG scripts may require GPU acceleration or model-specific setup.
- Results are deterministic only if the same model versions and inference
  settings are used.

## Ethical and methodological notes

This project evaluates LLMs on high-stakes medical exam material to assess
reliability and potential risks of AI-assisted decision making. The results are
**research-oriented** and do not constitute medical advice or certification.

## Citation

If you reference this work, please cite it as:

```
Ariceta Garcia, L. (2024). AI-Based Diagnosis: Ally or Risk?
An Analysis of Language Models. Master’s Thesis.
```

## Contact

For questions about the thesis or the pipeline, please contact the author listed
in the script headers.
