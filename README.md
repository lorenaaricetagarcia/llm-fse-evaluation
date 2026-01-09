# TFM_Final

This repository contains the code and artifacts for a TFM thesis on evaluating language models for MIR-style medical exam questions, including data preparation, prompt-based model evaluation, and retrieval-augmented generation (RAG) pipelines.

## Repository layout

- `code/1_data_preparation/` — Scripts to extract MIR PDFs, convert them to JSON, and enrich questions with correct answers, type (text/image), and metadata.
- `code/2_models/` — Prompting experiments and metrics computation for model outputs.
- `code/3_RAG/` — Wikipedia/PubMed RAG pipelines plus utilities and prompt configuration.
- `code/requirements.txt` — Python dependencies for the data/model/RAG workflows.
- `results/` — Output artifacts produced by the scripts (not all results are committed).

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r code/requirements.txt
   ```

Some workflows also rely on external tooling (e.g., Selenium WebDriver for scraping) and large models that may need to be installed separately.

## Data preparation

The data preparation pipeline is organized as a series of scripts in `code/1_data_preparation/` that are typically run in order:

1. `1_extract_pdf_from_ministerio.py` — Download/extract official MIR PDFs.
2. `2_convert_all_mir_pdf_to_json.py` — Convert PDFs into structured JSON.
3. `3_extract_correct_answer.py` — Parse the correct answers.
4. `4_add_correct_answer_to_json.py` — Merge answers into JSON files.
5. `5_remove_instructions.py` — Remove instruction-only pages from exams.
6. `6_add_type_text_or_image.py` — Tag questions as text or image-based.
7. `7_add_titulation_year.py` — Add metadata like titulation year.

Each script is intended to be run from the repository root. Adjust any hard-coded paths (see **Path configuration** below).

## Prompt-based model evaluation

The prompt experiments live in `code/2_models/`:

- `1_no_prompt.py`, `2_prompt_es.py`, `3_prompt_deepseek_phi3.py`, `4_prompt_en.py` — Run evaluations with different prompt variants.
- `metrics.py` — Computes per-exam accuracy metrics and exports CSV/Excel summaries.

## RAG pipelines

RAG workflows are under `code/3_RAG/` and include Wikipedia and PubMed variants. A helper script provides an interactive selection menu:

```bash
python code/3_RAG/RAG_main_selective.py
```

It lets you pick which models and pipelines to run and writes metrics to the `results/3_rag/` folder.

## Path configuration

Several scripts define a `BASE_DIR` constant pointing to a local machine path (for example `/home/xs1/Desktop/Lorena`). Update that variable to match your environment before running any pipelines. This applies to:

- `code/2_models/metrics.py`
- `code/3_RAG/RAG_main_selective.py`
- Additional scripts in the data-preparation and RAG directories

Search for `BASE_DIR` in the repository to locate all path definitions.

## Results

Outputs are written to subdirectories under `results/`, including:

- `results/1_data_preparation/` — JSON datasets for MIR exams.
- `results/2_models/` — Model outputs and prompt metrics.
- `results/3_rag/` — RAG run outputs and summary metrics.

## Notes

- Some scripts expect GPUs or large local models (e.g., `llama3`, `mistral`, `gemma`, `medllama2`). Ensure the runtime environment is configured accordingly.
- For reproducibility, keep the directory structure under `results/` consistent with the scripts’ expectations.
