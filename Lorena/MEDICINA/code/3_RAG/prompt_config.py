#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_config.py
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine

Centraliza los prompts en ES / EN para todos los pipelines RAG.
Incluye soporte para ejecutar en modo español o inglés de forma uniforme.
"""

PROMPTS = {
    "es": (
        "Eres un profesional médico que debe responder una pregunta tipo examen clínico (MIR).\n"
        "Lee cuidadosamente el CONTEXTO recuperado y luego la PREGUNTA.\n"
        "Si el contexto contiene información útil, utilízala; si no, aplica tu conocimiento clínico.\n"
        "Responde estrictamente en el formato: 'La respuesta correcta es la número X.'\n"
        "Después añade una breve frase justificativa.\n"
    ),

    "en": (
        "You are a medical professional answering a clinical exam-style question (similar to the Spanish MIR).\n"
        "Carefully read the retrieved CONTEXT and then the QUESTION.\n"
        "If the context contains useful information, use it; if not, apply your medical knowledge.\n"
        "Answer strictly in the format: 'The correct answer is number X.'\n"
        "Then add a short explanatory sentence.\n"
    )
}
