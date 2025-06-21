"""
Pacote `power_nlp` 

módulo para resolução de despacho contínuo com funções de custo não lineares.

Este pacote fornece:
- A classe `DespachoNLP` para resolução de problemas NLP com IPOPT.
- Utilitários para cálculo de índices de priorização (ISB, ISC, ISD).
- Funções auxiliares para definição de status ON/OFF, construção de z_fixo e Pg ótimo.

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from . import model_nlp

__all__ = ["model_nlp"]
