"""
Módulo de utilitários para apoio ao modelo de despacho contínuo.

Inclui funções de priorização (ISB, ISC, ISD), geração de status ON/OFF,
e conversão de DataFrame de status para dicionário z_fixo.
Também fornece a classe DespachoNLP para resolução de problemas
de despacho não linear contínuo com Pyomo.

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from .despacho_nlp import DespachoNLP
from .utils import desempenho

__all__ = [
 "DespachoNLP", "desempenho"
]
