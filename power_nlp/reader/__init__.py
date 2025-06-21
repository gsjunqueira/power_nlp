"""
Pacote `reader`

Este pacote contém funções especializadas para leitura de arquivos .m do MATLAB
utilizados em estudos de sistemas elétricos, como definição de unidades geradoras (DGER)
e curvas de carga horária (DLOAD). As funções extraem os dados desses blocos e os
transformam em estruturas Python (listas de dicionários) para posterior uso em modelos.

Módulos:
- read_m.py: implementa as funções de parsing de arquivos MATLAB .m para estruturas Python.

Funções exportadas:
- ler_dger_m
- ler_dload_m
- ler_uc_10ger_m

Autor: Giovani Santiago Junqueria
"""

__author__ = "Giovani Santiago Junqueira"

from .read_m import ler_dger_m, ler_dload_m, ler_m

__all__ = ["ler_dger_m", "ler_dload_m", "ler_m"]
