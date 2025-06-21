"""
Pacote `utils`

Este pacote agrupa utilitários diversos utilizados no projeto `power_nlp`.
Inclui funções de conversão de resultados, limpeza de diretórios e carregamento
de dados para o modelo de despacho ótimo de energia elétrica.

Módulos incluídos:

- `clean`: script principal para limpeza dos resultados


Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from .clean import limpar_cache_py


__all__ = ["limpar_cache_py"]
