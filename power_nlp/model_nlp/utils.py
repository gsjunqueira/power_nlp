"""
Módulo de utilitários para apoio ao modelo de despacho contínuo.

Inclui funções de priorização (ISB, ISC), geração de status ON/OFF,
e conversão de DataFrame de status para dicionário z_fixo.

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

import pandas as pd

def desempenho(fob_isa: float, fob_isb: float, fob_isc: float, fob_isd: float,
               fob_isg: float, fob_ils: float, fob_itr: float, fob_fb: float,
               tempos_execucao: dict) -> pd.DataFrame:
    """
    Monta um DataFrame com os valores da FOB e os tempos de execução de cada etapa
    para os três indicadores de priorização: ISB, ISC e ISD.

    Args:
        fob_isb (float): Valor da função objetivo para ISB.
        fob_isc (float): Valor da função objetivo para ISC.
        fob_isd (float): Valor da função objetivo para ISD.
        tempos_execucao (dict): Dicionário contendo os tempos de execução com as chaves:
            - 'etapa_leitura', 'priorizacao_isb', 'priorizacao_isc', 'priorizacao_isd'
            - 'solucao_isb', 'solucao_isc', 'solucao_isd', 'tempo_total'

    Returns:
        pd.DataFrame: DataFrame com índice ['ISB', 'ISC', 'ISD'] e colunas:
            ['FOB', 't_preparacao', 't_solver', 't_total']
    """
    dados = {
        'ISA': {
            'FOB': fob_isa,
            't_preparacao': tempos_execucao['priorizacao_isa'],
            't_solver': tempos_execucao['solucao_isa'],
            't_total': tempos_execucao['priorizacao_isa'] + tempos_execucao['solucao_isa'],
        },
        'ISB': {
            'FOB': fob_isb,
            't_preparacao': tempos_execucao['priorizacao_isb'],
            't_solver': tempos_execucao['solucao_isb'],
            't_total': tempos_execucao['priorizacao_isb'] + tempos_execucao['solucao_isb'],
        },
        'ISC': {
            'FOB': fob_isc,
            't_preparacao': tempos_execucao['priorizacao_isc'],
            't_solver': tempos_execucao['solucao_isc'],
            't_total': tempos_execucao['priorizacao_isc'] + tempos_execucao['solucao_isc'],
        },
        'ISD': {
            'FOB': fob_isd,
            't_preparacao': tempos_execucao['priorizacao_isd'],
            't_solver': tempos_execucao['solucao_isd'],
            't_total': tempos_execucao['priorizacao_isd'] + tempos_execucao['solucao_isd'],
        },
        'ISG': {
            'FOB': fob_isg,
            't_preparacao': tempos_execucao['priorizacao_isg'],
            't_solver': tempos_execucao['solucao_isg'],
            't_total': tempos_execucao['priorizacao_isg'] + tempos_execucao['solucao_isg'],
        },
        'ILS': {
            'FOB': fob_ils,
            't_preparacao': tempos_execucao['priorizacao_ils'],
            't_solver': tempos_execucao['solucao_ils'],
            't_total': tempos_execucao['priorizacao_ils'] + tempos_execucao['solucao_ils'],
        },
        'ITR': {
            'FOB': fob_itr,
            't_preparacao': tempos_execucao['priorizacao_itr'],
            't_solver': tempos_execucao['solucao_itr'],
            't_total': tempos_execucao['priorizacao_itr'] + tempos_execucao['solucao_itr'],
        },
        'IFB': {
            'FOB': fob_fb,
            't_preparacao': tempos_execucao['priorizacao_fb'],
            't_solver': tempos_execucao['solucao_fb'],
            't_total': tempos_execucao['priorizacao_fb'] + tempos_execucao['solucao_fb'],
        },
    }

    return pd.DataFrame.from_dict(dados, orient='index')
