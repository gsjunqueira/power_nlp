"""
Módulo `marg_cost_avg_power`

Este módulo implementa o cálculo do indicador ISB (Índice de Sensibilidade do 
Custo Linear) como heurística de priorização para o despacho de unidades térmicas.

A lógica consiste em:
- Calcular o índice ISB para cada gerador
- Priorizar os geradores com menor ISB
- Construir a matriz z_fixo com base na priorização
- Resolver o modelo DespachoNLP com os geradores selecionados
- Retornar os resultados, custos e tempos de execução

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from typing import List, Dict, Tuple
from time import time
import pandas as pd
from power_nlp.heuristicas import on_off, gerar_z_fixo, resultados_dataframe
from power_nlp.model_nlp import DespachoNLP


def is_b(b: float, c: float, pg: float) -> float:
    """
    Calcula o índice ISB de priorização para um gerador térmico.

    O índice é dado por:
        ISB = b + 2 * c * pg

    Onde:
        - b é o coeficiente linear do custo de geração
        - c é o coeficiente quadrático do custo
        - pg é o ponto médio de geração (ex: média entre pgmin e pgmax)

    Args:
        b (float): Coeficiente linear do custo.
        c (float): Coeficiente quadrático do custo.
        pg (float): Ponto de operação da usina.

    Returns:
        float: Valor do índice ISB.
    """
    return b + 2 * c * pg

def priorizar_isb(dger: List[dict]) -> List[str]:
    """
    Calcula o índice ISB para cada gerador e retorna a lista ordenada por menor ISB.

    Também insere o campo 'isb' diretamente nos dicionários da lista `dger`.

    Args:
        dger (List[dict]): Lista de dicionários com dados dos geradores, contendo
            ao menos 'id', 'b', 'c', 'pgmin' e 'pgmax'.

    Returns:
        List[str]: Lista de IDs dos geradores ordenados crescentemente pelo ISB.
    """
    for ger in dger:
        ger['isb'] = is_b(ger['b'], ger['c'], (ger['pgmin'] + ger['pgmax']) / 2)

    return [g["id"] for g in sorted(dger, key=lambda g: g['isb'])]

def indicador_isb(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISB para priorização do despacho de geradores térmicos.

    Executa os seguintes passos:
    - Calcula o índice ISB para os geradores
    - Gera a matriz de ativação z_fixo com base na priorização
    - Resolve o modelo de despacho não linear (DespachoNLP)
    - Extrai os resultados, custos por período, FOB e tempos de execução

    Args:
        dger (List[Dict]): Lista de dicionários com dados dos geradores térmicos.
        dload (List[Dict]): Lista de dicionários com dados de carga e reserva por período.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados da geração por período e unidade.
            - dict: Custos por período (ex: custo total, variável etc.).
            - float: Valor da função objetivo (FOB) da solução ISB.
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador isb
    inicio_isb = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_isb = priorizar_isb(dger)
    isb = on_off(dger, ordem_isb, dload)
    # print("ISB")
    # print(isb)
    z_isb = gerar_z_fixo(isb)

    # resolução para isb
    sol_isb = time()
    print('Calculando o índice ISB')
    m_isb = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isb)
    m_isb.solve()
    resul_isb, fob_isb = m_isb.get_resultados()
    custo_isb = m_isb.get_custos_tempo()
    df_isb = resultados_dataframe(resul_isb)
    fim = time()

    tempos = {
        "priorizacao": sol_isb-inicio_isb,
        "solucao": fim - sol_isb,
        "isb": ordem_isb 
    }

    return df_isb, custo_isb, fob_isb, tempos
