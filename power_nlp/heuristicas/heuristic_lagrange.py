"""
Módulo `heuristic_lagrange`

Este módulo implementa o cálculo do indicador ILS (Índice de Lagrange por Sensibilidade),
utilizado como heurística de priorização para o despacho de unidades térmicas.

A lógica consiste em:
- Ativar a modelagem ODF com todas as variáveis z[g, t] fixadas em 0
- Extrair os multiplicadores de Lagrange associados às variáveis x[g, t]
- Priorizar os geradores com maior sensibilidade (maior valor de lambda)
- Construir a matriz z_fixo com base nessa priorização
- Resolver o modelo DespachoNLP com os geradores selecionados
- Retornar os resultados, custos e tempos de execução

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from typing import List, Dict, Tuple
from collections import defaultdict
from time import time
import pandas as pd
from power_nlp.heuristicas import on_off, on_off_refinado, gerar_z_fixo, resultados_dataframe
from power_nlp.model_nlp import DespachoNLP

def lagrangianos(geradores: dict, dload: list) -> dict:
    """
    Executa o modelo DespachoNLP com ODF ativado para cada período, com todas as variáveis
    binárias z[g, t] fixadas em 0, permitindo que as variáveis contínuas x[g, t] representem a ativação.

    A função extrai os multiplicadores de Lagrange associados às restrições de x[g, t]
    (função sigmoidal ODF) e utiliza esses valores como critério de priorização.

    Args:
        geradores (dict): Lista de dicionários com os dados dos geradores (inclusive dummy),
            contendo chaves como 'id', 'a', 'b', 'c', 'pgmin' e 'pgmax'.
        dload (list): Lista de dicionários com demanda e reserva por período ('carga', 'reserva').

    Returns:
        dict: Mapeamento {t: [g1, g2, ...]} com os IDs dos geradores ordenados por sensibilidade
              decrescente (maior valor do multiplicador de Lagrange).
    """
    periodos = list(range(len(dload)))

    ute = [g['id'] for g in geradores]
    a = {g['id']: g['a'] for g in geradores}
    b = {g['id']: g['b'] for g in geradores}
    c = {g['id']: g['c'] for g in geradores}
    pgmin = {g['id']: g['pgmin'] for g in geradores}
    pgmax = {g['id']: g['pgmax'] for g in geradores}
    demanda = {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}

    resultados = {}

    for t in periodos:
        z_fixo = {(g, t): 0 for g in ute}  # desativa todas as UGs (ativa x)

        modelo = DespachoNLP(
            usinas=ute,
            periodos=[t],
            a=a,
            b=b,
            c=c,
            pmin=pgmin,
            pmax=pgmax,
            demanda=demanda,
            reserva=reserva,
            z_fixo=z_fixo
        )

        modelo.construir_modelo()
        modelo.usar_odf(True)
        modelo.solve(tee=False)

        # Multiplicadores associados à variável x[g, t]
        lambdas = modelo.get_lagrangianos()
        # modelo.diagnostico()
        resultados.update({(g, t): lambdas[(g, t)] for g in ute})

    prioridade_por_tempo = defaultdict(list)

    for (g, t), valor in resultados.items():
        prioridade_por_tempo[t].append((g, valor))

    prioridade_ordenada = {t: [g for g, _ in sorted(lista, key=lambda x: x[1], reverse=True)]
        for t, lista in prioridade_por_tempo.items()
    }
    return prioridade_ordenada

def indicador_ils(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ILS (Índice de Lagrange por Sensibilidade) para priorização do
    despacho de geradores térmicos com base na sensibilidade associada às variáveis x[g, t].

    Etapas:
    - Executa o modelo com ODF ativado e z_fixo = 0 para todos os geradores
    - Extrai os multiplicadores de Lagrange associados às restrições x[g, t]
    - Prioriza os geradores com maior sensibilidade (lambda mais alto)
    - Gera o vetor z_fixo com base nessa ordem de prioridade
    - Resolve o modelo de despacho com essa configuração
    - Retorna os resultados, custos, valor da função objetivo (FOB) e tempos

    Args:
        dger (List[Dict]): Lista com dados dos geradores térmicos.
        dload (List[Dict]): Lista com dados de carga e reserva por período.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados da geração por período e unidade.
            - dict: Custos por período (ex: custo total, variável etc.).
            - float: Valor da função objetivo (FOB).
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador sensibilidade de lagrange
    inicio_ils = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_ls = lagrangianos(dger, dload)
    ils = on_off(dger, ordem_ls, dload)
    z_ils = gerar_z_fixo(ils)

    # resolução para ils
    sol_ils = time()
    print('Calculando o índice ILS')
    m_ils = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_ils)
    m_ils.solve()
    resul_ils, fob_ils = m_ils.get_resultados()
    custo_ils = m_ils.get_custos_tempo()
    df_ils = resultados_dataframe(resul_ils)
    fim = time()

    tempos = {
        "priorizacao": sol_ils-inicio_ils,
        "solucao": fim - sol_ils,
        "ils": ordem_ls 
    }

    return df_ils, custo_ils, fob_ils, tempos

def indic_ils_ref(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ILS (Índice de Lagrange por Sensibilidade) para priorização do
    despacho de geradores térmicos com base na sensibilidade associada às variáveis x[g, t].

    Etapas:
    - Executa o modelo com ODF ativado e z_fixo = 0 para todos os geradores
    - Extrai os multiplicadores de Lagrange associados às restrições x[g, t]
    - Prioriza os geradores com maior sensibilidade (lambda mais alto)
    - Gera o vetor z_fixo com base nessa ordem de prioridade
    - Resolve o modelo de despacho com essa configuração
    - Retorna os resultados, custos, valor da função objetivo (FOB) e tempos

    Args:
        dger (List[Dict]): Lista com dados dos geradores térmicos.
        dload (List[Dict]): Lista com dados de carga e reserva por período.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados da geração por período e unidade.
            - dict: Custos por período (ex: custo total, variável etc.).
            - float: Valor da função objetivo (FOB).
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador sensibilidade de lagrange
    inicio_ils = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_ls = lagrangianos(dger, dload)
    ils = on_off_refinado(dger, ordem_ls, dload)
    z_ils = gerar_z_fixo(ils)

    # resolução para ils
    sol_ils = time()
    print('Calculando o índice ILS')
    m_ils = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_ils)
    m_ils.solve()
    resul_ils, fob_ils = m_ils.get_resultados()
    custo_ils = m_ils.get_custos_tempo()
    df_ils = resultados_dataframe(resul_ils)
    fim = time()

    tempos = {
        "priorizacao": sol_ils-inicio_ils,
        "solucao": fim - sol_ils,
        "ils": ordem_ls 
    }

    return df_ils, custo_ils, fob_ils, tempos
