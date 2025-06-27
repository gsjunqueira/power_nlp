"""
Módulo `marg_cost_full_load`

Este módulo implementa o cálculo do indicador ISC (Índice de Sensibilidade com
Custo de partida) como heurística de priorização para o despacho de unidades térmicas.

A lógica consiste em:
- Calcular o índice ISC para cada gerador
- Priorizar os geradores com menor ISC
- Construir a matriz z_fixo com base na priorização
- Resolver o modelo DespachoNLP com os geradores selecionados
- Retornar os resultados, custos e tempos de execução

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from typing import List, Dict, Tuple
from time import time
import pandas as pd
from power_nlp.heuristicas import on_off, on_off_refinado, gerar_z_fixo, resultados_dataframe
from power_nlp.model_nlp import DespachoNLP


def is_c(b: float, c: float, pg: float, cp: float, tmp: float) -> float:
    """
    Calcula o índice ISC de priorização para um gerador térmico.

    O índice considera custo de partida e operação plena, sendo dado por:
        ISC = b + 2 * c * pg + cp / (tmp * pg)

    Onde:
        - b é o coeficiente linear do custo de geração
        - c é o coeficiente quadrático
        - pg é a potência máxima da unidade
        - cp é o custo médio de partida (entre quente e frio)
        - tmp é o tempo mínimo de operação (MTU)

    Args:
        b (float): Coeficiente linear do custo.
        c (float): Coeficiente quadrático do custo.
        pg (float): Potência de geração no ponto considerado (ex: pgmax).
        cp (float): Custo médio de partida.
        tmp (float): Tempo mínimo de operação (MTU).

    Returns:
        float: Valor do índice ISC.
    """
    return b + 2 * c * pg + cp / (tmp * pg)

def priorizar_isc(dger: List[dict]) -> List[str]:
    """
    Calcula o índice ISC para cada gerador e retorna a lista ordenada por menor valor.

    Também insere os campos 'cp' (custo médio de partida) e 'isc' diretamente nos
    dicionários da lista `dger`.

    Args:
        dger (List[dict]): Lista de dicionários com dados dos geradores, contendo
            ao menos 'id', 'b', 'c', 'pgmax', 'hot', 'cold', 'mtu'.

    Returns:
        List[str]: Lista de IDs dos geradores ordenados crescentemente pelo índice ISC.
    """
    for ger in dger:
        ger['cp'] = (ger['hot'] + ger['cold']) / 2
        ger['isc'] = is_c(ger['b'], ger['c'], ger['pgmax'], ger['cp'], ger['mtu'])

    return [g["id"] for g in sorted(dger, key=lambda g: g['isc'])]

def indicador_isc(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISC para priorização do despacho de geradores térmicos.

    Executa os seguintes passos:
    - Calcula o índice ISC para os geradores
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
            - float: Valor da função objetivo (FOB) da solução ISC.
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador isc
    inicio_isc = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_isc = priorizar_isc(dger)
    isc = on_off(dger, ordem_isc, dload)
    z_isc = gerar_z_fixo(isc)

    # resolução para isc
    sol_isc = time()
    print('Calculando o índice ISC')
    m_isc = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isc)
    m_isc.solve()
    resul_isc, fob_isc = m_isc.get_resultados()
    custo_isc = m_isc.get_custos_tempo()
    df_isc = resultados_dataframe(resul_isc)
    fim = time()

    tempos = {
        "priorizacao": sol_isc-inicio_isc,
        "solucao": fim - sol_isc,
        "isc": ordem_isc 
    }

    return df_isc, custo_isc, fob_isc, tempos

def indic_isc_ref(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISC para priorização do despacho de geradores térmicos.

    Executa os seguintes passos:
    - Calcula o índice ISC para os geradores
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
            - float: Valor da função objetivo (FOB) da solução ISC.
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador isc
    inicio_isc = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_isc = priorizar_isc(dger)
    isc = on_off_refinado(dger, ordem_isc, dload)
    z_isc = gerar_z_fixo(isc)

    # resolução para isc
    sol_isc = time()
    print('Calculando o índice ISC')
    m_isc = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isc)
    m_isc.solve()
    resul_isc, fob_isc = m_isc.get_resultados()
    custo_isc = m_isc.get_custos_tempo()
    df_isc = resultados_dataframe(resul_isc)
    fim = time()

    tempos = {
        "priorizacao": sol_isc-inicio_isc,
        "solucao": fim - sol_isc,
        "isc": ordem_isc 
    }

    return df_isc, custo_isc, fob_isc, tempos
