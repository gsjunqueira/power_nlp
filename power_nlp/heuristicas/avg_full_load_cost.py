"""
Módulo `avg_full_load_cost`

Este módulo implementa o cálculo do indicador ISA (Índice de Sensibilidade do 
Custo Médio a Plena Carga) como heurística de priorização para o despacho de 
unidades térmicas.

A lógica consiste em:
- Calcular o índice ISA para cada gerador (custo médio a plena carga)
- Priorizar os geradores com menor ISA
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


def is_a(a: float, b: float, c: float, pg: float) -> float:
    """
    Calcula o índice ISA de priorização para um gerador térmico.

    O índice ISA é definido como o custo médio por unidade de energia gerada
    quando o gerador opera a plena carga:
        ISA = (a + b * pg + c * pg²) / pg

    Onde:
        - a, b, c: coeficientes da função de custo de geração
        - pg: ponto de operação (geralmente pgmax)

    Args:
        a (float): Termo constante do custo.
        b (float): Coeficiente linear do custo.
        c (float): Coeficiente quadrático do custo.
        pg (float): Potência de operação (típica: pgmax).

    Returns:
        float: Valor do índice ISA.
    """
    return (a + b * pg + c * pg ** 2) / pg

def priorizar_isa(dger: List[dict]) -> List[str]:
    """
    Calcula o índice ISA para cada gerador e retorna a lista ordenada por menor valor.

    Também insere o campo 'isa' diretamente nos dicionários da lista `dger`.

    Args:
        dger (List[dict]): Lista de dicionários com dados dos geradores,
            contendo ao menos 'id', 'a', 'b', 'c' e 'pgmax'.

    Returns:
        List[str]: Lista de IDs dos geradores ordenados crescentemente pelo índice ISA.
    """
    for ger in dger:
        ger['isa'] = is_a(ger['a'], ger['b'], ger['c'], ger['pgmax'])

    return [g["id"] for g in sorted(dger, key=lambda g: g['isa'])]

def indicador_isa(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISA para priorização do despacho de geradores térmicos.

    Executa os seguintes passos:
    - Calcula o índice ISA para os geradores
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
            - float: Valor da função objetivo (FOB) da solução ISA.
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador isa
    inicio_isa = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_isa = priorizar_isa(dger)
    isa = on_off(dger, ordem_isa, dload)
    z_isa = gerar_z_fixo(isa)

    # resolução para isb
    sol_isa = time()
    print('Calculando o índice ISB')
    m_isa = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isa)
    m_isa.solve()
    resul_isa, fob_isa = m_isa.get_resultados()
    custo_isa = m_isa.get_custos_tempo()
    df_isa = resultados_dataframe(resul_isa)
    fim = time()

    tempos = {
        "priorizacao": sol_isa-inicio_isa,
        "solucao": fim - sol_isa,
        "isa": ordem_isa
    }

    return df_isa, custo_isa, fob_isa, tempos

def indic_isa_ref(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISA para priorização do despacho de geradores térmicos.

    Executa os seguintes passos:
    - Calcula o índice ISA para os geradores
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
            - float: Valor da função objetivo (FOB) da solução ISA.
            - dict: Tempos de execução com as chaves 'priorizacao' e 'solucao'.
    """
    # indicador isa
    inicio_isa = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_isa = priorizar_isa(dger)
    isa = on_off_refinado(dger, ordem_isa, dload)
    z_isa = gerar_z_fixo(isa)

    # resolução para isb
    sol_isa = time()
    print('Calculando o índice ISB')
    m_isa = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isa)
    m_isa.solve()
    resul_isa, fob_isa = m_isa.get_resultados()
    custo_isa = m_isa.get_custos_tempo()
    df_isa = resultados_dataframe(resul_isa)
    fim = time()

    tempos = {
        "priorizacao": sol_isa-inicio_isa,
        "solucao": fim - sol_isa,
        "isa": ordem_isa
    }

    return df_isa, custo_isa, fob_isa, tempos
