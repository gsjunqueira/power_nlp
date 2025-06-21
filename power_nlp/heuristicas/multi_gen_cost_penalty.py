"""
Módulo `mult_gen_cost_penalty`

Este módulo implementa o cálculo do indicador ISG (Índice de Sensibilidade Geral),
uma heurística de priorização proposta para o despacho de unidades térmicas.

A lógica consiste em:
- Estimar o custo total de atender à demanda + reserva por gerador
- Penalizar geradores que não conseguem suprir a carga mínima
- Priorizar os geradores com menor custo estimado em cada período
- Construir a matriz z_fixo com base nessa priorização
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


def is_g(ger: List[dict], load: List[dict]):
    """
    Calcula o índice ISG de priorização com base no custo estimado para cada gerador
    atender à demanda total (carga + reserva) em cada período.

    A função aplica penalidades:
    - Se o gerador atende totalmente à demanda: custo normal
    - Se o gerador atende parcialmente: custo + penalidade proporcional
    - Se não atende: penalidade máxima

    Returns um dicionário ordenado com os geradores priorizados por menor custo estimado.

    Args:
        ger (List[dict]): Lista de dicionários com dados dos geradores.
        load (List[dict]): Lista de dicionários com dados de carga e reserva por período.

    Returns:
        dict: {t: [g1, g2, ...]} ordem de prioridade por período.
    """
    resultados = {}
    for t, carga in enumerate(load):
        demanda = carga['carga'] + carga['reserva']
        resultados[t] = {}
        for g in ger:
            if g['pgmax'] >= demanda:
                fob = g['a'] + g['b'] * demanda + g['c'] * demanda ** 2
            elif g['pgmax'] < demanda and g['pgmin'] <= carga['carga']:
                fob = g['a'] + g['b'] * g['pgmax'] + g['c'] * g['pgmax'] ** 2 + (
                    demanda - g['pgmax']) * 1000
            else:
                fob = demanda * 1000
            resultados[t][g['id']] = fob

    def prioridade(custos_fob: dict) -> dict:
        """
        Ordena os geradores por período com base no menor custo FOB estimado.

        Args:
            custos_fob (dict): {t: {g: fob}} com os custos estimados por gerador e período.

        Returns:
            dict: {t: [g1, g2, ...]} lista de geradores ordenados por custo crescente.
        """
        prioridade = {}
        for t, custos in custos_fob.items():
            prioridade[t] = sorted(custos, key=custos.get)
        return prioridade

    return prioridade(resultados)

def indicador_isg(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISG para priorização do despacho de geradores térmicos.

    Etapas:
    - Calcula o índice ISG com base em custo estimado por gerador e período
    - Gera a matriz binária de ativação z_fixo conforme a ordem de prioridade
    - Resolve o modelo de despacho não linear (DespachoNLP)
    - Retorna os resultados do modelo, custos por período, FOB total e tempos

    Args:
        dger (List[Dict]): Dados dos geradores térmicos (id, a, b, c, pgmin, pgmax).
        dload (List[Dict]): Dados de carga e reserva por período.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados da geração por unidade e período.
            - dict: Custos por período (ex: custo total, variável etc.).
            - float: Valor da função objetivo (FOB).
            - dict: Tempos de execução {'priorizacao', 'solucao'}.
    """
    # indicador giovani
    inicio_isg = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_isg = is_g(dger, dload)
    isg = on_off(dger, ordem_isg, dload)
    z_isg = gerar_z_fixo(isg)

    # resolução para isg
    sol_isg = time()
    print('Calculando o índice ISG')
    m_isg = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isg)
    m_isg.solve()
    resul_isg, fob_isg = m_isg.get_resultados()
    custo_isg = m_isg.get_custos_tempo()
    df_isg = resultados_dataframe(resul_isg)
    fim = time()

    tempos = {
        "priorizacao": sol_isg-inicio_isg,
        "solucao": fim - sol_isg,
        "isg": ordem_isg
    }

    return df_isg, custo_isg, fob_isg, tempos
