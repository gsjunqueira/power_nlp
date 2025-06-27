"""
Módulo `avg_cost_opt_point`

Este módulo implementa o cálculo do indicador ISD (Índice de Sensibilidade por
Custo Médio no Ponto Ótimo) como heurística de priorização para o despacho
de unidades térmicas.

A lógica consiste em:
- Rodar um modelo de despacho com todas as unidades ligadas
- Calcular a geração ótima (Pgt) de cada gerador
- Avaliar o custo médio específico ISD com base em Pgt
- Priorizar os geradores com menor ISD por período
- Resolver o despacho com base nessa priorização
- Retornar os resultados, custos e tempos de execução

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from typing import List, Dict, Tuple
from time import time
import pandas as pd
from power_nlp.heuristicas import on_off, on_off_refinado, gerar_z_fixo, resultados_dataframe
from power_nlp.model_nlp import DespachoNLP


def is_d(a: float, b: float, c: float, pg: float) -> float:
    """
    Calcula o índice ISD (custo médio específico) para um gerador.

    O índice ISD é definido por:
        ISD = (a + b * P + c * P²) / P

    Onde:
        - a, b, c: coeficientes da função de custo
        - pg: valor de geração (em MW)

    Args:
        a (float): Termo constante da função de custo.
        b (float): Coeficiente linear da função de custo.
        c (float): Coeficiente quadrático da função de custo.
        pg (float): Geração onde o índice será avaliado.

    Returns:
        float: Valor do índice ISD.
    """
    return (a + b * pg + c * pg ** 2) / pg

def priorizar_isd(dger: List[dict], pg_otimo: Dict[str, List[float]]
                                  ) -> Dict[int, List[str]]:
    """
    Calcula a ordem de prioridade por ISD para cada período de tempo.

    Utiliza a geração ótima Pgt de cada gerador para calcular:
        ISD[g,t] = (a + b * Pgt + c * Pgt²) / Pgt

    Se Pgt = 0, utiliza pgmin como fallback.

    Args:
        dger (List[dict]): Lista de dicionários com dados dos geradores.
        pg_otimo (Dict[str, List[float]]): Geração ótima por gerador e período.

    Returns:
        Dict[int, List[str]]: Ordem de prioridade por período {t: [g1, g2, ...]}.
    """
    usinas = [g['id'] for g in dger]
    tempo = len(next(iter(pg_otimo.values())))
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}

    prioridade_por_tempo = {}

    for t in range(tempo):
        isd_t = {
            g: is_d(a[g], b[g], c[g], pg_otimo[g][t]) if pg_otimo[g][t] > 0 else
            is_d(a[g], b[g], c[g], pgmin[g]) for g in usinas
        }
        prioridade_por_tempo[t] = [g for g, _ in sorted(isd_t.items(), key=lambda item: item[1])]

    return prioridade_por_tempo

def gerar_status_completo(geradores: List[dict], cargas: List[dict]) -> Dict[Tuple[str, int], int]:
    """
    Gera um dicionário z_fixo assumindo todas as usinas ligadas em todos os períodos.

    Args:
        geradores (List[dict]): Lista de dicionários com 'id' das usinas.
        cargas (List[dict]): Lista de dicionários com os períodos (para saber T).

    Returns:
        Dict[Tuple[str, int], int]: Mapeamento {(usina, t): 1}.
    """
    return {
        (g['id'], t): 1
        for g in geradores
        for t in range(len(cargas))
    }

def gerar_pg_otimo(dger, dload) -> Dict[str, List[float]]:
    """
    Resolve o modelo com todas as usinas ligadas (e pgmin = 0) para obter a geração ótima.

    Esta função simula o despacho ideal sem restrição de ativação de unidades
    (z_fixo = 1 para todas), apenas para extrair a geração ótima Pgt de cada gerador.

    Args:
        dger (List[dict]): Lista com dados dos geradores.
        dload (List[dict]): Lista com dados de carga e reserva por período.

    Returns:
        Dict[str, List[float]]: Geração ótima por gerador e período {g: [Pg_0, ..., Pg_T]}.
    """
    # z_fixo com tudo ligado
    z_fixo = gerar_status_completo(dger, dload)

    # Preparar os dados
    usinas = [g['id'] for g in dger]
    periodos = list(range(len(dload)))
    pmin0 = {g['id']: 0 for g in dger}
    pmax = {g['id']: g['pgmax'] for g in dger}
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    demanda = {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}

    modelo = DespachoNLP(usinas, periodos, a, b, c, pmin0, pmax, demanda, reserva, z_fixo)
    modelo.solve()
    resultados, _ = modelo.get_resultados()

    # Extrair Pgot por usina e período
    pg_otimo = {g: [] for g in usinas}
    for t in periodos:
        for g in usinas:
            pg_otimo[g].append(resultados[t][g]["geracao"])

    return pg_otimo

def indicador_isd(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISD para priorização do despacho de geradores térmicos.

    Etapas:
    - Resolve o despacho com todas as unidades ligadas para obter Pgt
    - Calcula o índice ISD por gerador e período
    - Prioriza os geradores com menor ISD
    - Gera z_fixo com base nessa priorização
    - Resolve o modelo de despacho com z_fixo
    - Extrai resultados e tempos de execução

    Args:
        dger (List[Dict]): Dados dos geradores.
        dload (List[Dict]): Dados de carga e reserva por período.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados de geração por período.
            - dict: Custos por período.
            - float: Valor da função objetivo (FOB).
            - dict: Tempos de execução {'priorizacao', 'solucao'}.
    """
    # indicador isd
    inicio_isd = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    print('Calculando o PG_Ótimo')
    pg_otimo = gerar_pg_otimo(dger, dload)
    ordem_isd = priorizar_isd(dger, pg_otimo)
    isd = on_off(dger, ordem_isd, dload)
    z_isd = gerar_z_fixo(isd)

    # resolução para isd
    sol_isd = time()
    print('Calculando o índice ISD')
    m_isd = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isd)
    m_isd.solve()
    resul_isd, fob_isd = m_isd.get_resultados()
    custo_isd = m_isd.get_custos_tempo()
    df_isd = resultados_dataframe(resul_isd)
    fim = time()

    tempos = {
        "priorizacao": sol_isd-inicio_isd,
        "solucao": fim - sol_isd,
        "isd": ordem_isd
    }

    return df_isd, custo_isd, fob_isd, tempos

def indic_isd_ref(dger: List[Dict], dload: List[Dict]) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ISD para priorização do despacho de geradores térmicos.

    Etapas:
    - Resolve o despacho com todas as unidades ligadas para obter Pgt
    - Calcula o índice ISD por gerador e período
    - Prioriza os geradores com menor ISD
    - Gera z_fixo com base nessa priorização
    - Resolve o modelo de despacho com z_fixo
    - Extrai resultados e tempos de execução

    Args:
        dger (List[Dict]): Dados dos geradores.
        dload (List[Dict]): Dados de carga e reserva por período.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados de geração por período.
            - dict: Custos por período.
            - float: Valor da função objetivo (FOB).
            - dict: Tempos de execução {'priorizacao', 'solucao'}.
    """
    # indicador isd
    inicio_isd = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    print('Calculando o PG_Ótimo')
    pg_otimo = gerar_pg_otimo(dger, dload)
    ordem_isd = priorizar_isd(dger, pg_otimo)
    isd = on_off_refinado(dger, ordem_isd, dload)
    z_isd = gerar_z_fixo(isd)

    # resolução para isd
    sol_isd = time()
    print('Calculando o índice ISD')
    m_isd = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_isd)
    m_isd.solve()
    resul_isd, fob_isd = m_isd.get_resultados()
    custo_isd = m_isd.get_custos_tempo()
    df_isd = resultados_dataframe(resul_isd)
    fim = time()

    tempos = {
        "priorizacao": sol_isd-inicio_isd,
        "solucao": fim - sol_isd,
        "isd": ordem_isd
    }

    return df_isd, custo_isd, fob_isd, tempos
