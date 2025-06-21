"""
Módulo `relevance_dispatch`

Este módulo implementa o cálculo do indicador ITR (Índice por Tabela de Relevância),
uma heurística baseada em priorizações combinadas de diferentes estratégias
(ISA, ISB, ISC, ISD, ISG e ILS) via sorteios ponderados.

A lógica consiste em:
- Realizar múltiplos sorteios entre heurísticas disponíveis por período
- Construir uma tabela de relevância com pesos acumulados por posição
- Gerar uma ordem final de prioridade por período
- Construir o vetor z_fixo com base nessa ordem
- Resolver o modelo DespachoNLP
- Retornar os resultados, custos e tempos de execução

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"


import random
from collections import defaultdict
from typing import List, Dict, Tuple
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from power_nlp.heuristicas import on_off, gerar_z_fixo, resultados_dataframe
from power_nlp.model_nlp import DespachoNLP



def expandir_lista_global(lista_fixa: List[str], periodos: List[int]) -> Dict[int, List[str]]:
    """
    Expande uma lista de prioridade global para todos os períodos, replicando seu conteúdo.

    Args:
        lista_fixa (List[str]): Lista fixa de IDs (ex: ISA, ISB ou ISC).
        periodos (List[int]): Lista de períodos (ex: [0, 1, 2, ...]).

    Returns:
        Dict[int, List[str]]: Dicionário {t: ordem}, com a mesma ordem replicada para cada t.
    """
    return {t: lista_fixa[:] for t in periodos}

def ordem_randomica(
    periodos: List[int],
    ordem_isa: List[str],
    ordem_isb: List[str],
    ordem_isc: List[str],
    ordem_isd: Dict[int, List[str]],
    ordem_isg: Dict[int, List[str]],
    ordem_ils: Dict[int, List[str]]
) -> Dict[int, List[str]]:
    """
    Gera uma ordenação aleatória por período, sorteando uma entre as heurísticas fornecidas.

    Args:
        periodos (List[int]): Lista de períodos de tempo.
        ordem_isa (List[str]): Ordem global da heurística ISA.
        ordem_isb (List[str]): Ordem global da heurística ISB.
        ordem_isc (List[str]): Ordem global da heurística ISC.
        ordem_isd (Dict[int, List[str]]): Ordem por período da heurística ISD.
        ordem_isg (Dict[int, List[str]]): Ordem por período da heurística ISG.
        ordem_ils (Dict[int, List[str]]): Ordem por período da heurística ILS.

    Returns:
        Dict[int, List[str]]: Mapeamento {t: lista_ordenada} com a ordem sorteada por período.
    """
    # Padronizar todos para Dict[int, List[str]]
    isa_dict = expandir_lista_global(ordem_isa, periodos)
    isb_dict = expandir_lista_global(ordem_isb, periodos)
    isc_dict = expandir_lista_global(ordem_isc, periodos)

    ordem_randomizada = {}

    for t in periodos:
        sorteio = random.choice(["isa", "isb", "isc", "isd", "isg", "ils"])
        if sorteio == "isa":
            ordem_randomizada[t] = isa_dict[t]
        elif sorteio == "isb":
            ordem_randomizada[t] = isb_dict[t]
        elif sorteio == "isc":
            ordem_randomizada[t] = isc_dict[t]
        elif sorteio == "isd":
            ordem_randomizada[t] = ordem_isd[t]
        elif sorteio == "isg":
            ordem_randomizada[t] = ordem_isg[t]
        else:  # ils
            ordem_randomizada[t] = ordem_ils[t]

    return ordem_randomizada

def tabela_relevancia(
    dger: List[dict],
    periodos: List[int],
    ordem_isa: List[str],
    ordem_isb: List[str],
    ordem_isc: List[str],
    ordem_isd: Dict[int, List[str]],
    ordem_isg: Dict[int, List[str]],
    ordem_ils: Dict[int, List[str]],
    n_iter: int = 100
) -> Dict[int, List[str]]:
    """
    Gera uma tabela de relevância por período com base em sorteios aleatórios entre heurísticas.

    Para cada sorteio, atribui pesos decrescentes às posições de prioridade, acumulando
    relevâncias por gerador. Ao final, ordena os geradores por relevância acumulada.

    Args:
        dger (List[dict]): Lista de geradores com campo 'id'.
        periodos (List[int]): Lista de períodos.
        ordem_isa, ordem_isb, ordem_isc: listas fixas.
        ordem_isd, ordem_isg, ordem_ils: dicionários por hora.
        n_iter (int): Número de sorteios por hora.

    Returns:
        Tuple:
            - Dict[int, List[str]]: Ordem final por hora com base na relevância média.
            - Dict[int, Dict[str, float]]: Valores brutos de relevância por gerador e período.
    """
    relevancia = {t: defaultdict(float) for t in periodos}
    todos_ids = [g["id"] for g in dger]

    for _ in range(n_iter):
        ordem_rand = ordem_randomica(
            periodos, ordem_isa, ordem_isb, ordem_isc, ordem_isd, ordem_isg, ordem_ils
        )
        for t in periodos:
            for pos, gid in enumerate(ordem_rand[t]):
                relevancia[t][gid] += 1 / (1 + pos)  # maior peso para primeiros
    # Converte em ordem final por hora
    ordem_final = {}
    for t in periodos:
        ordem_final[t] = sorted(todos_ids, key=lambda g: -relevancia[t][g])

    return ordem_final, relevancia

def heatmap(contagem: Dict[int, Dict[str, int]]):
    """
    Plota um heatmap da frequência com que cada gerador apareceu na 1ª posição por período.

    Args:
        contagem (Dict[int, Dict[str, int]]): Dicionário no formato {hora: {gerador: contagem}}.
    """
    # Converte o dicionário para DataFrame
    df_contagem = pd.DataFrame(contagem).fillna(0).astype(int).T

    # Ajuste opcional: ordenar colunas (geradores) numericamente
    df_contagem = df_contagem.reindex(sorted(df_contagem.columns), axis=1)

    # Plotar heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        df_contagem,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray"
    )

    plt.title("Heatmap de Frequência em 1º Lugar por Gerador e Hora")
    plt.xlabel("Geradores")
    plt.ylabel("Hora")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def indicador_itr(dger: List[Dict], dload: List[Dict], ordem: Dict
                  ) -> Tuple[pd.DataFrame, dict, float, dict]:
    """
    Aplica a heurística ITR (Índice por Tabela de Relevância) para priorização do despacho.

    Etapas:
    - Executa sorteios aleatórios entre as heurísticas para cada período
    - Constrói uma tabela de relevância com pesos acumulados por posição
    - Gera uma ordem final por período
    - Constrói a matriz z_fixo com base nessa priorização
    - Resolve o modelo de despacho não linear (DespachoNLP)
    - Extrai resultados, custos, função objetivo e tempos

    Args:
        dger (List[Dict]): Lista com dados dos geradores térmicos.
        dload (List[Dict]): Lista com dados de carga e reserva por período.
        ordem (Dict): Dicionário com as chaves 'ordem_isa', 'ordem_isb', 'ordem_isc',
                      'ordem_isd', 'ordem_isg' e 'ordem_ls'.

    Returns:
        Tuple:
            - pd.DataFrame: Resultados da geração por período e unidade.
            - dict: Custos por período (ex: total, variável etc.).
            - float: Valor da função objetivo (FOB).
            - dict: Tempos de execução ('priorizacao', 'solucao').
    """
    # indicador itr
    inicio_itr = time()
    periodos = list(range(len(dload)))
    ute = [g['id'] for g in dger]
    a = {g['id']: g['a'] for g in dger}
    b = {g['id']: g['b'] for g in dger}
    c = {g['id']: g['c'] for g in dger}
    pgmin = {g['id']: g['pgmin'] for g in dger}
    pgmax = {g['id']: g['pgmax'] for g in dger}
    demanda =  {t: dload[t]['carga'] for t in periodos}
    reserva = {t: dload[t]['reserva'] for t in periodos}
    ordem_tr, contagem_tr = tabela_relevancia(dger, range(len(dload)), ordem['ordem_isa'],
                                ordem['ordem_isb'], ordem['ordem_isc'], ordem['ordem_isd'],
                                ordem['ordem_isg'], ordem['ordem_ils'], n_iter=1000 )
    itr = on_off(dger, ordem_tr, dload)
    z_itr = gerar_z_fixo(itr)

    # resolução para a tabela de relevância
    sol_itr = time()
    print('Calculando o índice ITR')
    m_itr = DespachoNLP(ute, periodos, a, b, c, pgmin, pgmax, demanda, reserva, z_itr)
    m_itr.solve()
    resul_itr, fob_itr = m_itr.get_resultados()
    custo_itr = m_itr.get_custos_tempo()
    df_itr = resultados_dataframe(resul_itr)
    fim = time()

    tempos = {
        "priorizacao": sol_itr-inicio_itr,
        "solucao": fim - sol_itr,
        "itr": ordem_tr
    }

    heatmap(contagem_tr)

    return df_itr, custo_itr, fob_itr, tempos
