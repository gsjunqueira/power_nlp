"""
Módulo `forca_bruta`

Este módulo fornece funções auxiliares para a geração e avaliação de todas
as combinações viáveis de unidades geradoras térmicas por período de tempo,
baseando-se em limites de carga e reserva. A estratégia segue abordagem de 
busca exaustiva (força bruta) para determinar as melhores combinações que 
minimizam a função objetivo de despacho.

As funções implementadas neste módulo incluem:
- Geração de combinações viáveis de usinas por período
- Construção de vetores binários z_fixo compatíveis com o modelo DespachoNLP
- Avaliação da função objetivo para cada combinação
- Extração do melhor resultado por período em formato tabular

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from itertools import combinations
from typing import List, Dict, Set, Tuple
from time import time
import pandas as pd
from power_nlp.model_nlp import DespachoNLP

def comb_viaveis(
    geradores: List[Dict], cargas: List[Dict]
) -> Dict[int, List[Set[str]]]:
    """
    Gera todas as combinações viáveis de geradores para cada período.

    Uma combinação é considerada viável em um período t se:
    - a soma dos valores pgmin dos geradores for menor ou igual à carga
    - a soma dos valores pgmax for maior ou igual à carga acrescida da reserva

    Args:
        geradores (List[Dict]): Lista de geradores com chaves 'id', 'pgmin' e 'pgmax'.
        cargas (List[Dict]): Lista de dicionários com chaves 'carga' e 'reserva' por período.

    Returns:
        Dict[int, List[Set[str]]]: Mapeamento de período t para lista de conjuntos viáveis
        de IDs de geradores que satisfazem as restrições de carga e reserva.
    """
    usinas = [g["id"] for g in geradores]
    mapa = {g["id"]: g for g in geradores}
    combinacoes_por_tempo = {}

    for t, carga_info in enumerate(cargas):
        carga = carga_info["carga"]
        reserva = carga_info["reserva"]
        candidatos = []

        for k in range(1, len(usinas) + 1):
            for subset in combinations(usinas, k):
                soma_min = sum(mapa[g]["pgmin"] for g in subset)
                soma_max = sum(mapa[g]["pgmax"] for g in subset)

                if soma_min <= carga and soma_max >= carga + reserva:
                    candidatos.append(set(subset))

        combinacoes_por_tempo[t] = candidatos
    return combinacoes_por_tempo


def z_bruto_completo(combinacoes: Dict[int, List[Set[str]]]
                     ) -> Dict[int, List[Dict[Tuple[str, int], int]]]:
    """
    Constrói todos os vetores z_fixo compatíveis com DespachoNLP, por período.

    Cada vetor z_fixo é um dicionário indicando se o gerador g está ligado no
    período t (valor 1) ou não (valor 0), conforme a combinação fornecida.

    Args:
        combinacoes (Dict[int, List[Set[str]]]): Combinações viáveis por período t.

    Returns:
        Dict[int, List[Dict[Tuple[str, int], int]]]: Mapeamento de t para lista de vetores
        z_fixo válidos, com chaves (g, t) e valores binários.
    """
    z_bruto = {}

    usinas = sorted({g for lista in combinacoes.values() for s in lista for g in s})
    periodos = sorted(combinacoes.keys())

    for t in periodos:
        lista_para_t = []
        for subset in combinacoes[t]:
            z_fixo = {
                (g, t): 1 if g in subset else 0
                for g in usinas
            }
            lista_para_t.append(z_fixo)
        z_bruto[t] = lista_para_t

    return z_bruto

def melhor_fob_h(resultados: list, z_bruto: dict, ute: list) -> pd.DataFrame:
    """
    Extrai a melhor combinação por período com base no menor valor da função objetivo (FOB).

    Retorna um DataFrame com uma linha por período, contendo:
    - coluna 'hora' (índice do período)
    - colunas com os nomes dos geradores (valores 0/1)
    - coluna 'FOB' com o valor da função objetivo associado à melhor combinação

    Args:
        resultados (list): Lista de tuplas (t, j, fob) com período, índice da combinação e FOB.
        z_bruto (dict): Mapeamento de t para lista de z_fixos (dicts binários).
        ute (list): Lista dos nomes dos geradores (IDs).

    Returns:
        pd.DataFrame: DataFrame com as melhores combinações por hora e respectivas FOBs.
    """
    melhores_por_t = {}

    # Encontrar menor FOB por período
    for t, j, fob in resultados:
        if t not in melhores_por_t or fob < melhores_por_t[t][1]:
            melhores_por_t[t] = (j, fob)

    linhas = []
    for t in sorted(melhores_por_t):
        j, fob = melhores_por_t[t]
        z_fixo = z_bruto[t][j]
        linha = {'hora': t}
        for g in ute:
            linha[g] = z_fixo.get((g, t), 0)
        linha['FOB'] = fob
        linhas.append(linha)

    df = pd.DataFrame(linhas).sort_values(by='hora').reset_index(drop=True)
    return df

def forca_bruta(geradores: List[Dict], cargas: List[Dict]) -> Tuple[pd.DataFrame, dict]:
    """
    Executa busca exaustiva para encontrar a melhor combinação de geradores por hora.

    Para cada período:
    - Gera todas as combinações viáveis de geradores
    - Constrói vetores z_fixo
    - Resolve o modelo de despacho para cada combinação
    - Seleciona a combinação com menor valor da função objetivo (FOB)

    Args:
        geradores (List[Dict]): Lista de dicionários com dados dos geradores,
            contendo as chaves 'id', 'a', 'b', 'c', 'pgmin' e 'pgmax'.
        cargas (List[Dict]): Lista de dicionários com dados por período,
            contendo as chaves 'carga' e 'reserva'.

    Returns:
        Tuple[pd.DataFrame, dict]: 
            - DataFrame com a melhor combinação por período, incluindo colunas
              binárias por gerador e a FOB.
            - Dicionário com tempos de execução medidos para:
                * 'priorizacao': tempo de preparação das combinações
                * 'solucao': tempo de resolução dos modelos
    """
    inicio_fb = time()
    periodos = list(range(len(cargas)))
    ute = [g['id'] for g in geradores]
    a = {g['id']: g['a'] for g in geradores}
    b = {g['id']: g['b'] for g in geradores}
    c = {g['id']: g['c'] for g in geradores}
    pgmin = {g['id']: g['pgmin'] for g in geradores}
    pgmax = {g['id']: g['pgmax'] for g in geradores}
    demanda =  {t: cargas[t]['carga'] for t in periodos}
    reserva = {t: cargas[t]['reserva'] for t in periodos}

    combinacoes = comb_viaveis(geradores, cargas)
    z_bruto = z_bruto_completo(combinacoes)
    pesquisa = []

    print('Calculando o índice força bruta')
    sol_fb = time()
    for t, z_fixo in enumerate(z_bruto):  # percorre os períodos disponíveis
        for j, z_fixo in enumerate(z_bruto[t]):  # percorre os z_fixos para o período t
            m_fb = DespachoNLP(ute, [t], a, b, c, pgmin, pgmax, demanda, reserva, z_fixo)
            m_fb.solve()
            fob = m_fb.get_resultados()[1]
            pesquisa.append((t, j, fob))
    df_fob = melhor_fob_h(pesquisa, z_bruto, ute)
    fim = time()

    tempos = {
        "priorizacao": sol_fb-inicio_fb,
        "solucao": fim - sol_fb 
    }

    return df_fob, tempos
