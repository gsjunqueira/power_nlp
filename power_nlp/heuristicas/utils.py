"""
Módulo `utils`

Módulo de utilitários auxiliares para construção e interpretação do modelo de despacho.

Inclui funções para:
- Geração de status ON/OFF com base em prioridade e exigência de carga
- Conversão do status para dicionário z_fixo
- Conversão dos resultados do modelo para DataFrame de geração por usina e período

Autor: Giovani Santiago Junqueira
"""

from typing import List, Dict, Tuple, Union
import pandas as pd

def on_off(
    geradores: List[Dict],
    prioridades: Union[List[str], Dict[int, List[str]]],
    cargas: List[Dict]
) -> pd.DataFrame:
    """
    Gera um DataFrame binário com o status ON/OFF de cada gerador por período,
    com base na ordem de prioridade e na necessidade de atender à carga + reserva.

    A seleção de unidades é feita iterativamente até que:
    - a soma dos pgmin seja ≥ carga
    - a soma dos pgmax seja ≥ carga + reserva

    Args:
        geradores (List[Dict]): Lista de dicionários com dados dos geradores.
        prioridades (List[str] | Dict[int, List[str]]): Lista fixa ou dicionário
            com ordenações específicas por período.
        cargas (List[Dict]): Lista de dicionários com 'carga' e 'reserva' por hora.

    Returns:
        pd.DataFrame: DataFrame com colunas ['hora', <ids>], com valores 0 ou 1.
    """
    mapa_gerador = {g["id"]: g for g in geradores}
    resultados = []

    for t, c in enumerate(cargas):
        carga = c["carga"]
        demanda = carga + c["reserva"]
        ligados = []
        soma_gmax = 0
        soma_gmin = 0

        # Obter prioridade correta para o período t
        if isinstance(prioridades, dict):
            prioridade_t = prioridades[t]
        else:
            prioridade_t = prioridades

        for gid in prioridade_t:
            pgmin = mapa_gerador[gid]['pgmin']
            pgmax = mapa_gerador[gid]['pgmax']

            soma_gmax += pgmax
            soma_gmin += pgmin
            ligados.append(gid)

            if  soma_gmin <= carga and soma_gmax >= demanda:
                break

        resultados.append({
            "hora": c["hora"],
            "ligados": ligados,
        })

    # Conversão para DataFrame binário
    tempos = []
    for entrada in resultados:
        tempo = {'hora': entrada['hora']}
        for g in geradores:
            tempo[g['id']] = 1 if g['id'] in entrada['ligados'] else 0
        tempos.append(tempo)

    return pd.DataFrame(tempos)

def gerar_z_fixo(df_status: pd.DataFrame) -> Dict[Tuple[str, int], int]:
    """
    Converte um DataFrame binário de status (ON/OFF) por hora para um dicionário z_fixo.

    O formato retornado é adequado para uso direto no modelo de despacho com variáveis z[g, t].

    Args:
        df_status (pd.DataFrame): DataFrame com colunas 'hora' e uma coluna por gerador.

    Returns:
        Dict[Tuple[str, int], int]: Dicionário no formato {(usina, t): 0 ou 1}.
    """
    usinas = [col for col in df_status.columns if col != "hora"]
    periodos = list(df_status.index)

    return {
        (g, t): int(df_status.loc[t, g])
        for t in periodos
        for g in usinas
    }

def resultados_dataframe(resultados: dict) -> pd.DataFrame:
    """
    Converte os resultados do modelo de despacho para um DataFrame com os valores de geração.

    A estrutura de entrada deve ter:
        resultados[t][g] = {'status': 0 ou 1, 'geracao': valor em MW}

    Args:
        resultados (dict): Dicionário {t: {g: {'status': int, 'geracao': float}}}

    Returns:
        pd.DataFrame: DataFrame com índice temporal e colunas por gerador (valores em MW).
    """
    dados = {
        t: {g: info['geracao'] for g, info in usinas.items()}
        for t, usinas in resultados.items()
    }
    return pd.DataFrame.from_dict(dados, orient="index").sort_index()
