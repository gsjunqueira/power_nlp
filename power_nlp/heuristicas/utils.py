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

def on_off_refinado(
    geradores: List[Dict],
    prioridades: Union[List[str], Dict[int, List[str]]],
    cargas: List[Dict]
) -> pd.DataFrame:
    """
    Gera um DataFrame binário com o status ON/OFF de cada gerador por período,
    com base na ordem de prioridade e na necessidade de atender à carga + reserva.

    A lógica inclui um refinamento: se a soma dos pgmax já atende a demanda,
    o gerador atual é comparado ao próximo da lista para ver qual contribui com menor custo
    para a parcela restante de geração.

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
        soma_pgmin = 0
        soma_pgmax = 0

        if isinstance(prioridades, dict):
            prioridade_t = prioridades[t]
        else:
            prioridade_t = prioridades

        for i, gid in enumerate(prioridade_t):
            g = mapa_gerador[gid]
            soma_pgmin += g["pgmin"]
            soma_pgmax += g["pgmax"]
            ligados.append(gid)

            if soma_pgmin <= carga and soma_pgmax >= demanda:
                # Verificação de refinamento com próximo gerador
                ligados = refinar(ligados, prioridade_t, mapa_gerador, carga, demanda)
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

def refinar(
    ligados: List[str],
    prioridade_t: List[str],
    mapa_gerador: Dict[str, Dict],
    carga: float,
    demanda: float
) -> List[str]:
    """
    Refina a escolha do último gerador ligado trocando-o por outro mais econômico,
    desde que:
    - A troca mantenha a viabilidade (soma_pgmin ≤ carga e soma_pgmax ≥ demanda)
    - A geração estimada esteja dentro dos limites do gerador candidato
    - O custo estimado seja inferior ao atual

    Args:
        ligados (List[str]): Lista de UGs ativadas até o momento.
        prioridade_t (List[str]): Lista de prioridade para o período.
        mapa_gerador (dict): Dicionário {id: dados do gerador}.
        carga (float): Carga do período.
        demanda (float): Carga + reserva.

    Returns:
        List[str]: Lista de UGs ligadas (refinada).
    """
    if not ligados:
        return ligados  # nenhuma UG ligada

    g_atual_id = ligados[-1]
    g_atual = mapa_gerador[g_atual_id]

    # Soma das capacidades dos geradores anteriores
    anteriores = ligados[:-1]
    soma_pgmin = sum(mapa_gerador[g]["pgmin"] for g in anteriores)
    soma_pgmax = sum(mapa_gerador[g]["pgmax"] for g in anteriores)

    # Estimar quanto a última UG está contribuindo
    if soma_pgmax >= carga or soma_pgmax + g_atual["pgmin"] >= carga:
        geracao = g_atual["pgmin"]
    else:
        geracao = carga - soma_pgmax

    # Custo atual
    custo_atual = (
        g_atual["a"] + g_atual["b"] * geracao + g_atual["c"] * geracao**2
    )
    melhor_id = g_atual_id
    melhor_custo = custo_atual

    # Verificar candidatos restantes
    usados = set(ligados)
    for cand_id in prioridade_t:
        if cand_id in usados or cand_id == g_atual_id:
            continue
        g_cand = mapa_gerador[cand_id]

        nova_pgmin = soma_pgmin + g_cand["pgmin"]
        nova_pgmax = soma_pgmax + g_cand["pgmax"]

        if nova_pgmin <= carga and nova_pgmax >= demanda:
            if g_cand["pgmin"] <= geracao <= g_cand["pgmax"]:
                custo_cand = (
                    g_cand["a"] + g_cand["b"] * geracao + g_cand["c"] * geracao**2
                )
                if custo_cand < melhor_custo:
                    melhor_custo = custo_cand
                    melhor_id = cand_id

    # Substitui se for vantajoso
    if melhor_id != g_atual_id:
        ligados[-1] = melhor_id

    return ligados
