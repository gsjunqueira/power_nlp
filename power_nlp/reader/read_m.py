"""
Módulo `read_m`

Módulo para leitura de arquivos .m do MATLAB contendo dados de unidades geradoras (DGER)
e curva de carga horária (DLOAD), convertendo as informações em estruturas Python.

Este script é projetado para processar arquivos com a seguinte estrutura:

- DGER = [ ... ];  → Matriz com parâmetros dos geradores.
- DLOAD = [ ... ]; → Matriz com carga horária.

Retorna um dicionário contendo duas chaves: "DGER" e "DLOAD", cada uma com uma lista de dicionários.

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

import re

def ler_dger_m(conteudo_linhas):
    """
    Lê o bloco de dados DGER e converte para uma lista de dicionários.

    Parâmetros:
        conteudo_linhas (list[str]): Linhas contendo os dados do bloco DGER.

    Retorna:
        list[dict]: Lista de dicionários com colunas PGMIN, PGMAX, a, b, c, MTU, MTD,
        HOT_COST, COLD_COST, HTC.
    """
    colunas = [
        "id", "pgmin", "pgmax", "a", "b", "c", "mtu", "mtd", "hot", "cold", "htc"
    ]
    dados = []
    for linha in conteudo_linhas:
        linha = linha.strip()
        if not linha or linha.startswith('%') or 'DGER' in linha or '];' in linha:
            continue
        partes = re.split(r'\s+', linha.strip())
        if len(partes) == 11:
            valores = [float(p.lstrip('0') or '0') if p.replace('.', '', 1).isdigit() else float(p)
                           for p in partes]
            dados.append(dict(zip(colunas, valores)))
    return dados

def ler_dload_m(conteudo_linhas):
    """
    Lê o bloco de dados DLOAD e converte para uma lista de dicionários.

    Parâmetros:
        conteudo_linhas (list[str]): Linhas contendo os dados do bloco DLOAD.

    Retorna:
        list[dict]: Lista de dicionários com colunas hora, demanda_MW, demanda_percentual.
    """
    dload = []
    for linha in conteudo_linhas:
        partes = linha.strip().split()
        if len(partes) == 3:
            hora, mw, pct = map(int, partes)
            dload.append({
                'hora': hora,
                'carga': mw,
                'reserva': pct
            })
    return dload

def ler_m(filepath):
    """
    Lê um arquivo .m do MATLAB contendo os blocos DGER e DLOAD e os converte para
    um dicionário estruturado.

    Parâmetros:
        filepath (str): Caminho do arquivo .m a ser processado.

    Retorna:
        dict: Dicionário com duas chaves:
            - "DGER": lista de dicionários com dados dos geradores.
            - "DLOAD": lista de dicionários com dados da curva de carga horária.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    # Extrair bloco DGER
    inicio_dger = fim_dger = None
    for i, linha in enumerate(linhas):
        if 'DGER' in linha and '[' in linha:
            inicio_dger = i + 1
        if inicio_dger and '];' in linha and i > inicio_dger:
            fim_dger = i
            break
    conteudo_dger = linhas[inicio_dger:fim_dger]
    dger_dados = ler_dger_m(conteudo_dger)
    for ger in dger_dados:
        ger['id'] = f'GT{int(ger['id']):02}'

    # Extrair bloco DLOAD
    inicio_dload = fim_dload = None
    for i, linha in enumerate(linhas):
        if 'DLOAD' in linha and '[' in linha:
            inicio_dload = i + 1
        if inicio_dload and '];' in linha and i > inicio_dload:
            fim_dload = i
            break
    conteudo_dload = linhas[inicio_dload:fim_dload]
    dload_dados = ler_dload_m(conteudo_dload)

    return {"DGER": dger_dados, "DLOAD": dload_dados}
