"""
Arquivo principal de execução do modelo de despacho contínuo.

Este script:
- Lê os dados de entrada do arquivo .m
- Executa diversas heurísticas de priorização (ISA, ISB, ISC, ISD, ISG, ILS, ITR)
- Calcula o caso de referência por força bruta (IFB)
- Coleta tempos de execução, custos por período e valores da função objetivo
- Avalia o desempenho comparado de cada heurística
- Exibe resultados consolidados em tabelas e terminal

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from time import time
# from pprint import pprint
import pandas as pd
from power_nlp.reader import ler_m
from power_nlp.model_nlp import desempenho
from power_nlp.heuristicas import (indicador_isa, indicador_isb, indicador_isc, indicador_isd,
                                   indicador_isg, indicador_ils, indicador_itr, forca_bruta)

def main():
    """Função principal que orquestra a resolução do problema """

    # caminho = "data/UC_4UTES.m"  # ajuste conforme necessário
    caminho = "data/UC_10GER.m"  # ajuste conforme necessário
    dados = ler_m(caminho)
    dger = dados['DGER']
    dload = dados[ 'DLOAD']

    # pré processamento
    inicio = time()
    ger_isa, custo_isa, fob_isa, t_isa = indicador_isa(dger, dload)
    ger_isb, custo_isb, fob_isb, t_isb = indicador_isb(dger, dload)
    ger_isc, custo_isc, fob_isc, t_isc = indicador_isc(dger, dload)
    ger_isd, custo_isd, fob_isd, t_isd = indicador_isd(dger, dload)
    ger_isg, custo_isg, fob_isg, t_isg = indicador_isg(dger, dload)
    ger_ils, custo_ils, fob_ils, t_ils = indicador_ils(dger, dload)
    ordem = {'ordem_isa': t_isa['isa'], 'ordem_isb': t_isb['isb'], 'ordem_isc': t_isc['isc'],
             'ordem_isd': t_isd['isd'], 'ordem_isg': t_isg['isg'], 'ordem_ils': t_ils['ils']}
    ger_itr, custo_itr, fob_itr, t_itr = indicador_itr(dger, dload, ordem)
    df_forca_bruta, t_ifb = forca_bruta(dger, dload)
    fob_fb = df_forca_bruta['FOB'].sum()

    d_isa = pd.DataFrame.from_dict(custo_isa, orient="index", columns=["ISA"])
    d_isb = pd.DataFrame.from_dict(custo_isb, orient="index", columns=["ISB"])
    d_isc = pd.DataFrame.from_dict(custo_isc, orient="index", columns=["ISC"])
    d_isd = pd.DataFrame.from_dict(custo_isd, orient="index", columns=["ISD"])
    d_isg = pd.DataFrame.from_dict(custo_isg, orient="index", columns=["ISG"])
    d_ils = pd.DataFrame.from_dict(custo_ils, orient="index", columns=["ILS"])
    d_itr = pd.DataFrame.from_dict(custo_itr, orient="index", columns=["ITR"])
    d_ifb = pd.DataFrame({'IFB': df_forca_bruta['FOB']})

    df_custos = pd.concat([d_isa, d_isb, d_isc, d_isd, d_isg, d_ils, d_itr, d_ifb], axis=1)
    df_custos.index.name = "Hora"

    fim = time()
    tempos_execucao = {
        "priorizacao_isa": t_isa['priorizacao'],
        "priorizacao_isb": t_isb['priorizacao'],
        "priorizacao_isc": t_isc['priorizacao'],
        "priorizacao_isd": t_isd['priorizacao'],
        "priorizacao_isg": t_isg['priorizacao'],
        "priorizacao_ils": t_ils['priorizacao'],
        "priorizacao_itr": t_itr['priorizacao'],
        "priorizacao_fb": t_ifb['priorizacao'],
        "solucao_isa": t_isa['solucao'],
        "solucao_isb": t_isb['solucao'],
        "solucao_isc": t_isc['solucao'],
        "solucao_isd": t_isd['solucao'],
        "solucao_isg": t_isg['solucao'],
        "solucao_ils": t_ils['solucao'],
        "solucao_itr": t_itr['solucao'],
        "solucao_fb": t_ifb['solucao'],
        "tempo_total": fim - inicio
    }

    for etapa, tempo in tempos_execucao.items():
        print(f"⌛️ - Tempo da etapa {etapa}: {tempo:.4f} s")

    resultado = desempenho(fob_isa, fob_isb, fob_isc, fob_isd, fob_isg, fob_ils,
                           fob_itr, fob_fb, tempos_execucao)

    print(resultado)

    # print(pg_otimo)

    print(df_custos)

    print(df_forca_bruta)

    # pprint(dados)

if __name__ == "__main__":
    main()
