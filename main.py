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
                                   indicador_isg, indicador_ils, indicador_itr, forca_bruta,
                                   indic_isa_ref, indic_isb_ref, indic_isc_ref, indic_isd_ref,
                                   indic_isg_ref, indic_ils_ref)

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

    ger_isax, custo_isax, fob_isax, t_isax = indic_isa_ref(dger, dload)
    ger_isbx, custo_isbx, fob_isbx, t_isbx = indic_isb_ref(dger, dload)
    ger_iscx, custo_iscx, fob_iscx, t_iscx = indic_isc_ref(dger, dload)
    ger_isdx, custo_isdx, fob_isdx, t_isdx = indic_isd_ref(dger, dload)
    ger_isgx, custo_isgx, fob_isgx, t_isgx = indic_isg_ref(dger, dload)
    ger_ilsx, custo_ilsx, fob_ilsx, t_ilsx = indic_ils_ref(dger, dload)

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
    d_isax = pd.DataFrame.from_dict(custo_isax, orient="index", columns=["ISA"])
    d_isbx = pd.DataFrame.from_dict(custo_isbx, orient="index", columns=["ISB"])
    d_iscx = pd.DataFrame.from_dict(custo_iscx, orient="index", columns=["ISC"])
    d_isdx = pd.DataFrame.from_dict(custo_isdx, orient="index", columns=["ISD"])
    d_isgx = pd.DataFrame.from_dict(custo_isgx, orient="index", columns=["ISG"])
    d_ilsx = pd.DataFrame.from_dict(custo_ilsx, orient="index", columns=["ILS"])

    d_itr = pd.DataFrame.from_dict(custo_itr, orient="index", columns=["ITR"])
    d_ifb = pd.DataFrame({'IFB': df_forca_bruta['FOB']})

    df_custos = pd.concat([
        # d_isa, d_isax,
        # d_isb, d_isbx,
        # d_isc, d_iscx,
        # d_isd, d_isdx,
        d_isg, d_isgx,
        # d_ils, d_ilsx, d_itr,
        d_ifb], axis=1)
    df_custos.index.name = "Hora"

    fim = time()
    tempos_execucao = {
        "priorizacao_isa": t_isa['priorizacao'],
        "priorizacao_isax": t_isax['priorizacao'],
        "priorizacao_isb": t_isb['priorizacao'],
        "priorizacao_isbx": t_isbx['priorizacao'],
        "priorizacao_isc": t_isc['priorizacao'],
        "priorizacao_iscx": t_iscx['priorizacao'],
        "priorizacao_isd": t_isd['priorizacao'],
        "priorizacao_isdx": t_isdx['priorizacao'],
        "priorizacao_isg": t_isg['priorizacao'],
        "priorizacao_isgx": t_isgx['priorizacao'],
        "priorizacao_ils": t_ils['priorizacao'],
        "priorizacao_ilsx": t_ilsx['priorizacao'],
        "priorizacao_itr": t_itr['priorizacao'],
        "priorizacao_fb": t_ifb['priorizacao'],
        "solucao_isa": t_isa['solucao'],
        "solucao_isax": t_isax['solucao'],
        "solucao_isb": t_isb['solucao'],
        "solucao_isbx": t_isbx['solucao'],
        "solucao_isc": t_isc['solucao'],
        "solucao_iscx": t_iscx['solucao'],
        "solucao_isd": t_isd['solucao'],
        "solucao_isdx": t_isdx['solucao'],
        "solucao_isg": t_isg['solucao'],
        "solucao_isgx": t_isgx['solucao'],
        "solucao_ils": t_ils['solucao'],
        "solucao_ilsx": t_ilsx['solucao'],
        "solucao_itr": t_itr['solucao'],
        "solucao_fb": t_ifb['solucao'],
        "tempo_total": fim - inicio
    }

    for etapa, tempo in tempos_execucao.items():
        print(f"⌛️ - Tempo da etapa {etapa}: {tempo:.4f} s")

    resultado = desempenho(fob_isa, fob_isax, fob_isb, fob_isbx, fob_isc, fob_iscx, 
                           fob_isd, fob_isdx, fob_isg, fob_isgx, fob_ils, fob_ilsx,
                           fob_itr, fob_fb, tempos_execucao)

    print(resultado)

    # print(pg_otimo)

    print(df_custos)

    print(df_forca_bruta)

    # pprint(dados)

if __name__ == "__main__":
    main()
