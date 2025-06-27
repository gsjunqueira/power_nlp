"""
Pacote `heuristicas`

Módulo de integração de heurísticas e utilitários para despacho contínuo.

Este pacote reúne funções e indicadores utilizados no modelo de despacho não linear contínuo
implementado com Pyomo. Inclui:

- Funções de apoio: geração de status ON/OFF, vetores z_fixo e transformação de resultados.
- Indicadores heurísticos: ISA, ISB, ISC, ISD, ISG, ILS e ITR.
- Estratégia de força bruta como referência para comparação de heurísticas.
- Utilitários de visualização como o heatmap de relevância.

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from .utils import on_off, on_off_refinado, gerar_z_fixo, resultados_dataframe
from .avg_full_load_cost import indicador_isa, indic_isa_ref
from .marg_cost_avg_power import indicador_isb, indic_isb_ref
from .marg_cost_full_load import indicador_isc, indic_isc_ref
from .avg_cost_opt_point import indicador_isd, indic_isd_ref
from .multi_gen_cost_penalty import indicador_isg, indic_isg_ref
from .heuristic_lagrange import indicador_ils, indic_ils_ref
from .relevance_dispatch import indicador_itr
from .forca_bruta import forca_bruta

__all__ = [
    "on_off", "on_off_refinado", "gerar_z_fixo", "resultados_dataframe",
    "indicador_isa",
    "indic_isa_ref",
    "indicador_isb",
    "indic_isb_ref",
    "indicador_isc",
    "indic_isc_ref",
    "indicador_isd",
    "indic_isd_ref",
    "indicador_isg",
    "indic_isg_ref",
    "indicador_ils",
    "indic_ils_ref",
    "indicador_itr",
    "forca_bruta"
]
