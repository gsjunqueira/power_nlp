"""
Classe DespachoNLP: resolve problemas de despacho contínuo não linear
com funções de custo exponencial e quadrática, respeitando limites de
geração e status fixo das unidades geradoras por período.

Autor: Giovani Santiago Junqueira
"""

__author__ = "Giovani Santiago Junqueira"

from typing import Dict
from pyomo.environ import (ConcreteModel, Set, Var, Objective, Suffix, Param, exp,
                           Constraint, minimize, SolverFactory, value, NonNegativeReals)

class DespachoNLP:
    """
    Modelo de despacho contínuo não linear com múltiplas usinas e períodos.

    A função objetivo é a soma dos custos operacionais não lineares por
    usina e por período. O modelo respeita limites de geração (mínimo e máximo)
    e impõe que a geração total atenda a demanda em cada período.

    O status de operação das usinas (ligada ou desligada) é fornecido externamente
    via o parâmetro z_fixo, e utilizado para ativar ou desativar a geração.
    """
    def __init__(self, usinas, periodos, a, b, c, pmin, pmax, demanda, reserva, z_fixo):
        """
        Inicializa a classe e armazena os parâmetros do modelo.

        Args:
            usinas (list[str]): Lista de identificadores das usinas térmicas.
            periodos (list[int]): Lista dos períodos de tempo.
            a (dict): Coeficientes exponenciais de custo por usina (a[g]).
            b (dict): Coeficientes quadráticos de custo por usina (b[g]).
            pmin (dict): Limites mínimos de geração por usina.
            pmax (dict): Limites máximos de geração por usina.
            demanda (dict): Demanda total do sistema por período.
            z_fixo (dict): Status fixo (0 ou 1) de operação da usina no período,
            indexado por (usina, período).
        """
        self.model = None
        self.usinas = usinas
        self.periodos = periodos
        self.a = a
        self.b = b
        self.c = c
        self.pmin = pmin
        self.pmax = pmax
        self.demanda = demanda
        self.reserva = reserva
        self.z_fixo = z_fixo
        self.alpha = 1000.0
        self._usar_odf = False
        self.pcmin = {t: 0.0 for t in periodos}
        self.pcmax = {t: 2e3 for t in periodos}
        self.rho = 999999.0

    def construir_modelo(self):
        """
        Constrói o modelo Pyomo com variáveis de geração, função objetivo
        e restrições de balanço de carga por período.

        O modelo é armazenado no atributo `self.model`.
        """
        m = ConcreteModel()
        m.dual_x = Suffix(direction=Suffix.IMPORT)
        m.dual = Suffix(direction=Suffix.IMPORT)
        m.G = Set(initialize=self.usinas)
        m.T = Set(initialize=self.periodos)
        m.P = Var(m.G, m.T, domain=NonNegativeReals)
        m.x = Var(m.G, m.T, bounds=(0, 1), initialize=0)
        m.PC = Var(m.T, domain=NonNegativeReals)
        m.z = Param(m.G, m.T, initialize=self.z_fixo, mutable=True)

        m.pcmin = Param(m.T, initialize=self.pcmin)
        m.pcmax = Param(m.T, initialize=self.pcmax)

        m.restr_sup = Constraint(m.G, m.T, rule=self._restr_sup)
        m.restr_inf = Constraint(m.G, m.T, rule=self._restr_inf)
        m.restr_demanda = Constraint(m.T, rule=self._restr_demanda)
        m.restr_reserva = Constraint(m.T, rule=self._restr_reserva)

        m.restr_sup_odf = Constraint(m.G, m.T, rule=self._restr_sup_odf)
        m.restr_inf_odf = Constraint(m.G, m.T, rule=self._restr_inf_odf)
        m.restr_demanda_odf = Constraint(m.T, rule=self._restr_demanda_odf)
        m.restr_reserva_odf = Constraint(m.T, rule=self._restr_reserva_odf)

        m.x_lb = Constraint(m.G, m.T, rule=self._restr_x_lb)
        m.x_ub = Constraint(m.G, m.T, rule=self._restr_x_ub)
        m.restr_pc_sup = Constraint(m.T, rule=lambda m, t: m.PC[t] <= m.pcmax[t])
        m.restr_pc_inf = Constraint(m.T, rule=lambda m, t: m.PC[t] >= m.pcmin[t])

        # Desativar as restrições com ODF por padrão
        m.restr_sup_odf.deactivate()
        m.restr_inf_odf.deactivate()
        m.restr_demanda_odf.deactivate()
        m.restr_reserva_odf.deactivate()

        self.model = m

        m.obj = Objective(rule=self._construir_objetivo, sense=minimize)

    def usar_odf(self, ativar: bool = False):
        """
        Ativa ou desativa o modo ODF no modelo.

        Quando ODF está ativado:
            - A função objetivo usa o termo _termo_a_odf(), ponderado por ODF(x).
            - As restrições padrão (com z[g, t]) são desativadas.
            - As restrições alternativas (com ODF) são ativadas.
            - A variável x[g, t] tem seus limites fixados em [0.0, 1e-4].

        Quando ODF está desativado:
            - A função objetivo usa o termo _termo_a(), com z[g, t].
            - As restrições com ODF são desativadas.
            - As restrições com z[g, t] são ativadas.
            - Os limites de x[g, t] são removidos (None).

        Args:
            ativar (bool): True para ativar ODF(x), False para retornar ao modelo padrão (default).
        """
        m = self.model
        self._usar_odf = ativar

        if ativar:
            for g in self.usinas:
                for t in self.periodos:
                    m.x[g, t].value = 1e-5
                    m.P[g, t].value = self.pmin[g]
                    m.PC[t].value = self.demanda[t]

            # Ativar restrições com ODF
            m.restr_sup_odf.activate()
            m.restr_inf_odf.activate()
            m.restr_demanda_odf.activate()
            m.restr_reserva_odf.activate()
            m.x_lb.activate()
            m.x_ub.activate()

            # Desativar restrições padrão
            m.restr_sup.deactivate()
            m.restr_inf.deactivate()
            m.restr_demanda.deactivate()
            m.restr_reserva.deactivate()

            # Limitar x[g, t] para ativação do modo contínuo próximo de 0
            for g in self.usinas:
                for t in self.periodos:
                    m.x[g, t].setlb(0.0)
                    m.x[g, t].setub(1e-4)

        else:
            # Ativar restrições padrão
            m.restr_sup.activate()
            m.restr_inf.activate()
            m.restr_demanda.activate()
            m.restr_reserva.activate()

            # Desativar restrições com ODF
            m.restr_sup_odf.deactivate()
            m.restr_inf_odf.deactivate()
            m.restr_demanda_odf.deactivate()
            m.restr_reserva_odf.deactivate()
            m.x_lb.deactivate()
            m.x_ub.deactivate()

            # Remover limites de x[g, t] (desnecessário no modo z)
            for g in self.usinas:
                for t in self.periodos:
                    m.x[g, t].setlb(None)
                    m.x[g, t].setub(None)

    def odf(self, x):
        """
        Função ODF(x): transformação sigmoidal utilizada para ponderar
        o impacto da variável de decisão x em restrições e na função objetivo.

        Args:
            x: variável Pyomo (m.x[g, t])

        Returns:
            expressão Pyomo com ODF(x)
        """
        return (exp(self.alpha * x) - 1) / (exp(self.alpha * x) + 1)


    def _restr_sup(self, m, g, t):
        """
        Restrição de limite superior de geração (modo ODF desativado).

        Neste modo, a geração P[g, t] está condicionada ao status binário
        da usina (z[g, t]). Isso significa que, se a usina estiver desligada,
        seu limite superior será zero. Quando ligada, o limite é Pmax[g].

        Args:
            m: Modelo Pyomo contendo as variáveis.
            g: Identificador da usina.
            t: Índice do período.

        Returns:
            Expressão simbólica Pyomo representando:
            P[g, t] <= Pmax[g] * z[g, t]
        """
        return m.P[g, t] <= self.pmax[g] * m.z[g, t]

    def _restr_sup_odf(self, m, g, t):
        """
        Restrição de limite superior de geração (modo ODF ativado).

        Quando ODF está ativado, a geração P[g, t] não depende do status
        binário da usina (z[g, t]). O limite superior é fixo em Pmax[g].

        Args:
            m: modelo Pyomo.
            g: usina.
            t: período.

        Returns:
            Restrição simbólica: P[g, t] <= Pmax[g]
        """
        return m.P[g, t] <= self.pmax[g]

    def _restr_inf(self, m, g, t):
        """
        Restrição de limite inferior de geração (modo ODF desativado).

        Neste modo, a geração P[g, t] está condicionada ao status binário
        da usina (z[g, t]). Isso significa que, se a usina estiver desligada,
        seu limite inferior será zero. Quando ligada, o limite é Pmax[g].

        Args:
            m: Modelo Pyomo contendo as variáveis.
            g: Identificador da usina.
            t: Índice do período.

        Returns:
            Expressão simbólica Pyomo representando:
            P[g, t] >= Pmax[g] * z[g, t]
        """
        return m.P[g, t] >= self.pmin[g] * m.z[g, t]

    def _restr_inf_odf(self, m, g, t):
        """
        Restrição de limite inferior de geração (modo ODF ativado).

        Quando ODF está ativado, a geração P[g, t] não depende do status
        binário da usina (z[g, t]). O limite inferior é fixo em Pmin[g].

        Args:
            m: modelo Pyomo.
            g: usina.
            t: período.

        Returns:
            Restrição simbólica: P[g, t] >= Pmin[g]
        """
        return m.P[g, t] >= self.pmin[g]

    def _restr_demanda(self, m, t):
        """
        Restrição de atendimento da demanda (modo ODF desativado).

        Quando ODF está desativado, considera-se que apenas as usinas ligadas
        (z[g, t] = 1) contribuem com geração. A soma ponderada das potências
        deve atender exatamente à demanda do período.

        Args:
            m: Modelo Pyomo contendo as variáveis.
            t: Índice do período.

        Returns:
            Expressão simbólica Pyomo representando:
            sum(P[g, t] * z[g, t] for g) == demanda[t]
        """
        return sum(m.P[g, t] * m.z[g, t] for g in m.G) == self.demanda[t]

    def _restr_demanda_odf(self, m, t):
        """
        Restrição de atendimento da demanda (modo ODF ativado).

        Quando ODF está ativado, a geração útil de cada usina é ponderada
        pela função sigmoide ODF(x[g, t]).

        Args:
            m: modelo Pyomo.
            t: período.

        Returns:
            Restrição simbólica: soma das gerações ponderadas por ODF(x) = demanda
        """
        return sum(self.odf(m.x[g, t]) * m.P[g, t] for g in m.G)+ m.PC[t] == self.demanda[t]

    def _restr_reserva(self, m, t):
        """
        Restrição de reserva operacional (modo ODF desativado).

        Considera-se que apenas as usinas ligadas (z[g, t] = 1) estão
        disponíveis para fornecer potência. A soma das capacidades máximas
        das usinas em operação deve ser suficiente para atender à demanda
        mais a reserva exigida no período.

        Args:
            m: Modelo Pyomo contendo os parâmetros e variáveis.
            t: Índice do período.

        Returns:
            Expressão simbólica Pyomo representando:
            sum(Pmax[g] * z[g, t] for g) >= demanda[t] + reserva[t]
        """
        return sum(self.pmax[g] * m.z[g, t] for g in m.G) >= self.demanda[t] + self.reserva[t]

    def _restr_reserva_odf(self, m, t):
        """
        Restrição de reserva operacional (modo ODF ativado).

        Quando ODF está ativado, a capacidade máxima disponível é
        ponderada pela ODF(x[g, t]), pois não há uso de z[g, t].

        Args:
            m: modelo Pyomo.
            t: período.

        Returns:
            Restrição simbólica: soma das capacidades ponderadas >= demanda + reserva
        """
        return sum(self.odf(m.x[g, t]) * self.pmax[g]
                   for g in m.G) + m.pcmax[t] >= self.demanda[t] + self.reserva[t]

    def _restr_x_lb(self, m, g, t):
        """
        Restrição de limite inferior para a variável contínua x[g, t],
        utilizada na formulação com ODF(x). Essa restrição é ativada
        apenas quando o modo ODF estiver ativado.

        Args:
            m: modelo Pyomo.
            g: identificador da usina.
            t: período.

        Returns:
            Expressão simbólica: x[g, t] >= 0.0
        """
        return m.x[g, t] >= 9.999e-5

    def _restr_x_ub(self, m, g, t):
        """
        Restrição de limite superior para a variável contínua x[g, t],
        limitando sua ativação a no máximo 1e-4 para efeito da função ODF(x).
        Essa restrição permite a extração do multiplicador de Lagrange π_xi.

        Args:
            m: modelo Pyomo.
            g: identificador da usina.
            t: período.

        Returns:
            Expressão simbólica: x[g, t] <= 1e-4
        """
        return m.x[g, t] <= 1e-4

    def _termo_a(self, m):
        """
        Define a função objetivo do modelo: minimização do custo total
        de operação das usinas ligadas, ao longo de todos os períodos.

        O custo é composto por uma parcela exponencial e uma quadrática.

        Args:
            m: modelo Pyomo.

        Returns:
            expressão Pyomo a ser minimizada.
        """
        return sum(
            self.a[g] * m.z[g, t] + self.b[g] * m.P[g, t] + self.c[g] * m.P[g, t] ** 2
            for g in m.G for t in m.T
        )

    def _termo_a_odf(self, m):
        """
        Calcula a parcela A da função objetivo utilizando a função ODF(x),
        aplicada ao custo de operação de cada usina.

        O custo ponderado é dado por:
            ODF(x) · (a + b·P + c·P²)

        Esta versão substitui o uso de z[g, t] e permite a modelagem contínua
        da decisão de despacho, útil para análises de sensibilidade e suavização.

        Args:
            m: Modelo Pyomo com variáveis P[g, t] e x[g, t] definidas.

        Returns:
            expressão simbólica Pyomo para a parte A da FOB com ODF(x).
        """
        return sum(
            (self.a[g] + self.b[g] * m.P[g, t] + self.c[g] * m.P[g, t] ** 2) * self.odf(m.x[g, t])
            for g in m.G for t in m.T
        )

    def _termo_d(self, m):
        """
        Termo D: penalidade pelo uso da potência de convergência PC[t],
        multiplicada por custo escalar (rho).
        """
        return sum(self.rho * m.PC[t] for t in m.T)

    def _construir_objetivo(self, m):
        """
        Constrói a função objetivo padrão (apenas termo A).
        Termos B, C, D e ODF serão incluídos posteriormente com métodos específicos.
        """
        if self._usar_odf:
            objetivo = self._termo_a_odf(m) + self._termo_d(m)
        else:
            objetivo = self._termo_a(m)
        return objetivo

    def solve(self, tee=False):
        """
        Resolve o modelo usando o solver IPOPT.

        Args:
            tee (bool): Se True, exibe a saída do solver no console.

        Returns:
            SolverResults: Objeto de resultado retornado pelo solver.
        """
        if self.model is None:
            self.construir_modelo()

        solver = SolverFactory("ipopt", executable='/users/gsjunqueira/SOLVER/Ipopt/bin/ipopt')
        return solver.solve(self.model, tee=tee)

    def get_resultados(self):
        """
        Extrai os resultados de geração ótima para cada usina e período.

        Returns:
            tuple: (resultados, custo_total)
                - resultados (dict): Dicionário aninhado com formato:
                    {periodo: {usina: {'status': int, 'geracao': float}}}
                - custo_total (float): Valor total da função objetivo otimizada.
        """
        if self.model is None:
            raise RuntimeError("Modelo ainda não construído ou resolvido.")

        resultados = {}
        for t in self.periodos:
            resultados[t] = {}
            for g in self.usinas:
                p = value(self.model.P[g, t])
                status = self.z_fixo[(g, t)]
                resultados[t][g] = {
                    "status": status,
                    "geracao": p,
                }

        custo_total = value(self.model.obj)
        # 🔍 Diagnóstico: quantas usinas ligadas por hora

        return resultados, custo_total

    def get_custos_tempo(self) -> Dict[int, float]:
        """
        Calcula o custo da função objetivo por período de tempo.

        Returns:
            dict: Dicionário {t: custo} com o custo total de cada período.
        """
        if self.model is None:
            raise RuntimeError("Modelo ainda não resolvido.")

        custos = {}
        for t in self.periodos:
            custo_t = 0.0
            for g in self.usinas:
                p = value(self.model.P[g, t])
                z = self.z_fixo.get((g, t), 1)
                custo = (self.a[g] + self.b[g] * p + self.c[g] * p**2) * z
                custo_t += custo
            custos[t] = custo_t
        return custos

    def get_lagrangianos(self) -> Dict[tuple, float]:
        """
        Extrai os multiplicadores de Lagrange associados às restrições de
        limite superior da variável x[g, t] (x[g, t] <= 1e-4), representando
        o impacto marginal da ativação da unidade no custo total (πₓᵢ(t)).

        Returns:
            dict: Dicionário {(g, t): lambda_x} com os multiplicadores
                de Lagrange associados a cada usina e período.
        """
        if self.model is None:
            raise RuntimeError("Modelo ainda não construído ou resolvido.")

        if not hasattr(self.model, "dual_x"):
            raise RuntimeError("O sufixo 'dual_x' não foi ativado no modelo.")

        m = self.model
        lambdas = {}

        for g in self.usinas:
            for t in self.periodos:
                lambda_xt = m.dual[m.x_ub[g, t]]
                lambdas[(g, t)] = lambda_xt

        return lambdas

    def diagnostico(self):
        """
        Diagnóstico dos valores de x, geração P e Lagrangiano λ_x para cada (g, t).
        Útil para verificar se a restrição superior de x[g, t] está ativa e sensível.
        """
        m = self.model
        lambdas = self.get_lagrangianos()
        print(f"{'Usina':<6} {'Tempo':<5} {'x[g,t]':>15} {'P[g,t]':>10} {'λ_x[g,t]':>15}")
        for g in self.usinas:
            for t in self.periodos:
                xval = value(m.x[g, t])
                pval = value(m.P[g, t])
                lval = lambdas[(g, t)]
                print(f"{g:<6} {t:<5} {xval:15.12f} {pval:10.2f} {lval:15.12f}")
