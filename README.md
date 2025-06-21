
# Power-NLP

Modelo de despacho não linear contínuo para unidades geradoras térmicas, com múltiplas heurísticas de priorização e benchmark de desempenho.

Este projeto foi desenvolvido como parte do curso de Métodos de Otimização, com foco na resolução do problema de despacho econômico usando Pyomo e estratégias baseadas em sensibilidade de custo e heurísticas de priorização.

---

## ⚙️ Funcionalidades

- Leitura de dados no formato `.m` para sistemas térmicos
- Resolução do despacho contínuo usando Pyomo (sem variáveis binárias)
- Implementação de heurísticas baseadas em:
  - Custo médio (`ISA`)
  - Custo marginal (`ISB`, `ISC`)
  - Custo médio ótimo (`ISD`)
  - Penalidade de múltiplas unidades (`ISG`)
  - Sensibilidade de Lagrange com ODF (`ILS`)
  - Tabela de relevância por sorteios (`ITR`)
- Comparação com método de força bruta (`IFB`)
- Análise de desempenho e visualização de resultados

---

## 🗂️ Estrutura do Projeto

```bash
.
├── data/                      # Arquivos de entrada (.m)
│   ├── UC_4UTES.m
│   └── UC_10GER.m
├── power_nlp/                 # Pacote principal
│   ├── heuristicas/          # Heurísticas (ISA, ISB, ISD, etc.)
│   ├── model_nlp/            # Implementação do modelo em Pyomo
│   ├── reader/               # Leitor de arquivos .m
│   └── utils/                # Utilitários e pré-processamento
├── utils/
│   └── clean.py              # Ferramentas auxiliares (ex: limpeza)
├── tests/                    # (Reservado para testes futuros)
├── main.py                   # Arquivo principal de execução
├── modelo_pyomo.txt          # Exemplo textual do modelo (documentação)
├── LICENSE                   # Licença MIT
├── pyproject.toml            # Gerenciador de dependências (Poetry)
└── README.md                 # Este arquivo
```

---

## 🚀 Execução

### 1. Instalação

Este projeto utiliza [Poetry](https://python-poetry.org/) para gerenciamento de pacotes.

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/power-nlp.git
cd power-nlp

# Instalar dependências
poetry install

# Ativar o ambiente virtual
poetry shell
```

---

### 2. Executar o modelo

```bash
python main.py
```

Você pode alternar entre os casos em `main.py`, por exemplo:

```python
caminho = "data/UC_4UTES.m"
# ou
caminho = "data/UC_10GER.m"
```

---

## 📊 Resultados

Ao final da execução, são exibidos:

- Custos por hora para cada heurística (ISA, ISB, ..., ITR, IFB)
- Tempos de execução por etapa
- Análise de desempenho consolidada

---

## 📄 Licença

Este projeto está licenciado sob os termos da [MIT License](LICENSE).

---

## 👨‍💻 Autor

**Giovani Santiago Junqueira**  
Mestrando em Engenharia Elétrica – Sistemas de Potência  
Universidade Federal de Juiz de Fora  
