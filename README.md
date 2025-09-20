# **Decision Match AI 🤖**

## **📝 Descrição do Projeto**

Este projeto foi desenvolvido como solução para o **Datathon Decision**, um desafio focado em aplicar Inteligência Artificial para otimizar os processos de recrutamento e seleção da empresa Decision, uma especialista em bodyshop de TI.

O principal desafio da Decision é encontrar o talento ideal para cada vaga de forma ágil e precisa. O processo atual, embora robusto, enfrenta dores como a falta de padronização em entrevistas e a dificuldade em escalar a análise de um grande volume de candidatos.

A solução proposta é o **Decision Match AI**, um MVP (Minimum Viable Product) de um sistema de recomendação inteligente. A ferramenta utiliza um modelo de Machine Learning para analisar o perfil de um candidato em relação a uma vaga e gerar um **"score de compatibilidade"**. O objetivo é ranquear os candidatos, permitindo que o time de recrutadores foque seu tempo e energia nos perfis mais promissores, otimizando todo o fluxo de seleção.

O pipeline de Machine Learning foi construído seguindo as melhores práticas de MLOps, incluindo etapas de processamento de dados, engenharia de features, competição entre múltiplos modelos (Regressão Logística, XGBoost, LightGBM, CatBoost) com otimização de hiperparâmetros, avaliação de métricas e testes unitários para garantir a robustez da solução.

## **🛠️ Stack Utilizada**

* **Linguagem:** Python 3  
* **Análise e Processamento de Dados:** Pandas, NumPy  
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost  
* **Web App (Dashboard):** Streamlit  
* **Visualização de Dados:** Matplotlib, Seaborn  
* **Serialização do Modelo:** Joblib  
* **Testes:** Pytest

## **📂 Estrutura do Repositório**

├── app/  
│   └── app.py                  \# Script principal da aplicação com Streamlit  
├── data/  
│   ├── raw/                    \# Onde os dados brutos em JSON devem ser colocados  
│   └── processed/              \# Ficheiros intermediários gerados pelo pipeline (ignorados pelo .gitignore)  
├── models/  
│   └── modelo\_decision\_match\_ai.joblib \# Modelo vencedor treinado e serializado  
├── notebooks/  
│   ├── 1\_Data\_Processing.ipynb     \# Notebook da Etapa 1: Consolidação dos Dados  
│   ├── 2\_Feature\_Engineering.ipynb \# Notebook da Etapa 2: Criação de Features  
├── results/  
│   ├── metrics.json            \# Métricas de performance do modelo vencedor  
│   └── matriz\_confusao.png     \# Imagem da matriz de confusão  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── app\_utils.py            \# Funções de lógica da aplicação Streamlit  
│   ├── train.py                \# Script para treino e seleção do melhor modelo  
│   └── evaluate.py             \# Script para avaliação do modelo vencedor  
├── tests/  
│   ├── \_\_init\_\_.py  
│   ├── mock\_data.py            \# Dados falsos para os testes  
│   └── test\_app\_functions.py   \# Testes unitários das funções  
├── .gitignore                    \# Ficheiro para ignorar pastas e ficheiros  
├── README.md                     \# Documentação do projeto  
└── requirements.txt              \# Bibliotecas e versões para o ambiente

## **⚙️ Instruções de Instalação**

Para executar este projeto localmente, siga os passos abaixo.

**1\. Pré-requisitos:**

* Ter o [Python 3.8+](https://www.python.org/downloads/) instalado.  
* Ter o pip (gestor de pacotes do Python) instalado.

**2\. Clone o Repositório:**

git clone \[https://github.com/SEU\_USUARIO/SEU\_REPOSITORIO.git\](https://github.com/SEU\_USUARIO/SEU\_REPOSITORIO.git)  
cd SEU\_REPOSITORIO

**3\. Crie um Ambiente Virtual (Recomendado):**

\# Para Windows  
python \-m venv venv  
venv\\Scripts\\activate

\# Para macOS/Linux  
python3 \-m venv venv  
source venv/bin/activate

**4\. Instale as Dependências:**

pip install \-r requirements.txt

## **🚀 Pipeline de Execução (Ordem Correta)**

Para gerar os resultados do zero, siga esta ordem de execução. **Execute todos os comandos a partir do diretório raiz do projeto** (ex: Tech5/).

**Etapa 1: Processamento de Dados (Notebooks)**

* Coloque os ficheiros de dados brutos (vagas.json, prospects.json, applicants.json) no diretório data/raw/.  
* Execute os notebooks Jupyter na seguinte ordem:  
  1. notebooks/1\_Data\_Processing.ipynb  
  2. notebooks/2\_Feature\_Engineering.ipynb

**Etapa 2: Treino e Avaliação do Modelo (Scripts)**

* Execute os scripts Python a partir do terminal:  
  1. **Treinar o modelo:**  
     python src/train.py

  2. **Avaliar o modelo vencedor:**  
     python src/evaluate.py

**Etapa 3: Executar Testes Unitários (Opcional)**

* Para verificar a integridade das funções:  
  pytest

**Etapa 4: Iniciar a Aplicação Streamlit**

* Para interagir com a aplicação web:  
  streamlit run app/app.py  
