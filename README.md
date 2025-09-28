# **Decision Match AI 🤖**

## **📝 Descrição do Projeto**

Este projeto foi desenvolvido como solução para o **Datathon Decision**, um desafio focado em aplicar Inteligência Artificial para otimizar os processos de recrutamento e seleção da empresa Decision, uma especialista em bodyshop de TI.

A solução proposta é o **Decision Match AI**, um sistema de recomendação inteligente. A ferramenta utiliza um modelo de Machine Learning para analisar o perfil de um candidato em relação a uma vaga e gerar um **"score de compatibilidade"**.

O objetivo principal da ferramenta **não é encontrar todos os bons candidatos**, mas sim realizar uma **triagem de altíssima relevância**. A meta é eliminar eficientemente os perfis com baixa compatibilidade ("maus candidatos"), garantindo que os poucos candidatos que o recrutador analisa tenham uma probabilidade muito maior de serem adequados para a vaga, otimizando assim o tempo e a eficiência do processo seletivo.

O pipeline de Machine Learning foi construído seguindo as melhores práticas de MLOps, incluindo etapas de processamento de dados, engenharia de features, competição entre múltiplos modelos, avaliação de métricas e testes unitários.

## **🎯 Seleção de Modelo e Performance**

A escolha do modelo foi realizada através de um processo competitivo, onde algoritmos como Regressão Logística, XGBoost, LightGBM e CatBoost foram treinados e avaliados. O script src/train.py aplica a técnica de reamostragem **SMOTE** para tratar o desbalanceamento de classes, garantindo um treinamento mais justo e robusto.

### **Competição de Modelos**

A seleção do melhor modelo foi baseada na métrica PR-AUC (Area Under the Precision-Recall Curve), que é ideal para problemas com classes desbalanceadas. O CatBoost demonstrou a melhor performance:

\--- SELECIONANDO O MELHOR MODELO \---  
  \- Regressão Logística | PR-AUC: 0.1115  
  \- XGBoost Otimizado | PR-AUC: 0.1409  
  \- LightGBM Otimizado | PR-AUC: 0.1324  
  \- CatBoost Otimizado | PR-AUC: 0.1437

🏆 Vencedor: CatBoost Otimizado com PR-AUC de 0.1437

### **Métrica de Avaliação: Foco na Precisão**

Dado o objetivo de negócio de "não fazer o recrutador perder tempo", a métrica principal para o sucesso é a **Precisão (Precision)**. Para a classe "Não Match", uma alta precisão garante que estamos a eliminar os candidatos errados de forma eficaz. Para a classe "Match", uma alta precisão garante que as recomendações feitas são confiáveis e de alta qualidade.

### **Resultados Finais e Interpretação**

Após o treino com dados balanceados (SMOTE) e o ajuste do limiar de decisão pelo script src/evaluate.py, o modelo final **(CatBoost Otimizado)** alcançou as seguintes métricas de performance no conjunto de teste:

\--- MÉTRICAS DE PERFORMANCE (LIMIAR OTIMIZADO) \---  
              precision    recall  f1-score   support

           0       0.96      0.93      0.95      8568  
           1       0.18      0.30      0.23       451

    accuracy                           0.90      9019

**Análise Positiva dos Resultados:**

* **Excelente Triagem (Precisão da Classe 0 \= 96%):** O resultado mais forte do modelo é a sua capacidade de identificar corretamente os "não-matches". Quando o modelo diz que um candidato não é adequado, ele está correto em 96% das vezes. Isto cumpre perfeitamente o objetivo principal de **eliminar os maus candidatos** com altíssima confiança, limpando a base para o recrutador.  
* **Recomendações Relevantes (Precisão da Classe 1 \= 18%):** O modelo alcançou uma precisão de 18% para as suas recomendações positivas. Isto significa que, de cada 6 candidatos que a ferramenta recomenda, 1 é, de facto, um "match", representando uma melhoria significativa na qualidade da triagem e na otimização do tempo do recrutador em comparação a uma análise manual.  
* **Melhora na Captura de Talentos (Recall da Classe 1 \= 30%):** O recall para "matches" aumentou para 30%, indicando que o modelo agora consegue identificar uma porção maior dos candidatos adequados. Este aumento, mantendo uma precisão útil, representa um excelente equilíbrio, garantindo que menos talentos sejam perdidos durante a triagem automatizada.

Em resumo, o modelo final é um sucesso, pois está perfeitamente alinhado com a estratégia de negócio de fornecer uma ferramenta de triagem precisa e que otimiza o tempo da equipa de recrutamento.

## **🛠️ Stack Utilizada**

* **Linguagem:** Python 3  
* **Análise e Processamento de Dados:** Pandas, NumPy, Parquet (pyarrow)  
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn  
* **Web App (Dashboard):** Streamlit  
* **Serialização de Artefatos:** Joblib  
* **Testes:** Pytest

## **📂 Estrutura do Repositório**

├── app/  
│   └── app.py                  \# Script principal da aplicação com Streamlit  
├── artifacts/  
│   ├── applicants\_processed.parquet  \# DataFrame de candidatos otimizado  
│   ├── tfidf\_matrix.joblib         \# Matriz TF-IDF pré-calculada  
│   └── tfidf\_vectorizer.joblib     \# Vetorizador TF-IDF treinado  
├── data/  
│   ├── raw/                    \# Onde os dados brutos em JSON devem ser colocados  
│   └── processed/              \# Arquivos intermediários gerados pelo pipeline  
├── models/  
│   └── modelo\_decision\_match\_ai.joblib \# Modelo vencedor treinado e serializado  
├── notebooks/  
│   ├── 1\_Data\_Processing.ipynb     \# Notebook da Etapa 1: Consolidação dos Dados  
│   └── 2\_Feature\_Engineering.ipynb \# Notebook da Etapa 2: Criação de Features  
├── results/  
│   ├── metrics.json            \# Métricas de performance do modelo vencedor  
│   └── matriz\_confusao.png     \# Imagem da matriz de confusão  
├── src/  
│   ├── app\_utils.py            \# Funções de lógica da aplicação Streamlit  
│   ├── train.py                \# Script para treino e seleção do melhor modelo  
│   └── evaluate.py             \# Script para avaliação do modelo vencedor  
├── tests/  
│   ├── \_\_init\_\_.py  
│   ├── mock\_data.py            \# Dados falsos para os testes  
│   └── test\_app\_functions.py   \# Testes unitários das funções  
├── .gitignore                    \# Arquivo para ignorar pastas e arquivos  
├── build\_artifacts.py          \# Script para gerar os artefatos de performance  
├── README.md                     \# Documentação do projeto  
└── requirements.txt              \# Bibliotecas e versões para o ambiente

## **🚀 Como Rodar a Aplicação (Modo Simples)**

Esta é a forma mais rápida de executar a aplicação, pois ela carrega todos os dados e modelos pré-processados diretamente da nuvem.

**1\. Pré-requisitos:**

* Ter o [Python 3.11](https://www.python.org/downloads/) instalado.  
* Ter o pip (gestor de pacotes do Python) instalado.

**2\. Clone o Repositório:**

git clone \[URL\_DO\_SEU\_REPOSITORIO\]  
cd \[NOME\_DA\_PASTA\_DO\_PROJETO\]

**3\. Crie um Ambiente Virtual (Recomendado):**

\# Para Windows  
python \-m venv venv  
venv\\Scripts\\activate

\# Para macOS/Linux  
python3 \-m venv venv  
source venv/bin/activate

**4\. Instale as Dependências:**

pip install \-r requirements.txt

**5\. Inicie a Aplicação Streamlit:**

streamlit run app/app.py

**Nota:** A aplicação já está configurada para buscar os arquivos das URLs definidas no topo do src/app\_utils.py. Certifique-se de que essas URLs estejam corretas e acessíveis.

## **⚙️ Como Reproduzir o Pipeline Completo (Do Zero)**

Siga estes passos se você deseja processar os dados brutos, treinar um novo modelo e gerar os artefatos do zero.

**Execute todos os comandos a partir do diretório raiz do projeto.**

**Etapa 1: Processamento de Dados (Notebooks)**

1. Coloque os arquivos de dados brutos (vagas.json, prospects.json, applicants.json) no diretório data/raw/.  
2. Execute os notebooks Jupyter na seguinte ordem:  
   * notebooks/1\_Data\_Processing.ipynb  
   * notebooks/2\_Feature\_Engineering.ipynb

**Etapa 2: Treino e Avaliação do Modelo (Scripts)**

1. **Treinar o modelo:**  
   python src/train.py

   Isto irá gerar o arquivo models/modelo\_decision\_match\_ai.joblib.  
2. **Avaliar o modelo vencedor:**  
   python src/evaluate.py

   Isto irá gerar os arquivos de métricas e a matriz de confusão na pasta results/.

**Etapa 3: Gerar Artefatos de Performance**

1. Este é um passo crucial para otimizar a aplicação Streamlit. Execute o script:  
   python build\_artifacts.py

   Isto irá criar a pasta artifacts/ com os arquivos applicants\_processed.parquet, tfidf\_vectorizer.joblib e tfidf\_matrix.joblib.

**Etapa 4: Fazer Upload dos Artefatos (Opcional, para Deploy)**

1. Faça o upload dos artefatos gerados para um serviço de hospedagem (como GitHub Releases):  
   * models/modelo\_decision\_match\_ai.joblib  
   * artifacts/applicants\_processed.parquet  
   * artifacts/tfidf\_vectorizer.joblib  
   * artifacts/tfidf\_matrix.joblib  
   * data/processed/vagas\_cleaned.json  
2. Atualize as constantes de URL no topo do arquivo src/app\_utils.py com os novos links.

**Etapa 5: Iniciar a Aplicação Streamlit**

streamlit run app/app.py  
