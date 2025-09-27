Documento de Pipeline: Decision Match AI
Projeto: Decision Match AI

Autores: Alessanda, Erick, Fausto, Lucas

Data: 26 de setembro de 2025

Versão: 2.0

1. Visão Geral
Este documento descreve o pipeline de Machine Learning e operações (MLOps) para o projeto Decision Match AI. O objetivo do pipeline é transformar dados brutos de recrutamento numa aplicação web interativa que classifica candidatos com base no seu "score de compatibilidade" com vagas de TI e Administrativas. O processo é dividido em quatro etapas sequenciais: Engenharia de Dados, Treinamento e Avaliação do Modelo, Geração de Artefatos para Performance e Execução da Aplicação.

2. Etapa 1: Engenharia de Dados
Esta fase inicial foca na recolha, limpeza, consolidação e criação de características (features) a partir dos dados brutos de múltiplas fontes.

Objetivo: Produzir um conjunto de dados limpo e enriquecido, pronto para o treinamento do modelo.

Fontes de Dados (Entradas):

vagas.json: Dados brutos sobre as vagas abertas.

prospects.json: Dados sobre o interesse de candidatos em vagas.

applicants.json: Perfis detalhados dos candidatos.

Processo:

Consolidação: O notebook notebooks/1_Data_Processing.ipynb é executado para unificar os três ficheiros JSON brutos numa base de dados única.

Engenharia de Features: O notebook notebooks/2_Feature_Engineering.ipynb é executado para calcular features analíticas, como similaridade de texto, compatibilidade de idiomas e senioridade.

Artefatos Gerados (Saídas):

data/processed/vagas_cleaned.json: Ficheiro intermediário com dados de vagas limpos.

data/processed/applicants_cleaned.json: Ficheiro intermediário com dados de candidatos limpos e achatados.

data/processed/dados_com_features.json: O conjunto de dados final, pronto para a próxima etapa.

3. Etapa 2: Treinamento e Avaliação do Modelo
Nesta fase, o conjunto de dados com features é utilizado para treinar e validar múltiplos algoritmos de Machine Learning, resultando na seleção do modelo com melhor performance para o objetivo de negócio.

Objetivo: Produzir um modelo de classificação serializado, validado e otimizado.

Entrada:

data/processed/dados_com_features.json.

Processo:

Reamostragem (SMOTE): O script src/train.py aplica a técnica SMOTE nos dados de treino para corrigir o desbalanceamento de classes, criando exemplos sintéticos da classe minoritária ("match").

Treinamento Competitivo: O mesmo script treina e otimiza quatro modelos (Regressão Logística, XGBoost, LightGBM e CatBoost) com os dados balanceados.

Seleção do Vencedor: O modelo com o maior score de PR-AUC (Average Precision) no conjunto de teste é selecionado como o vencedor.

Avaliação e Ajuste de Limiar: O script src/evaluate.py é executado para gerar métricas de performance detalhadas e encontrar o limiar de decisão ótimo que maximiza o F1-Score, equilibrando precisão e recall.

Artefatos Gerados (Saídas):

models/modelo_decision_match_ai.joblib: O objeto do modelo treinado e otimizado.

results/metrics.json: Um ficheiro com as métricas de performance (com o limiar padrão e o otimizado).

results/matriz_confusao_otimizada.png: A matriz de confusão gerada com o limiar de decisão otimizado.

results/precision_recall_curve.png: Gráfico da curva Precision-Recall vs. Limiar.

4. Etapa 3: Geração de Artefatos para Performance
Para garantir que a aplicação final seja rápida e responsiva, esta etapa pré-processa os dados dos candidatos que serão consumidos pela interface do utilizador.

Objetivo: Realizar os cálculos mais pesados de forma offline para otimizar o tempo de carregamento da aplicação.

Entrada:

URL para applicants_cleaned.json no GitHub Releases.

Processo:

O script build_artifacts.py é executado. Ele baixa os dados dos candidatos, realiza a engenharia de features (texto combinado, níveis de idioma, etc.) e a vetorização de texto (TF-IDF), salvando os resultados em formatos otimizados.

Artefatos Gerados (Saídas):

artifacts/applicants_processed.parquet: Um DataFrame em formato Parquet, com todas as features dos candidatos já calculadas.

artifacts/tfidf_vectorizer.joblib: O objeto do vetorizador TF-IDF já treinado.

artifacts/tfidf_matrix.joblib: A matriz TF-IDF pré-calculada, pronta para comparações de similaridade.

5. Etapa 4: Execução da Aplicação Streamlit
A fase final do pipeline, onde os artefatos gerados são consumidos por uma interface web interativa.

Objetivo: Disponibilizar o modelo de IA para os utilizadores finais (recrutadores).

Entradas:

Os artefatos gerados nas etapas 2 e 3, hospedados na nuvem (GitHub Releases).

Processo:

O comando streamlit run app/app.py é executado.

O app.py, auxiliado pelas funções em src/app_utils.py, carrega os artefatos da nuvem.

A aplicação renderiza a interface, permitindo que o utilizador filtre vagas e receba um ranking de candidatos compatíveis em tempo real.

Resultado Final:

Uma aplicação web funcional e performática, o Decision Match AI, disponível para otimizar o processo de recrutamento.