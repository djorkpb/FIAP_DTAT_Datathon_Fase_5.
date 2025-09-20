import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import json

def evaluate_model():
    """
    Carrega o modelo treinado e o conjunto de teste, calcula as métricas de performance
    e salva os resultados (métricas e imagem) no diretório 'results/'.
    """
    print("--- INICIANDO ETAPA DE AVALIAÇÃO ---")
    
    # Caminhos relativos ao diretório raiz do projeto
    model_path = 'models/modelo_decision_match_ai.joblib'
    test_data_dir = 'data/processed'
    X_test_path = os.path.join(test_data_dir, 'X_test.json')
    y_test_path = os.path.join(test_data_dir, 'y_test.json')
    
    # --- ALTERAÇÃO: Caminhos organizados no diretório 'results' ---
    results_dir = 'results'
    image_path = os.path.join(results_dir, 'matriz_confusao.png')
    metrics_path = os.path.join(results_dir, 'metrics.json')
    # --- FIM DA ALTERAÇÃO ---
    
    # Garante que o diretório de resultados exista
    os.makedirs(results_dir, exist_ok=True)
    
    # Bloco 1: Carregar modelo e dados de teste
    print(f"\nCarregando modelo de '{model_path}' e dados de teste de '{test_data_dir}'...")
    try:
        model = joblib.load(model_path)
        X_test = pd.read_json(X_test_path, lines=True)
        y_test = pd.read_json(y_test_path, lines=True, typ='series')
        print("Modelo e dados de teste carregados com sucesso.")
    except FileNotFoundError:
        print("Erro: Arquivos de modelo ou de teste não encontrados. Execute o script de treinamento primeiro.")
        return
        
    # Bloco 2: Gerar predições
    print("\nGerando predições no conjunto de teste...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)
    
    # Bloco 3: Calcular e Exibir Métricas
    print("\n--- MÉTRICAS DE PERFORMANCE ---")
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"  - AUC-ROC: {auc_roc:.4f}")
    print(f"  - PR-AUC (Average Precision): {pr_auc:.4f}")
    print("\n  - Relatório de Classificação:")
    print(classification_report(y_test, y_pred_class))
    
    # Bloco 4: Gerar e Salvar Matriz de Confusão
    print("\nGerando a Matriz de Confusão...")
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Match', 'Match'], yticklabels=['Não Match', 'Match'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {type(model).__name__}')
    plt.savefig(image_path)
    print(f"Matriz de confusão salva como '{image_path}'.")
    
    # Salvar métricas em um arquivo JSON
    metrics = {
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'classification_report': classification_report(y_test, y_pred_class, output_dict=True)
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas salvas em '{metrics_path}'.")

    print("\n--- ETAPA DE AVALIAÇÃO CONCLUÍDA ---")

if __name__ == "__main__":
    evaluate_model()

