import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import json
import numpy as np

def evaluate_model():
    """
    Carrega o modelo, avalia com o limiar padrão (0.5), encontra o limiar ótimo
    e reavalia o modelo com o novo limiar.
    """
    print("--- INICIANDO ETAPA DE AVALIAÇÃO COM AJUSTE DE LIMIAR ---")
    
    # Caminhos
    model_path = 'models/modelo_decision_match_ai.joblib'
    test_data_dir = 'data/processed'
    X_test_path = os.path.join(test_data_dir, 'X_test.json')
    y_test_path = os.path.join(test_data_dir, 'y_test.json')
    results_dir = 'results'
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Carregar modelo e dados
    print(f"\nCarregando modelo e dados de teste...")
    try:
        model = joblib.load(model_path)
        X_test = pd.read_json(X_test_path, lines=True)
        y_test = pd.read_json(y_test_path, lines=True, typ='series')
    except FileNotFoundError:
        print("Erro: Arquivos de modelo ou de teste não encontrados. Execute o script de treinamento primeiro.")
        return
        
    print("\nGerando predições (probabilidades)...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # --- AVALIAÇÃO COM LIMIAR PADRÃO (0.5) ---
    print("\n--- MÉTRICAS DE PERFORMANCE (LIMIAR PADRÃO = 0.5) ---")
    y_pred_class_default = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_class_default))

    # --- ENCONTRANDO O LIMIAR ÓTIMO ---
    print("\n--- ENCONTRANDO O LIMIAR DE DECISÃO ÓTIMO ---")
    
    # 1. Gerar a Curva Precision-Recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Adiciona o limiar 1.0 para garantir que os arrays tenham o mesmo tamanho
    thresholds = np.append(thresholds, 1.0)
    
    # 2. Calcular o F1-Score para cada limiar
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) # Trata a divisão por zero
    
    # 3. Encontrar o limiar que maximiza o F1-Score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Limiar que maximiza o F1-Score (equilíbrio Precision/Recall): {optimal_threshold:.4f}")
    print(f"  - Com este limiar, a Precisão é: {precisions[optimal_idx]:.4f}")
    print(f"  - Com este limiar, o Recall é: {recalls[optimal_idx]:.4f}")
    print(f"  - F1-Score máximo: {f1_scores[optimal_idx]:.4f}")

    # --- AVALIAÇÃO COM O NOVO LIMIAR OTIMIZADO ---
    print("\n--- MÉTRICAS DE PERFORMANCE (LIMIAR OTIMIZADO) ---")
    y_pred_class_tuned = (y_pred_proba >= optimal_threshold).astype(int)
    print(classification_report(y_test, y_pred_class_tuned))
    
    # --- VISUALIZAÇÃO E SALVAMENTO ---
    print("\nGerando visualizações e salvando resultados...")
    
    # Gráfico da Curva Precision-Recall
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, label='Precisão')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1-Score', linestyle='--')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Limiar Ótimo ({optimal_threshold:.2f})')
    plt.title('Curva Precision-Recall vs. Limiar de Decisão')
    plt.xlabel('Limiar de Decisão')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    pr_curve_path = os.path.join(results_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    print(f"Gráfico da curva Precision-Recall salvo em '{pr_curve_path}'.")

    # Matriz de Confusão com o novo limiar
    cm = confusion_matrix(y_test, y_pred_class_tuned)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Match', 'Match'], yticklabels=['Não Match', 'Match'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão (Limiar Otimizado)')
    cm_path = os.path.join(results_dir, 'matriz_confusao_otimizada.png')
    plt.savefig(cm_path)
    print(f"Nova matriz de confusão salva em '{cm_path}'.")
    
    # Salvar métricas
    metrics = {
        'limiar_padrao_0.5': classification_report(y_test, y_pred_class_default, output_dict=True),
        'limiar_otimo': {
            'valor': optimal_threshold,
            'report': classification_report(y_test, y_pred_class_tuned, output_dict=True)
        }
    }
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas atualizadas salvas em '{metrics_path}'.")

    print("\n--- ETAPA DE AVALIAÇÃO CONCLUÍDA ---")

if __name__ == "__main__":
    evaluate_model()
    
