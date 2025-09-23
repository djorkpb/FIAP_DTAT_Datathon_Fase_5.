import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import average_precision_score
import joblib
import os

def train_model():
    """
    Carrega os dados, otimiza os hiperparâmetros de TODOS os modelos de boosting,
    treina, seleciona o melhor e salva o modelo e o conjunto de teste.
    """
    print("--- INICIANDO ETAPA DE TREINAMENTO (OTIMIZAÇÃO COMPLETA) ---")

    features_path = 'data/processed/dados_com_features.json'
    model_path = 'models/modelo_decision_match_ai.joblib'
    test_data_dir = 'data/processed'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    print(f"\nCarregando dados de '{features_path}'...")
    try:
        df = pd.read_json(features_path, lines=True)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{features_path}' não foi encontrado.")
        return

    print("\nPreparando os dados para o modelo...")
    features = [
        'similaridade_texto', 
        'match_nivel_ingles', 
        'match_nivel_espanhol',
        'match_sap',
        'match_anos_experiencia', 
        'skills_match_score', 
        'match_nivel_profissional'
    ]
    X = df[features]
    y = df['match']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dados divididos em {len(X_train)} para treino e {len(X_test)} para teste.")

    X_test.to_json(os.path.join(test_data_dir, 'X_test.json'), orient='records', lines=True)
    y_test.to_json(os.path.join(test_data_dir, 'y_test.json'), orient='records', lines=True)
    print(f"Conjunto de teste salvo em '{test_data_dir}'.")

    # --- Bloco de Treinamento e Otimização ---
    
    print("\n--- TREINANDO E OTIMIZANDO MODELOS ---")

    # Modelo 1: Regressão Logística (sem otimização, serve como baseline)
    print("\n1. Treinando Regressão Logística...")
    log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Parâmetros comuns para os modelos de boosting
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Modelo 2: Otimização do XGBoost
    print("\n2. Treinando e otimizando XGBoost...")
    param_grid_xgb = {
        'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight),
        param_distributions=param_grid_xgb, n_iter=20, scoring='average_precision', cv=3, verbose=1, random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    xgb_clf_tuned = xgb_search.best_estimator_

    # Modelo 3: Otimização do LightGBM
    print("\n3. Treinando e otimizando LightGBM...")
    param_grid_lgb = {
        'n_estimators': [100, 200, 300], 'max_depth': [-1, 3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [20, 31, 40],
        'subsample': [0.7, 0.8], 'colsample_bytree': [0.7, 0.8]
    }
    lgb_search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(objective='binary', metric='average_precision', random_state=42, class_weight='balanced', verbose=-1),
        param_distributions=param_grid_lgb, n_iter=20, scoring='average_precision', cv=3, verbose=1, random_state=42, n_jobs=-1
    )
    lgb_search.fit(X_train, y_train)
    lgb_clf_tuned = lgb_search.best_estimator_

    # Modelo 4: Otimização do CatBoost
    print("\n4. Treinando e otimizando CatBoost...")
    param_grid_cat = {
        'iterations': [100, 200, 300], 'depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1], 'l2_leaf_reg': [1, 3, 5],
        'subsample': [0.7, 0.8]
    }
    cat_search = RandomizedSearchCV(
        estimator=cb.CatBoostClassifier(loss_function='Logloss', eval_metric='PRAUC', random_seed=42, auto_class_weights='Balanced', verbose=0),
        param_distributions=param_grid_cat, n_iter=20, scoring='average_precision', cv=3, verbose=1, random_state=42, n_jobs=-1
    )
    cat_search.fit(X_train, y_train)
    cat_clf_tuned = cat_search.best_estimator_

    # --- Bloco de Seleção ---
    print("\n--- SELECIONANDO O MELHOR MODELO ---")
    
    models = {
        "Regressão Logística": log_reg,
        "XGBoost Otimizado": xgb_clf_tuned,
        "LightGBM Otimizado": lgb_clf_tuned,
        "CatBoost Otimizado": cat_clf_tuned
    }
    
    scores = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_pred_proba)
        scores[name] = pr_auc
        print(f"  - {name} | PR-AUC: {pr_auc:.4f}")
        
    vencedor_nome = max(scores, key=scores.get)
    modelo_vencedor = models[vencedor_nome]
    
    print(f"\n🏆 Vencedor: {vencedor_nome} com PR-AUC de {scores[vencedor_nome]:.4f}")
    
    joblib.dump(modelo_vencedor, model_path)
    print(f"\nModelo vencedor salvo com sucesso em '{model_path}'.")
    print("\n--- ETAPA DE TREINAMENTO CONCLUÍDA ---")

if __name__ == "__main__":
    train_model()

