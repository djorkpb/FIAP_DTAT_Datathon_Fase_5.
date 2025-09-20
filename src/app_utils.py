import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st
import joblib
import json

# --- FUNÇÕES DE PRÉ-COMPUTAÇÃO OTIMIZADAS ---

@st.cache_resource(show_spinner=False)
def get_precomputed_data():
    """
    Função pesada que pré-processa todos os candidatos uma única vez.
    O resultado (vectorizer e applicant_features) fica em cache.
    """
    try:
        applicants_df = pd.read_json('https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./releases/download/datathon_fase_5/applicants.json', orient='index')
        applicants_df.index.name = 'id_candidato'
        applicants_df.reset_index(inplace=True)
        
        mask = applicants_df['cv_pt'].notna() & applicants_df['cv_pt'].str.contains(
            'qualificaç(?:ão|ões)', case=False, na=False, regex=True
        )
        applicants_df = applicants_df[mask].copy() # Usa .copy() para evitar SettingWithCopyWarning

        if 'informacoes_profissionais' in applicants_df.columns:
            applicants_df['informacoes_profissionais'] = applicants_df['informacoes_profissionais'].apply(
                lambda x: {**x, 'area_atuacao': x.get('area_atuacao', '').rstrip('-').strip()} if isinstance(x, dict) else x
            )

        applicants_df['candidato_text'] = applicants_df.apply(get_candidate_text, axis=1)
        
        vectorizer = TfidfVectorizer()
        candidato_tfidf_matrix = vectorizer.fit_transform(applicants_df['candidato_text'])
        
        mapa_niveis = {'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Técnico': 3, 'Avançado': 4, 'Fluente': 5}
        applicants_df['candidato_nivel_ingles_num'] = applicants_df['formacao_e_idiomas'].apply(lambda x: mapa_niveis.get(str(x.get('nivel_ingles', 'Nenhum')).strip() or 'Nenhum', 0) if isinstance(x, dict) else 0)
        applicants_df['candidato_nivel_espanhol_num'] = applicants_df['formacao_e_idiomas'].apply(lambda x: mapa_niveis.get(str(x.get('nivel_espanhol', 'Nenhum')).strip() or 'Nenhum', 0) if isinstance(x, dict) else 0)
        applicants_df['candidato_tem_sap'] = applicants_df['candidato_text'].str.contains('sap', flags=re.IGNORECASE, regex=True).astype(int)
        
        def extract_years(text):
            if not isinstance(text, str): return 0
            matches = re.findall(r'(\d{1,2})\s+an[o|os]', text, re.IGNORECASE)
            return max([int(y) for y in matches]) if matches else 0
        applicants_df['candidato_anos_exp_val'] = applicants_df['candidato_text'].apply(extract_years)

        mapa_senioridade_num = {'júnior': 1, 'junior': 1, 'pleno': 2, 'sênior': 3, 'senior': 3, 'especialista': 4}
        def get_seniority_num(text):
            if not isinstance(text, str): return 0
            text = text.lower()
            for level, value in mapa_senioridade_num.items():
                if level in text: return value
            return 0
        applicants_df['candidato_nivel_prof_num'] = applicants_df['informacoes_profissionais'].apply(lambda x: get_seniority_num(x.get('nivel_profissional', '')) if isinstance(x, dict) else 0)

        # Otimização de performance: Pré-computação de competências
        COMPREHENSIVE_SKILLS_LIST = [
            'python', 'java', 'javascript', 'typescript', 'c#', 'c++', 'php', 'ruby', 'go', 'swift', 'kotlin', 'sql', 'pl/sql',
            'react', 'angular', 'vue', 'svelte', 'jquery', 'node.js', 'django', 'flask', 'spring', 'ruby on rails', '.net', 'laravel',
            'aws', 'azure', 'google cloud', 'gcp', 'oracle cloud', 'oci', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 
            'sql server', 'dynamodb', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ansible', 'terraform', 'ci/cd',
            'agile', 'scrum', 'kanban', 'api', 'rest', 'graphql', 'microservices', 'linux', 'unix', 'html', 'css', 'sap'
        ]
        applicants_df['candidate_skill_set'] = applicants_df['candidato_text'].apply(
            lambda text: {s for s in COMPREHENSIVE_SKILLS_LIST if s in text}
        )

        return applicants_df, vectorizer, candidato_tfidf_matrix

    except FileNotFoundError as e:
        st.error(f"Erro ao carregar applicants.json: {e}.")
        return None, None, None

@st.cache_data(show_spinner=False)
def load_base_data():
    """Carrega dados que raramente mudam (modelo e vagas)."""
    try:
        model = joblib.load('models/modelo_decision_match_ai.joblib')
        
        with open('data/raw/vagas.json', 'r', encoding='utf-8') as f:
            vagas_data = json.load(f)
        
        processed_vagas = []
        for vaga_id, data in vagas_data.items():
            record = {'id_vaga': vaga_id}
            if isinstance(data.get('informacoes_basicas'), dict):
                for key, value in data['informacoes_basicas'].items():
                    record[f'info.{key}'] = value
            if isinstance(data.get('perfil_vaga'), dict):
                for key, value in data['perfil_vaga'].items():
                    record[f'perfil.{key}'] = value
            processed_vagas.append(record)
        vagas_df = pd.DataFrame(processed_vagas)
        
        if 'perfil.areas_atuacao' in vagas_df.columns:
            vagas_df['perfil.areas_atuacao'] = vagas_df['perfil.areas_atuacao'].str.rstrip('-').str.strip()
        
        if 'perfil.nivel_ingles' in vagas_df.columns:
            vagas_df['perfil.nivel_ingles'] = vagas_df['perfil.nivel_ingles'].fillna('Nenhum')
        if 'perfil.nivel_espanhol' in vagas_df.columns:
            vagas_df['perfil.nivel_espanhol'] = vagas_df['perfil.nivel_espanhol'].fillna('Nenhum')

    except FileNotFoundError as e:
        st.error(f"Erro ao carregar arquivos essenciais: {e}.")
        return None, None
        
    return model, vagas_df

# --- FUNÇÕES AUXILIARES ---

def get_vaga_text(vaga_series):
    """Combina os textos relevantes de uma vaga."""
    texts_to_join = []
    for key in ['info.titulo_vaga', 'perfil.principais_atividades', 'perfil.competencia_tecnicas_e_comportamentais']:
        text = vaga_series.get(key)
        if isinstance(text, str) and text.strip():
            texts_to_join.append(text.lower())
    return " ".join(texts_to_join)

def get_candidate_text(candidato_series):
    """Combina os textos de um candidato."""
    texts_to_join = []
    if isinstance(candidato_series.get('cv_pt'), str):
        texts_to_join.append(candidato_series['cv_pt'].lower())
    if isinstance(candidato_series.get('informacoes_profissionais'), dict):
        for key in ['conhecimentos_tecnicos', 'area_atuacao', 'qualificacoes']:
            text = candidato_series['informacoes_profissionais'].get(key)
            if isinstance(text, str):
                texts_to_join.append(text.lower())
    return ' '.join(texts_to_join)

def get_explanation_strings(vaga_series, candidato_series, vaga_anos_exp, candidato_anos_exp):
    """Gera as strings descritivas para a UI."""
    vaga_req_ingles_str = str(vaga_series.get('perfil.nivel_ingles', 'Nenhum')).strip() or 'Nenhum'
    vaga_req_espanhol_str = str(vaga_series.get('perfil.nivel_espanhol', 'Nenhum')).strip() or 'Nenhum'
    
    formacao = candidato_series.get('formacao_e_idiomas', {})
    if not isinstance(formacao, dict): formacao = {}
    
    candidato_ingles = str(formacao.get('nivel_ingles', 'Nenhum')).strip() or 'Nenhum'
    candidato_espanhol = str(formacao.get('nivel_espanhol', 'Nenhum')).strip() or 'Nenhum'

    vaga_sap = "Sim" if vaga_series.get('info.vaga_sap') == 'Sim' else "Não"
    candidato_text = get_candidate_text(candidato_series)
    candidato_sap = "Sim" if 'sap' in candidato_text else "Não"
    
    mapa_senioridade = {'júnior': 'Júnior', 'junior': 'Júnior', 'pleno': 'Pleno', 'sênior': 'Sênior', 'senior': 'Sênior', 'especialista': 'Especialista'}
    vaga_senioridade = 'Nenhum'
    for key, value in mapa_senioridade.items():
        if key in str(vaga_series.get('perfil.nivel profissional', '')).lower():
            vaga_senioridade = value
            break
    candidato_senioridade = 'Nenhum'
    if isinstance(candidato_series.get('informacoes_profissionais'), dict):
        for key, value in mapa_senioridade.items():
            if key in str(candidato_series['informacoes_profissionais'].get('nivel_profissional', '')).lower():
                candidato_senioridade = value
                break

    return {
        'vaga_req_ingles': vaga_req_ingles_str, 'candidato_tem_ingles': candidato_ingles,
        'vaga_req_espanhol': vaga_req_espanhol_str, 'candidato_tem_espanhol': candidato_espanhol,
        'vaga_req_sap': vaga_sap, 'candidato_tem_sap': candidato_sap,
        'vaga_senioridade': vaga_senioridade, 'candidato_senioridade': candidato_senioridade,
        'vaga_anos_exp': vaga_anos_exp, 'candidato_anos_exp': candidato_anos_exp,
    }

def run_bulk_analysis(_vaga_series, _applicants_processed_df, _model, _vectorizer, _candidato_tfidf_matrix):
    """Executa a análise de compatibilidade de forma otimizada."""
    results_df = _applicants_processed_df.copy()
    
    vaga_text = get_vaga_text(_vaga_series)
    vaga_vector = _vectorizer.transform([vaga_text])
    results_df['similaridade_texto'] = cosine_similarity(vaga_vector, _candidato_tfidf_matrix)[0]
    
    mapa_niveis = {'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Técnico': 3, 'Avançado': 4, 'Fluente': 5}
    vaga_req_ingles_num = mapa_niveis.get(str(_vaga_series.get('perfil.nivel_ingles', 'Nenhum')).strip(), 0)
    vaga_req_espanhol_num = mapa_niveis.get(str(_vaga_series.get('perfil.nivel_espanhol', 'Nenhum')).strip(), 0)
    results_df['match_nivel_ingles'] = (results_df['candidato_nivel_ingles_num'] >= vaga_req_ingles_num).astype(int)
    results_df['match_nivel_espanhol'] = (results_df['candidato_nivel_espanhol_num'] >= vaga_req_espanhol_num).astype(int)

    vaga_req_sap = 1 if _vaga_series.get('info.vaga_sap') == 'Sim' else 0
    results_df['match_sap'] = ((vaga_req_sap == 0) | (results_df['candidato_tem_sap'] == 1)).astype(int)
    
    def extract_years(text):
        if not isinstance(text, str): return 0
        matches = re.findall(r'(\d{1,2})\s+an[o|os]', text, re.IGNORECASE)
        return max([int(y) for y in matches]) if matches else 0
    vaga_anos_exp_val = extract_years(vaga_text)
    results_df['vaga_anos_exp_val'] = vaga_anos_exp_val
    results_df['match_anos_experiencia'] = (results_df['candidato_anos_exp_val'] >= vaga_anos_exp_val).astype(int)

    mapa_senioridade_num = {'júnior': 1, 'junior': 1, 'pleno': 2, 'sênior': 3, 'senior': 3, 'especialista': 4}
    def get_seniority_num(text):
        if not isinstance(text, str): return 0
        text = text.lower()
        for level, value in mapa_senioridade_num.items():
            if level in text: return value
        return 0
    vaga_nivel_prof_num = get_seniority_num(_vaga_series.get('perfil.nivel profissional', ''))
    results_df['match_nivel_profissional'] = (results_df['candidato_nivel_prof_num'] >= vaga_nivel_prof_num).astype(int)

    COMPREHENSIVE_SKILLS_LIST = [
        'python', 'java', 'javascript', 'typescript', 'c#', 'c++', 'php', 'ruby', 'go', 'swift', 'kotlin', 'sql', 'pl/sql',
        'react', 'angular', 'vue', 'svelte', 'jquery', 'node.js', 'django', 'flask', 'spring', 'ruby on rails', '.net', 'laravel',
        'aws', 'azure', 'google cloud', 'gcp', 'oracle cloud', 'oci', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 
        'sql server', 'dynamodb', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ansible', 'terraform', 'ci/cd',
        'agile', 'scrum', 'kanban', 'api', 'rest', 'graphql', 'microservices', 'linux', 'unix', 'html', 'css', 'sap'
    ]
    req_skills = {s for s in COMPREHENSIVE_SKILLS_LIST if s in vaga_text}
    results_df['required_skills'] = [sorted(list(req_skills))] * len(results_df)

    if not req_skills:
        results_df['skills_match_score'] = 1.0
    else:
        results_df['skills_match_score'] = results_df['candidate_skill_set'].apply(
            lambda cand_skills: len(req_skills.intersection(cand_skills)) / len(req_skills)
        )
    
    results_df['matched_skills'] = results_df.apply(lambda row: sorted(list(set(row['required_skills']) & row['candidate_skill_set'])), axis=1)
    results_df['extra_skills'] = results_df.apply(lambda row: sorted(list(row['candidate_skill_set'] - set(row['required_skills']))), axis=1)
    
    feature_columns = [
        'similaridade_texto', 'match_nivel_ingles', 'match_nivel_espanhol',
        'match_sap', 'match_anos_experiencia', 'skills_match_score', 
        'match_nivel_profissional'
    ]
    scores = _model.predict_proba(results_df[feature_columns])[:, 1]
    results_df['Score'] = scores
    
    results_df['Nome'] = results_df['infos_basicas'].apply(lambda x: x.get('nome', 'N/A') if isinstance(x, dict) else 'N/A')
    results_df.rename(columns={'id_candidato': 'ID'}, inplace=True)
    
    return results_df.sort_values(by='Score', ascending=False).head(10)

