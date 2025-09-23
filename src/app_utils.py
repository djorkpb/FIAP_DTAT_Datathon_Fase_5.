import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st
import joblib
import json
import requests
import io
import ast

# --- FONTES DE DADOS NA NUVEM (PADRONIZADO) ---
VAGAS_URL = "https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./releases/download/datathon_fase_5/vagas_cleaned.json"
MODEL_URL = "https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./blob/main/models/modelo_decision_match_ai.joblib"

# !!! SUBSTITUA PELAS SUAS URLS DOS ARTEFATOS GERADOS PELO build_artifacts.py !!!
APPLICANTS_PROCESSED_URL = "https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./releases/download/datathon_fase_5/applicants_processed.parquet"
VECTORIZER_URL = "https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./releases/download/datathon_fase_5/tfidf_vectorizer.joblib"
MATRIX_URL = "https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./releases/download/datathon_fase_5/tfidf_matrix.joblib"



# --- FUNÇÕES AUXILIARES GERAIS ---

def extract_years(text):
    """Extrai o número de anos de experiência de um texto."""
    if not isinstance(text, str): return 0
    matches = re.findall(r'(\d{1,2})\s+an[o|os]', text, re.IGNORECASE)
    return max([int(y) for y in matches]) if matches else 0

def get_seniority_num(text):
    """Extrai o nível de senioridade numérico de um texto."""
    mapa_senioridade_num = {
        'júnior': 1, 'junior': 1,
        'analista': 2, 'pleno': 2,
        'sênior': 3, 'senior': 3,
        'especialista': 4, 'coordenador': 4,
        'gerente': 5
    }
    if not isinstance(text, str): return 0
    text = text.lower()
    for level, value in mapa_senioridade_num.items():
        if level in text: return value
    return 0 # Retorna 0 se nenhum nível for encontrado (Não Informado)

# --- FUNÇÕES DE CARREGAMENTO OTIMIZADAS ---

@st.cache_resource(show_spinner=False)
def get_precomputed_data():
    """Carrega os artefatos dos candidatos já processados da nuvem."""
    try:
        applicants_df = pd.read_parquet(APPLICANTS_PROCESSED_URL)

        response_vectorizer = requests.get(VECTORIZER_URL)
        response_vectorizer.raise_for_status()
        vectorizer = joblib.load(io.BytesIO(response_vectorizer.content))

        response_matrix = requests.get(MATRIX_URL)
        response_matrix.raise_for_status()
        candidato_tfidf_matrix = joblib.load(io.BytesIO(response_matrix.content))
        
        return applicants_df, vectorizer, candidato_tfidf_matrix
        
    except Exception as e:
        st.error(f"Erro ao carregar os artefatos pré-processados da nuvem: {e}")
        st.info("Verifique se as URLs dos artefatos (.parquet, .joblib) estão corretas e acessíveis.")
        return None, None, None


@st.cache_data(show_spinner=False)
def load_base_data():
    """Carrega, processa e otimiza dados de modelo e vagas a partir de URLs."""
    try:
        response_model = requests.get(MODEL_URL)
        response_model.raise_for_status()
        model = joblib.load(io.BytesIO(response_model.content))
        
        vagas_nested_df = pd.read_json(VAGAS_URL, orient='index')
        vagas_nested_df.reset_index(inplace=True)
        vagas_nested_df.rename(columns={'index': 'id_vaga'}, inplace=True)

        info_df = pd.json_normalize(vagas_nested_df['informacoes_basicas']).add_prefix('info.')
        perfil_df = pd.json_normalize(vagas_nested_df['perfil_vaga']).add_prefix('perfil.')
        
        vagas_df = pd.concat([
            vagas_nested_df.drop(columns=['informacoes_basicas', 'perfil_vaga']),
            info_df,
            perfil_df
        ], axis=1)

        vagas_df['vaga_text'] = vagas_df.apply(get_vaga_text, axis=1)
        level_map = {level: i for i, level in enumerate(['Nenhum', 'Básico', 'Intermediário', 'Técnico', 'Avançado', 'Fluente'])}
        vagas_df['nivel_num_ingles'] = vagas_df['perfil.nivel_ingles'].map(level_map).fillna(0)
        vagas_df['nivel_num_espanhol'] = vagas_df['perfil.nivel_espanhol'].map(level_map).fillna(0)

        return model, vagas_df
        
    except Exception as e:
        st.error(f"Erro ao carregar arquivos essenciais da nuvem: {e}.")
        return None, None

# --- FUNÇÕES AUXILIARES ---

def get_vaga_text(vaga_series):
    """Combina os textos relevantes de uma vaga."""
    texts_to_join = [
        str(vaga_series.get('info.titulo_vaga', '')),
        str(vaga_series.get('perfil.principais_atividades', '')),
        str(vaga_series.get('perfil.competencia_tecnicas_e_comportamentais', ''))
    ]
    return " ".join(texts_to_join).lower()

def get_candidate_text(candidato_series):
    """Combina os textos de um candidato."""
    texts_to_join = [
        str(candidato_series.get('cv_pt', '')),
        str(candidato_series.get('informacoes_profissionais.conhecimentos_tecnicos', '')),
        str(candidato_series.get('informacoes_profissionais.area_atuacao', '')),
        str(candidato_series.get('informacoes_profissionais.qualificacoes', ''))
    ]
    return ' '.join(texts_to_join).lower()

def get_explanation_strings(vaga_series, candidato_series, vaga_anos_exp, candidato_anos_exp):
    """Gera as strings descritivas para a UI (estrutura achatada)."""
    
    # Lógica aprimorada para tratar corretamente valores nulos ou vazios
    vaga_senioridade_val = vaga_series.get('perfil.nivel profissional')
    if vaga_senioridade_val and str(vaga_senioridade_val).strip():
        vaga_senioridade = str(vaga_senioridade_val).strip()
    else:
        vaga_senioridade = "Não Informado"

    candidato_senioridade_val = candidato_series.get('informacoes_profissionais.nivel_profissional')
    if candidato_senioridade_val and str(candidato_senioridade_val).strip():
        candidato_senioridade = str(candidato_senioridade_val).strip()
    else:
        candidato_senioridade = "Não Informado"
    
    return {
        'vaga_req_ingles': str(vaga_series.get('perfil.nivel_ingles', 'Nenhum')),
        'candidato_tem_ingles': str(candidato_series.get('formacao_e_idiomas.nivel_ingles', 'Nenhum')),
        'vaga_req_espanhol': str(vaga_series.get('perfil.nivel_espanhol', 'Nenhum')),
        'candidato_tem_espanhol': str(candidato_series.get('formacao_e_idiomas.nivel_espanhol', 'Nenhum')),
        'vaga_req_sap': str(vaga_series.get('info.vaga_sap', 'Não')),
        'candidato_tem_sap': "Sim" if candidato_series.get('candidato_tem_sap') == 1 else "Não",
        'vaga_senioridade': vaga_senioridade,
        'candidato_senioridade': candidato_senioridade,
        'vaga_anos_exp': vaga_anos_exp,
        'candidato_anos_exp': candidato_anos_exp,
    }

def run_bulk_analysis(_vaga_series, _applicants_processed_df, _model, _vectorizer, _candidato_tfidf_matrix):
    """Executa a análise de compatibilidade de forma otimizada."""
    results_df = _applicants_processed_df.copy()
    
    def safe_to_set(value):
        if isinstance(value, str):
            try: return set(ast.literal_eval(value))
            except (ValueError, SyntaxError): return set()
        elif hasattr(value, '__iter__'): return set(value)
        return set()

    results_df['candidate_skill_set'] = results_df['candidate_skill_set'].apply(safe_to_set)

    vaga_text = get_vaga_text(_vaga_series)
    vaga_vector = _vectorizer.transform([vaga_text])
    results_df['similaridade_texto'] = cosine_similarity(vaga_vector, _candidato_tfidf_matrix)[0]
    
    mapa_niveis = {'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Técnico': 3, 'Avançado': 4, 'Fluente': 5}
    vaga_req_ingles_num = mapa_niveis.get(str(_vaga_series.get('perfil.nivel_ingles', 'Nenhum')).strip(), 0)
    vaga_req_espanhol_num = mapa_niveis.get(str(_vaga_series.get('perfil.nivel_espanhol', 'Nenhum')).strip(), 0)
    results_df['match_nivel_ingles'] = (results_df['candidato_nivel_ingles_num'] >= vaga_req_ingles_num).astype(int)
    results_df['match_nivel_espanhol'] = (results_df['candidato_nivel_espanhol_num'] >= vaga_req_espanhol_num).astype(int)

    vaga_req_sap = 1 if _vaga_series.get('info.vaga_sap') == 'Sim' else 0
    results_df['match_sap'] = ((vaga_req_sap == 0) | ((vaga_req_sap == 1) & (results_df['candidato_tem_sap'] == 1))).astype(int)
    
    vaga_anos_exp_val = extract_years(vaga_text)
    results_df['vaga_anos_exp_val'] = vaga_anos_exp_val
    results_df['match_anos_experiencia'] = (results_df['candidato_anos_exp_val'] >= vaga_anos_exp_val).astype(int)

    vaga_nivel_prof_num = get_seniority_num(_vaga_series.get('perfil.nivel profissional', ''))
    results_df['match_nivel_profissional'] = (results_df['candidato_nivel_prof_num'] >= vaga_nivel_prof_num).astype(int)

    COMPREHENSIVE_SKILLS_LIST = [
        'python', 'java', 'javascript', 'typescript', "c#", 'c++', 'php', 'ruby', 'go', 'swift', 'kotlin', 'sql', 'pl/sql',
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
    
    if 'infos_basicas.nome' in results_df.columns:
        results_df['Nome'] = results_df['infos_basicas.nome'].fillna('Nome Indisponível')
    else:
        results_df['Nome'] = 'Nome Indisponível'
        
    results_df.rename(columns={'id_candidato': 'ID'}, inplace=True)
    
    return results_df.sort_values(by='Score', ascending=False).head(10)
