# build_artifacts.py (VERSÃO CORRIGIDA E PORTÁTIL)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib
import os

# --- Definição de Caminhos ---
# Encontra o caminho absoluto para a pasta raiz do projeto, baseando-se na localização deste script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Define o caminho para a pasta de saída 'artifacts' dentro da raiz do projeto
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# URL do arquivo de candidatos
APPLICANTS_URL = "https://github.com/djorkpb/FIAP_DTAT_Datathon_Fase_5./releases/download/datathon_fase_5/applicants_cleaned.json"

def get_candidate_text(candidato_series):
    """Combina os textos de um candidato (estrutura achatada)."""
    texts_to_join = [
        str(candidato_series.get('cv_pt', '')),
        str(candidato_series.get('informacoes_profissionais.conhecimentos_tecnicos', '')),
        str(candidato_series.get('informacoes_profissionais.area_atuacao', '')),
        str(candidato_series.get('informacoes_profissionais.qualificacoes', ''))
    ]
    return ' '.join(texts_to_join).lower()

print("--- Iniciando o pré-processamento dos dados dos candidatos ---")

# 1. Carregar os dados brutos
print("1/5 - Baixando e carregando 'applicants_cleaned.json'...")
applicants_df = pd.read_json(APPLICANTS_URL, orient='index')
applicants_df.index.name = 'id_candidato'
applicants_df.reset_index(inplace=True)

# 2. Aplicar filtros iniciais
mask = applicants_df['cv_pt'].notna() & applicants_df['cv_pt'].str.contains(
    'qualificaç(?:ão|ões)', case=False, na=False, regex=True
)
applicants_df = applicants_df[mask].copy()

# 3. Engenharia de Features (a parte lenta)
print("2/5 - Realizando engenharia de features (esta parte pode demorar)...")
applicants_df['candidato_text'] = applicants_df.apply(get_candidate_text, axis=1)

mapa_niveis = {'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Técnico': 3, 'Avançado': 4, 'Fluente': 5}
if 'formacao_e_idiomas.nivel_ingles' not in applicants_df.columns:
    applicants_df['formacao_e_idiomas.nivel_ingles'] = 'Nenhum'
if 'formacao_e_idiomas.nivel_espanhol' not in applicants_df.columns:
    applicants_df['formacao_e_idiomas.nivel_espanhol'] = 'Nenhum'
applicants_df['candidato_nivel_ingles_num'] = applicants_df['formacao_e_idiomas.nivel_ingles'].fillna('Nenhum').map(mapa_niveis)
applicants_df['candidato_nivel_espanhol_num'] = applicants_df['formacao_e_idiomas.nivel_espanhol'].fillna('Nenhum').map(mapa_niveis)
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
if 'informacoes_profissionais.nivel_profissional' not in applicants_df.columns:
    applicants_df['informacoes_profissionais.nivel_profissional'] = ''
applicants_df['candidato_nivel_prof_num'] = applicants_df['informacoes_profissionais.nivel_profissional'].fillna('').apply(get_seniority_num)

COMPREHENSIVE_SKILLS_LIST = [
    'python', 'java', 'javascript', 'typescript', "c#", 'c++', 'php', 'ruby', 'go', 'swift', 'kotlin', 'sql', 'pl/sql',
    'react', 'angular', 'vue', 'svelte', 'jquery', 'node.js', 'django', 'flask', 'spring', 'ruby on rails', '.net', 'laravel',
    'aws', 'azure', 'google cloud', 'gcp', 'oracle cloud', 'oci', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 
    'sql server', 'dynamodb', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ansible', 'terraform', 'ci/cd',
    'agile', 'scrum', 'kanban', 'api', 'rest', 'graphql', 'microservices', 'linux', 'unix', 'html', 'css', 'sap'
]
applicants_df['candidate_skill_set'] = applicants_df['candidato_text'].apply(
    lambda text: {s for s in COMPREHENSIVE_SKILLS_LIST if s in text}
)

# 4. Treinar o Vetorizador (outra parte lenta)
print("3/5 - Treinando o vetorizador TF-IDF...")
vectorizer = TfidfVectorizer()
candidato_tfidf_matrix = vectorizer.fit_transform(applicants_df['candidato_text'])

# 5. Salvar os artefatos
print(f"4/5 - Salvando os artefatos processados em: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Salva o DataFrame processado em um formato rápido (Parquet)
applicants_df.to_parquet(os.path.join(OUTPUT_DIR, 'applicants_processed.parquet'))

# Salva o vetorizador e a matriz
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.joblib'))
joblib.dump(candidato_tfidf_matrix, os.path.join(OUTPUT_DIR, 'tfidf_matrix.joblib'))

print(f"5/5 - Processo concluído! Artefatos salvos na pasta '{OUTPUT_DIR}'.")
print("Próximo passo: Faça o upload dos arquivos desta pasta para o GitHub Releases.")