import sys
import os
import pandas as pd
import pytest

# Adiciona o diretório raiz ao path para encontrar o módulo 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importa as funções atuais que estão a ser utilizadas na aplicação
from src.app_utils import get_vaga_text, get_candidate_text, get_explanation_strings

# --- DADOS DE TESTE (MOCKS) ---
# Criamos aqui os dados "achatados", que é como eles são usados após o carregamento na aplicação.

MOCK_VAGA_FLAT = pd.Series({
    'id_vaga': 'V001',
    'info.titulo_vaga': 'Desenvolvedor Python Sênior',
    'info.vaga_sap': 'Não',
    'perfil.principais_atividades': 'Desenvolvimento de APIs REST com Django.',
    'perfil.competencia_tecnicas_e_comportamentais': 'Experiência com Python, Django e Docker.',
    'perfil.nivel_ingles': 'Avançado',
    'perfil.nivel profissional': 'Sênior'
})

MOCK_CANDIDATE_IDEAL = pd.Series({
    'id_candidato': 'C101',
    'cv_pt': 'Minhas qualificações incluem 5 anos de experiência com Python e Django.',
    'informacoes_profissionais.conhecimentos_tecnicos': 'Python, Django, Docker, APIs REST',
    'informacoes_profissionais.area_atuacao': 'Desenvolvimento Backend',
    'informacoes_profissionais.nivel_profissional': 'Sênior',
    'formacao_e_idiomas.nivel_ingles': 'Fluente',
    'candidato_tem_sap': 0 # Feature pré-calculada
})

MOCK_CANDIDATE_MISMATCH = pd.Series({
    'id_candidato': 'C102',
    'cv_pt': 'Sou um desenvolvedor júnior com foco em Javascript.',
    'informacoes_profissionais.conhecimentos_tecnicos': 'Javascript, React',
    'informacoes_profissionais.area_atuacao': 'Desenvolvimento Frontend',
    'informacoes_profissionais.nivel_profissional': 'Júnior',
    'formacao_e_idiomas.nivel_ingles': 'Básico',
    'candidato_tem_sap': 0
})

MOCK_CANDIDATE_NO_INFO = pd.Series({
    'id_candidato': 'C103',
    'cv_pt': 'Currículo sem informações de nível profissional.',
    'informacoes_profissionais.nivel_profissional': None, # Simula um campo vazio
    'formacao_e_idiomas.nivel_ingles': 'Nenhum',
    'candidato_tem_sap': 0
})


def test_get_vaga_text():
    """Verifica se a função de extração de texto da vaga funciona corretamente com dados achatados."""
    texto_esperado = "desenvolvedor python sênior desenvolvimento de apis rest com django. experiência com python, django e docker."
    texto_gerado = get_vaga_text(MOCK_VAGA_FLAT)
    assert texto_gerado == texto_esperado

def test_get_candidate_text():
    """Verifica se a função de extração de texto do candidato funciona com dados achatados."""
    texto_esperado = "minhas qualificações incluem 5 anos de experiência com python e django. python, django, docker, apis rest desenvolvimento backend "
    texto_gerado = get_candidate_text(MOCK_CANDIDATE_IDEAL)
    assert texto_gerado == texto_esperado

def test_get_explanation_strings_ideal_candidate():
    """Testa a lógica de explicação para o candidato ideal, que deve ter match em tudo."""
    explanation = get_explanation_strings(MOCK_VAGA_FLAT, MOCK_CANDIDATE_IDEAL, 5, 5)

    assert explanation['vaga_senioridade'] == 'Sênior'
    assert explanation['candidato_senioridade'] == 'Sênior'
    assert explanation['vaga_req_ingles'] == 'Avançado'
    assert explanation['candidato_tem_ingles'] == 'Fluente'

def test_get_explanation_strings_mismatch_candidate():
    """Testa a lógica de explicação para um candidato com incompatibilidade."""
    explanation = get_explanation_strings(MOCK_VAGA_FLAT, MOCK_CANDIDATE_MISMATCH, 5, 1)

    assert explanation['vaga_senioridade'] == 'Sênior'
    assert explanation['candidato_senioridade'] == 'Júnior'
    assert explanation['vaga_req_ingles'] == 'Avançado'
    assert explanation['candidato_tem_ingles'] == 'Básico'

def test_get_explanation_strings_no_info_candidate():
    """Testa se a função trata corretamente os campos vazios, retornando 'Não Informado'."""
    explanation = get_explanation_strings(MOCK_VAGA_FLAT, MOCK_CANDIDATE_NO_INFO, 5, 0)

    assert explanation['candidato_senioridade'] == 'Não Informado'
    assert explanation['candidato_tem_ingles'] == 'Nenhum'
