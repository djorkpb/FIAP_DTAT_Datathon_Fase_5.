import sys
import os
import pandas as pd
import pytest

# Adiciona o diretório raiz ao path para encontrar o módulo 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app_utils import get_vaga_text, get_candidate_text, get_explanation_data
from tests.mock_data import MOCK_VAGA, MOCK_CANDIDATES

def test_get_vaga_text():
    """Verifica se a função de extração de texto da vaga funciona corretamente."""
    vaga_series = pd.Series(MOCK_VAGA)
    texto_esperado = "desenvolvedor sap sênior desenvolvimento e manutenção de sistemas sap. otimização de processos. experiência com abap e fiori."
    texto_gerado = get_vaga_text(vaga_series)
    assert texto_gerado == texto_esperado

def test_get_candidate_text():
    """Verifica se a função de extração de texto do candidato funciona corretamente."""
    candidato_series = pd.Series(MOCK_CANDIDATES[0]) # Usa o candidato ideal
    texto_esperado = "minhas qualificações incluem vasta experiência com sap abap. abap, fiori, hana desenvolvimento sap"
    texto_gerado = get_candidate_text(candidato_series)
    assert texto_gerado == texto_esperado

def test_explanation_data_ideal_candidate():
    """Testa a lógica de explicação para o candidato ideal."""
    vaga_series = pd.Series(MOCK_VAGA)
    candidato_series = pd.Series(MOCK_CANDIDATES[0]) # Candidato Ideal

    explanation = get_explanation_data(vaga_series, candidato_series)

    assert explanation['similaridade_texto'] > 0
    assert explanation['match_ingles'] == 1
    assert explanation['match_sap'] == 1
    assert explanation['vaga_req_ingles'] == 'Avançado'
    assert explanation['candidato_tem_ingles'] == 'Fluente'

def test_explanation_data_ingles_nao_atende():
    """Testa a lógica de explicação para um candidato com inglês abaixo do requisito."""
    vaga_series = pd.Series(MOCK_VAGA)
    candidato_series = pd.Series(MOCK_CANDIDATES[2]) # Candidato com Inglês Básico

    explanation = get_explanation_data(vaga_series, candidato_series)

    assert explanation['match_ingles'] == 0
    assert explanation['vaga_req_ingles'] == 'Avançado'
    assert explanation['candidato_tem_ingles'] == 'Básico'

def test_explanation_data_sap_nao_atende():
    """Testa a lógica de explicação para um candidato que não menciona SAP."""
    vaga_series = pd.Series(MOCK_VAGA)
    candidato_series = pd.Series(MOCK_CANDIDATES[1]) # Candidato Sem SAP

    explanation = get_explanation_data(vaga_series, candidato_series)

    assert explanation['match_sap'] == 0
    assert explanation['candidato_tem_sap'] == "Não"

