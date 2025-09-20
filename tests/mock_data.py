# Arquivo com dados falsos (mocks) para usar nos testes unitários.
# Isso torna os testes independentes dos arquivos JSON reais e muito mais rápidos.

MOCK_VAGA = {
    'informacoes_basicas': {
        'titulo_vaga': 'Desenvolvedor SAP Sênior',
        'vaga_sap': 'Sim'
    },
    'perfil_vaga': {
        'principais_atividades': 'Desenvolvimento e manutenção de sistemas SAP. Otimização de processos.',
        'competencia_tecnicas_e_comportamentais': 'Experiência com ABAP e Fiori.',
        'nivel_ingles': 'Avançado'
    }
}

MOCK_CANDIDATES = [
    # Candidato 1: Perfil ideal, deve passar no filtro e ter bom match.
    {
        'id_candidato': '101',
        'cv_pt': 'Minhas qualificações incluem vasta experiência com SAP ABAP.',
        'informacoes_profissionais': {
            'conhecimentos_tecnicos': 'ABAP, Fiori, HANA',
            'area_atuacao': 'Desenvolvimento SAP'
        },
        'formacao_e_idiomas': {
            'nivel_ingles': 'Fluente'
        },
        'infos_basicas': {'nome': 'Candidato Ideal'}
    },
    # Candidato 2: Não menciona SAP, deve ter um match_sap = 0.
    {
        'id_candidato': '102',
        'cv_pt': 'Vasta experiência em desenvolvimento. Minhas qualificações são robustas.',
        'informacoes_profissionais': {
            'conhecimentos_tecnicos': 'Java, Python, SQL',
            'area_atuacao': 'Desenvolvimento Backend'
        },
        'formacao_e_idiomas': {
            'nivel_ingles': 'Avançado'
        },
        'infos_basicas': {'nome': 'Candidato Sem SAP'}
    },
    # Candidato 3: Inglês abaixo do requisito.
    {
        'id_candidato': '103',
        'cv_pt': 'Qualificações em SAP. Experiência com ABAP.',
        'informacoes_profissionais': {
            'conhecimentos_tecnicos': 'ABAP',
            'area_atuacao': 'SAP'
        },
        'formacao_e_idiomas': {
            'nivel_ingles': 'Básico'
        },
        'infos_basicas': {'nome': 'Candidato Inglês Básico'}
    },
    # Candidato 4: CV não contém a palavra "qualificações", deve ser filtrado.
    {
        'id_candidato': '104',
        'cv_pt': 'Sou um desenvolvedor experiente.',
        'informacoes_profissionais': {
            'conhecimentos_tecnicos': 'ABAP, SAP',
            'area_atuacao': 'SAP'
        },
        'formacao_e_idiomas': {
            'nivel_ingles': 'Fluente'
        },
        'infos_basicas': {'nome': 'Candidato Sem Qualificações'}
    }
]

