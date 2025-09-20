import sys
import os
import streamlit as st
import pandas as pd

# --- CORREÇÃO DO PATH ---
# Garante que o script consegue encontrar a pasta 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FIM DA CORREÇÃO ---

from src.app_utils import (
    load_base_data,
    get_precomputed_data,
    run_bulk_analysis, 
    get_explanation_strings,
    get_vaga_text
)

def main():
    """Função principal que executa a aplicação Streamlit."""
    
    st.set_page_config(layout="wide")
    
    # --- CSS ATUALIZADO ---
    st.markdown("""
    <style>
        /* Estilo para o botão do CV */
        div[data-testid="stExpander"] div[data-testid="stExpander"] summary {
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
        }
        div[data-testid="stExpander"] div[data-testid="stExpander"] summary:hover {
            background-color: #e9ecef;
        }
        /* Classe para o texto de incompatibilidade (com tamanho de fonte padronizado) */
        .mismatch {
            color: red; /* Cor da fonte vermelha */
            background-color: #f0f2f6; /* Fundo cinza claro, como o do `code` */
            padding: 1px 5px;
            border-radius: 0.25rem;
            font-family: monospace; /* Fonte monoespaçada, como a do `code` */
            font-size: 0.9em; /* Garante o mesmo tamanho da fonte do `code` */
        }
        /* Novas classes para as habilidades */
        .matched-skill {
            color: green;
            background-color: #f0f2f6;
            padding: 1px 5px;
            border-radius: 0.25rem;
            font-family: monospace;
            font-size: 0.9em;
        }
        .extra-skill {
            color: #31333F; /* Cor de texto padrão do Streamlit */
            background-color: #f0f2f6;
            padding: 1px 5px;
            border-radius: 0.25rem;
            font-family: monospace;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)
    # --- FIM DO CSS ---
    
    st.title('🎯 Decision Match AI')
    st.subheader('Encontre os Melhores Talentos para Suas Vagas')

    # Inicializa o estado da sessão para controlar o fluxo
    if 'filtered_vagas' not in st.session_state:
        st.session_state.filtered_vagas = None
    if 'selected_vaga' not in st.session_state:
        st.session_state.selected_vaga = None

    with st.spinner("Carregando e processando os dados iniciais..."):
        model, vagas_df = load_base_data()
        applicants_df, vectorizer, candidato_tfidf_matrix = get_precomputed_data()

    if model is not None and vagas_df is not None and applicants_df is not None:
        
        # --- BARRA LATERAL COM FILTROS (REORDENADA) ---
        with st.sidebar:
            st.header("Filtros de Vagas")
            
            level_order = ['Nenhum', 'Básico', 'Intermediário', 'Técnico', 'Avançado', 'Fluente']
            level_map = {level: i for i, level in enumerate(level_order)}

            # 1. Filtro Cliente (Seleção Única)
            clientes_options = sorted(vagas_df['info.cliente'].dropna().unique())
            selected_cliente = st.selectbox("Cliente:", options=clientes_options, index=None, placeholder="Selecione...")

            # 2. Filtro Área de Atuação
            areas_options = sorted(vagas_df['perfil.areas_atuacao'].dropna().unique())
            selected_areas = st.multiselect("Área de Atuação:", options=areas_options, placeholder="Selecione...")

            # 3. Filtro Divisão da Empresa
            divisoes_options = sorted(vagas_df['info.empresa_divisao'].dropna().unique())
            selected_divisoes = st.multiselect("Divisão da Empresa:", options=divisoes_options, placeholder="Selecione...")
            
            # 4. Filtro Habilidades da Vaga
            COMPREHENSIVE_SKILLS_LIST = [
                'python', 'java', 'javascript', 'typescript', 'c#', 'c++', 'php', 'ruby', 'go', 'swift', 'kotlin', 'sql', 'pl/sql',
                'react', 'angular', 'vue', 'svelte', 'jquery', 'node.js', 'django', 'flask', 'spring', 'ruby on rails', '.net', 'laravel',
                'aws', 'azure', 'google cloud', 'gcp', 'oracle cloud', 'oci', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 
                'sql server', 'dynamodb', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ansible', 'terraform', 'ci/cd',
                'agile', 'scrum', 'kanban', 'api', 'rest', 'graphql', 'microservices', 'linux', 'unix', 'html', 'css', 'sap'
            ]
            selected_skills = st.multiselect("Habilidades da Vaga:", options=sorted(COMPREHENSIVE_SKILLS_LIST), placeholder="Selecione...")

            # 5. Filtro Vaga SAP
            selected_sap = st.selectbox("Vaga SAP:", options=["Indiferente", "Sim", "Não"], index=0)

            # 6. Filtro Nível Mínimo de Inglês
            ingles_levels = vagas_df['perfil.nivel_ingles'].dropna()
            niveis_ingles_options = sorted([lvl for lvl in ingles_levels.unique() if lvl], key=lambda x: level_map.get(x, 99))
            selected_ingles = st.selectbox("Nível Mínimo de Inglês:", options=niveis_ingles_options, index=None, placeholder="Selecione...")

            # 7. Filtro Nível Mínimo de Espanhol
            espanhol_levels = vagas_df['perfil.nivel_espanhol'].dropna()
            niveis_espanhol_options = sorted([lvl for lvl in espanhol_levels.unique() if lvl], key=lambda x: level_map.get(x, 99))
            selected_espanhol = st.selectbox("Nível Mínimo de Espanhol:", options=niveis_espanhol_options, index=None, placeholder="Selecione...")

            if st.button("Filtrar Vagas", type="primary"):
                st.session_state.selected_vaga = None
                filtered_df = vagas_df.copy()
                if selected_cliente:
                    filtered_df = filtered_df[filtered_df['info.cliente'] == selected_cliente]
                if selected_areas:
                    filtered_df = filtered_df[filtered_df['perfil.areas_atuacao'].isin(selected_areas)]
                if selected_divisoes:
                    filtered_df = filtered_df[filtered_df['info.empresa_divisao'].isin(selected_divisoes)]
                if selected_skills:
                    vaga_texts = filtered_df.apply(get_vaga_text, axis=1)
                    for skill in selected_skills:
                        if not vaga_texts.empty:
                            vaga_texts = vaga_texts[vaga_texts.str.contains(skill, regex=False)]
                    filtered_df = filtered_df.loc[vaga_texts.index]
                if selected_sap != "Indiferente":
                    filtered_df = filtered_df[filtered_df['info.vaga_sap'] == selected_sap]
                if selected_ingles:
                    min_level = level_map.get(selected_ingles, 0)
                    filtered_df['nivel_num'] = filtered_df['perfil.nivel_ingles'].map(level_map).fillna(0)
                    filtered_df = filtered_df[filtered_df['nivel_num'] >= min_level]
                if selected_espanhol:
                    min_level = level_map.get(selected_espanhol, 0)
                    filtered_df['nivel_num'] = filtered_df['perfil.nivel_espanhol'].map(level_map).fillna(0)
                    filtered_df = filtered_df[filtered_df['nivel_num'] >= min_level]
                st.session_state.filtered_vagas = filtered_df

        # --- ÁREA DE CONTEÚDO PRINCIPAL ---
        
        if st.session_state.selected_vaga is not None:
            vaga_selecionada = st.session_state.selected_vaga
            
            if st.button("← Voltar à lista de vagas"):
                st.session_state.selected_vaga = None
                st.rerun()

            top_10_candidates = run_bulk_analysis(vaga_selecionada, applicants_df, model, vectorizer, candidato_tfidf_matrix)
            
            st.subheader(f"Top 10 Candidatos para: {vaga_selecionada['info.titulo_vaga']}")

            if top_10_candidates.empty:
                st.warning("Nenhum candidato compatível encontrado.")
            else:
                for index, candidate_row in top_10_candidates.iterrows():
                    expander_title = f"**{index + 1}. {candidate_row['Nome']} ({candidate_row['ID']})** (Score: {candidate_row['Score']:.2%})"
                    with st.expander(expander_title):
                        candidato_selecionado = applicants_df[applicants_df['id_candidato'] == candidate_row['ID']].iloc[0]
                        
                        explanation_strings = get_explanation_strings(
                            vaga_selecionada, 
                            candidato_selecionado,
                            candidate_row['vaga_anos_exp_val'],
                            candidate_row['candidato_anos_exp_val']
                        )

                        st.markdown("##### Fatores da Recomendação:")
                        
                        st.markdown(f"- **Análise de Texto (Similaridade): `{candidate_row['similaridade_texto']:.2%}`**")
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Avalia a correspondência de palavras-chave entre o CV e a vaga.*", unsafe_allow_html=True)
                        
                        st.markdown(f"- **Score de Habilidades Técnicas: `{candidate_row['skills_match_score']:.2%}`**")
                        
                        vaga_skills = candidate_row['required_skills']
                        matched_skills_set = set(candidate_row['matched_skills'])
                        
                        required_skills_html_parts = []
                        for skill in vaga_skills:
                            if skill in matched_skills_set:
                                required_skills_html_parts.append(f"<span class='matched-skill'>{skill}</span>")
                            else:
                                required_skills_html_parts.append(f"<span class='mismatch'>{skill}</span>")
                        
                        required_str = ", ".join(required_skills_html_parts) if required_skills_html_parts else "Nenhuma"
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- **Habilidades da Vaga:** {required_str}", unsafe_allow_html=True)

                        candidate_skills_list = []
                        if 'matched_skills' in candidate_row and candidate_row['matched_skills']:
                            candidate_skills_list.extend([f"<span class='matched-skill'>{s}</span>" for s in candidate_row['matched_skills']])
                        if 'extra_skills' in candidate_row and candidate_row['extra_skills']:
                             candidate_skills_list.extend([f"<span class='extra-skill'>{s}</span>" for s in candidate_row['extra_skills']])
                        
                        candidate_skills_html = ", ".join(candidate_skills_list) if candidate_skills_list else "Nenhuma"
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- **Habilidades do Candidato:** {candidate_skills_html}", unsafe_allow_html=True)

                        if candidate_row['match_nivel_profissional']:
                            st.markdown(f"- ✅ **Compatibilidade de Nível Profissional: Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_senioridade']}` | Candidato possui: `{explanation_strings['candidato_senioridade']}`*", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- ❌ **Compatibilidade de Nível Profissional: Não Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_senioridade']}` | Candidato possui: <span class='mismatch'>{explanation_strings['candidato_senioridade']}</span>*", unsafe_allow_html=True)

                        if candidate_row['match_anos_experiencia']:
                            st.markdown(f"- ✅ **Compatibilidade de Anos de Experiência: Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_anos_exp']}` anos | Candidato menciona: `{explanation_strings['candidato_anos_exp']}` anos*", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- ❌ **Compatibilidade de Anos de Experiência: Não Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_anos_exp']}` anos | Candidato menciona: <span class='mismatch'>{explanation_strings['candidato_anos_exp']} anos</span>*", unsafe_allow_html=True)

                        if candidate_row['match_nivel_ingles']:
                            st.markdown(f"- ✅ **Compatibilidade de Idioma (Inglês): Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_req_ingles']}` | Candidato possui: `{explanation_strings['candidato_tem_ingles']}`*", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- ❌ **Compatibilidade de Idioma (Inglês): Não Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_req_ingles']}` | Candidato possui: <span class='mismatch'>{explanation_strings['candidato_tem_ingles']}</span>*", unsafe_allow_html=True)
                        
                        if candidate_row['match_nivel_espanhol']:
                            st.markdown(f"- ✅ **Compatibilidade de Idioma (Espanhol): Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_req_espanhol']}` | Candidato possui: `{explanation_strings['candidato_tem_espanhol']}`*", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- ❌ **Compatibilidade de Idioma (Espanhol): Não Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige: `{explanation_strings['vaga_req_espanhol']}` | Candidato possui: <span class='mismatch'>{explanation_strings['candidato_tem_espanhol']}</span>*", unsafe_allow_html=True)

                        if candidate_row['match_sap']:
                            st.markdown(f"- ✅ **Compatibilidade SAP: Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige SAP: `{explanation_strings['vaga_req_sap']}` | Candidato menciona SAP: `{explanation_strings['candidato_tem_sap']}`*", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- ❌ **Compatibilidade SAP: Não Atende**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Vaga exige SAP: `{explanation_strings['vaga_req_sap']}` | Candidato menciona SAP: <span class='mismatch'>{explanation_strings['candidato_tem_sap']}</span>*", unsafe_allow_html=True)
                        
                        with st.expander("Ver Resumo do Currículo (CV)"):
                            cv_text = candidato_selecionado.get('cv_pt', 'Currículo não disponível.')
                            st.text_area("CV", value=cv_text, height=300, label_visibility="collapsed")
        
        elif st.session_state.filtered_vagas is not None:
            st.subheader("Vagas Encontradas")
            vagas_filtradas = st.session_state.filtered_vagas
            if vagas_filtradas.empty:
                st.info("Nenhuma vaga encontrada com os filtros selecionados.")
            else:
                for _, vaga in vagas_filtradas.iterrows():
                    cols = st.columns([0.8, 0.2])
                    with cols[0]:
                        st.markdown(f"**{vaga['info.titulo_vaga']}**")
                        st.markdown(f"<small>Cliente: {vaga['info.cliente']} | Área: {vaga['perfil.areas_atuacao']}</small>", unsafe_allow_html=True)
                    with cols[1]:
                        if st.button("Analisar Candidatos", key=vaga['id_vaga']):
                            st.session_state.selected_vaga = vaga
                            st.rerun()

        else:
            st.info("Utilize o painel à esquerda para filtrar as vagas e clique em 'Filtrar Vagas' para começar.")

if __name__ == "__main__":
    main()

