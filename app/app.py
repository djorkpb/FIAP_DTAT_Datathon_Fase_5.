import sys
import os
import streamlit as st
import pandas as pd
import re

# Garante que o script consegue encontrar a pasta 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app_utils import (
    load_base_data,
    get_precomputed_data,
    run_bulk_analysis, 
    get_explanation_strings,
    get_vaga_text
)

def main():
    """Função principal que executa a aplicação Streamlit."""
    
    st.set_page_config(
        page_title="Decision Match AI",
        page_icon="🎯",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { width: 450px !important; }
        .mismatch { color: red; background-color: #f0f2f2; padding: 1px 5px; border-radius: 0.25rem; font-family: monospace; font-size: 0.9em; }
        .matched-skill { color: green; background-color: #f0f2f2; padding: 1px 5px; border-radius: 0.25rem; font-family: monospace; font-size: 0.9em; }
        .extra-skill { color: #31333F; background-color: #f0f2f2; padding: 1px 5px; border-radius: 0.25rem; font-family: monospace; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title('🎯 Decision Match AI')
    st.subheader('Encontre os Melhores Talentos para Suas Vagas')

    filter_keys = ['sel_title', 'sel_empresa', 'ms_area', 'ms_divisao', 
                   'ms_skills', 'sel_sap', 'sel_ingles', 'sel_espanhol']
    
    if 'filters_initialized' not in st.session_state:
        st.session_state.filters_initialized = True
        for key in filter_keys:
            if key.startswith('ms_'): st.session_state[key] = []
            elif key == 'sel_sap': st.session_state[key] = "Indiferente"
            else: st.session_state[key] = None

    if 'selected_vaga' not in st.session_state:
        st.session_state.selected_vaga = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    with st.spinner("Carregando dados e modelos. Isso pode levar um tempo..."):
        model, vagas_df = load_base_data()
        applicants_df, vectorizer, candidato_tfidf_matrix = get_precomputed_data()

    if model is not None and vagas_df is not None and applicants_df is not None:
        
        with st.sidebar:
            st.header("🔍 Filtros Dinâmicos de Vagas")

            def on_filter_change():
                st.session_state.current_page = 1
                st.session_state.selected_vaga = None

            def clear_filters_callback():
                for key in filter_keys:
                    if key.startswith('ms_'): st.session_state[key] = []
                    elif key == 'sel_sap': st.session_state[key] = "Indiferente"
                    else: st.session_state[key] = None
                st.session_state.selected_vaga = None
                st.session_state.current_page = 1
            
            df_selection = vagas_df.copy()

            col1, col2 = st.columns(2)
            with col1:
                title_options = sorted(vagas_df['info.titulo_vaga'].dropna().unique())
                st.selectbox("Título da Vaga:", options=title_options, key='sel_title', placeholder="Selecione...", on_change=on_filter_change, index=None)
                if st.session_state.get('sel_title'):
                    df_selection = df_selection[df_selection['info.titulo_vaga'] == st.session_state.sel_title]

                clientes_options = sorted(df_selection['info.cliente'].dropna().unique())
                st.selectbox("Empresa:", options=clientes_options, key='sel_empresa', placeholder="Selecione...", on_change=on_filter_change, index=None)
                if st.session_state.get('sel_empresa'):
                    df_selection = df_selection[df_selection['info.cliente'] == st.session_state.sel_empresa]

                areas_options = sorted(df_selection['perfil.areas_atuacao'].dropna().unique())
                st.multiselect("Área de Atuação:", options=areas_options, key='ms_area', placeholder="Selecione...", on_change=on_filter_change)
                if st.session_state.get('ms_area'):
                    df_selection = df_selection[df_selection['perfil.areas_atuacao'].isin(st.session_state.ms_area)]

                divisoes_options = sorted(df_selection['info.empresa_divisao'].dropna().unique())
                st.multiselect("Divisão da Decision:", options=divisoes_options, key='ms_divisao', placeholder="Selecione...", on_change=on_filter_change)
                if st.session_state.get('ms_divisao'):
                    df_selection = df_selection[df_selection['info.empresa_divisao'].isin(st.session_state.ms_divisao)]

            with col2:
                COMPREHENSIVE_SKILLS_LIST = sorted(['python', 'java', 'sql', 'aws', 'azure', 'react', 'angular', 'power bi', 'tableau'])
                st.multiselect("Habilidades da Vaga:", options=COMPREHENSIVE_SKILLS_LIST, key='ms_skills', placeholder="Selecione...", on_change=on_filter_change)
                
                if st.session_state.get('ms_skills'):
                    for skill in st.session_state.ms_skills:
                        df_selection = df_selection[df_selection['vaga_text'].str.contains(skill, regex=False)]
                
                st.selectbox("Vaga SAP:", options=["Indiferente", "Sim", "Não"], key='sel_sap', on_change=on_filter_change)
                if st.session_state.get('sel_sap') and st.session_state.get('sel_sap') != "Indiferente":
                    df_selection = df_selection[df_selection['info.vaga_sap'] == st.session_state.sel_sap]
                
                level_map = {level: i for i, level in enumerate(['Nenhum', 'Básico', 'Intermediário', 'Técnico', 'Avançado', 'Fluente'])}
                
                niveis_ingles_options = sorted(vagas_df['perfil.nivel_ingles'].unique(), key=lambda x: level_map.get(x, 99))
                st.selectbox("Nível Mínimo de Inglês:", options=niveis_ingles_options, key='sel_ingles', placeholder="Selecione...", on_change=on_filter_change, index=None)
                
                if st.session_state.get('sel_ingles'):
                    min_level = level_map.get(st.session_state.sel_ingles, 0)
                    df_selection = df_selection[df_selection['nivel_num_ingles'] >= min_level]
                
                niveis_espanhol_options = sorted(vagas_df['perfil.nivel_espanhol'].unique(), key=lambda x: level_map.get(x, 99))
                st.selectbox("Nível Mínimo de Espanhol:", options=niveis_espanhol_options, key='sel_espanhol', placeholder="Selecione...", on_change=on_filter_change, index=None)
                
                if st.session_state.get('sel_espanhol'):
                    min_level = level_map.get(st.session_state.sel_espanhol, 0)
                    df_selection = df_selection[df_selection['nivel_num_espanhol'] >= min_level]
            
            st.button("Limpar Filtros", use_container_width=True, on_click=clear_filters_callback)
            st.session_state.filtered_vagas = df_selection
            st.info(f"**{len(df_selection)} vagas encontradas.**")

        if st.session_state.selected_vaga is not None:
            vaga_selecionada = st.session_state.selected_vaga
            
            if st.button("← Voltar à lista de vagas"):
                st.session_state.selected_vaga = None
                st.rerun()

            top_10_candidates = run_bulk_analysis(vaga_selecionada, applicants_df, model, vectorizer, candidato_tfidf_matrix)
            
            st.subheader(f"Vaga: {vaga_selecionada['info.titulo_vaga']}")
            st.markdown(f"**Empresa:** {vaga_selecionada.get('info.cliente', 'N/A')}")
            vaga_description = vaga_selecionada.get('perfil.principais_atividades', '')
            if vaga_description:
                with st.expander("Ver Descrição da Vaga"):
                    st.markdown(vaga_description)
            st.markdown("---")

            st.subheader("**🏆 Top 10 Candidatos Recomendados:**")
            
            if top_10_candidates.empty:
                st.warning("Nenhum candidato compatível encontrado.")
            else:
                for index, candidate_row in top_10_candidates.reset_index(drop=True).iterrows():
                    expander_title = f"**{index + 1}. {candidate_row['Nome']} ({candidate_row['ID']})** (Score: {candidate_row['Score']:.2%})"
                    with st.expander(expander_title):
                        candidato_selecionado = applicants_df[applicants_df['id_candidato'] == candidate_row['ID']].iloc[0]
                        
                        # --- BLOCO COM A SINTAXE CORRETA ---
                        explanation_strings = get_explanation_strings(
                            vaga_selecionada, 
                            candidato_selecionado,
                            candidate_row['vaga_anos_exp_val'],
                            candidate_row['candidato_anos_exp_val']
                        )

                        st.markdown("##### Fatores da Recomendação:")
                        st.markdown(f"- **Análise de Texto (Similaridade): `{candidate_row['similaridade_texto']:.2%}`**")
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
                            st.text_area("CV", value=cv_text, height=300, label_visibility="collapsed", key=f"cv_text_area_{candidate_row['ID']}")

        elif 'filtered_vagas' in st.session_state and st.session_state.filtered_vagas is not None:
            st.subheader("Vagas Encontradas")
            vagas_filtradas = st.session_state.filtered_vagas

            if vagas_filtradas.empty:
                st.info("Nenhuma vaga encontrada com os filtros selecionados.")
            else:
                vagas_filtradas = vagas_filtradas.copy().sort_values('info.titulo_vaga')
                
                items_per_page = 20
                total_items = len(vagas_filtradas)
                total_pages = (total_items + items_per_page - 1) // items_per_page

                if st.session_state.current_page > total_pages and total_pages > 0:
                    st.session_state.current_page = 1
                
                start_index = (st.session_state.current_page - 1) * items_per_page
                end_index = start_index + items_per_page
                vagas_para_exibir = vagas_filtradas.iloc[start_index:end_index]

                for _, vaga in vagas_para_exibir.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{vaga['info.titulo_vaga']}**")
                            st.markdown(f"<small>Empresa: {vaga.get('info.cliente', 'N/A')} | Área: {vaga.get('perfil.areas_atuacao', 'N/A')}</small>", unsafe_allow_html=True)
                        with col2:
                            if st.button("Analisar Candidatos", key=vaga['id_vaga'], use_container_width=True):
                                st.session_state.selected_vaga = vaga
                                st.rerun()
                        
                        vaga_description = vaga.get('perfil.principais_atividades', '')
                        if vaga_description:
                            with st.expander("Ver Descrição da Vaga"):
                                st.markdown(vaga_description)
                                
                    st.markdown("---")
                
                if total_pages > 1:
                    st.markdown("---")
                    p_col1, p_col2, p_col3 = st.columns([3, 1, 3])
                    with p_col1:
                        if st.session_state.current_page > 1:
                            if st.button("⬅️ Anterior", use_container_width=True):
                                st.session_state.current_page -= 1
                                st.rerun()
                    with p_col2:
                        st.write(f"Página {st.session_state.current_page} de {total_pages}")
                    with p_col3:
                        if st.session_state.current_page < total_pages:
                            if st.button("Próxima ➡️", use_container_width=True):
                                st.session_state.current_page += 1
                                st.rerun()
        else:
            st.info("👋 Bem-vindo! Utilize o painel à esquerda para filtrar as vagas.")

if __name__ == "__main__":
    main()