import warnings
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import socceraction.atomic.spadl as atomicspadl

import futmetria
from vaep import vaep
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(layout="wide", page_title="Análise de Futebol com VAEP")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
MODELS_DIR = BASE_DIR / "modelos"

EVENTS_FILE = "events_England.json"
MATCHES_FILE = "matches_England.json"
PLAYERS_FILE = "players.json"
TEAMS_FILE = "teams.json"
SPADL_CSV_FILE = "wyscount_england_events_spadl.csv"
MODELS_PKL_FILE = "modelos.pkl"

@st.cache_data(show_spinner="Carregando e processando dados de eventos...")
def load_and_process_event_data():
    events_path = DATA_DIR / EVENTS_FILE
    matches_path = DATA_DIR / MATCHES_FILE
    spadl_csv_path = DATA_DIR / SPADL_CSV_FILE

    if not events_path.is_file():
        st.error(f"Arquivo de eventos não encontrado: '{EVENTS_FILE}' em '{DATA_DIR}'.")
        st.stop()
    if not matches_path.is_file():
        st.error(f"Arquivo de partidas não encontrado: '{MATCHES_FILE}' em '{DATA_DIR}'.")
        st.stop()

    events = futmetria.load_events(events_path)
    matches = futmetria.load_matches(matches_path)

    if spadl_csv_path.is_file():
        spadl_df = pd.read_csv(spadl_csv_path)
    else:
        st.warning("Primeiro processamento: Convertendo eventos para o formato SPADL (isso pode levar um tempo)...")
        spadl_df = futmetria.spadl_transform(events, matches)
        spadl_df.to_csv(spadl_csv_path, index=False)
        st.success("Dados SPADL transformados e salvos!")

    atomic_spadl_df = atomicspadl.convert_to_atomic(spadl_df)
    actions = futmetria.gera_a(atomic_spadl_df)
    eventVaep = vaep(spadl_df)

    aVaep = actions.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    return aVaep

@st.cache_data
def get_team_vaep(team_id):
    """
    Verifica se os dados VAEP para um time já existem em um arquivo Parquet.
    - Se existir, lê e retorna o DataFrame.
    - Se não existir, processa os dados, salva em Parquet e retorna o DataFrame.
    """
    # --- CAMINHOS (Paths) ---
    output_dir = Path("vaep")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{team_id}.parquet"

    # --- PASSO 1: VERIFICAR SE O ARQUIVO JÁ EXISTE ---
    if output_path.is_file():
        aVaep = pd.read_parquet(output_path)
        st.success("Dados carregados com sucesso!")
        return aVaep

    # --- PASSO 2: SE O ARQUIVO NÃO EXISTE, PROCESSAR DO ZERO ---
    placeholder = st.empty()
    placeholder.warning(f"Dados para o time {team_id} não encontrados. Iniciando processamento completo. Por favor, aguarde.")

    # --- Carregamento de Dados ---
    events_path = DATA_DIR / EVENTS_FILE
    matches_path = DATA_DIR / MATCHES_FILE
    spadl_csv_path = DATA_DIR / SPADL_CSV_FILE

    if not events_path.is_file():
        st.error(f"Arquivo de eventos não encontrado: '{EVENTS_FILE}' em '{DATA_DIR}'.")
        st.stop()
    if not matches_path.is_file():
        st.error(f"Arquivo de partidas não encontrado: '{MATCHES_FILE}' em '{DATA_DIR}'.")
        st.stop()

    events = futmetria.load_events(events_path)
    matches = futmetria.load_matches(matches_path)
    matches_id = matches[matches['teamId'] == team_id]['matchId']
    
    if spadl_csv_path.is_file():
        spadl_df = pd.read_csv(spadl_csv_path)
    else:
        spadl_df = futmetria.spadl_transform(events, matches)
        spadl_df.to_csv(spadl_csv_path, index=False)
    
    # --- Processamento VAEP ---
    spadl_df = spadl_df[spadl_df.game_id.isin(matches_id)].reset_index(drop=True)
    
    if spadl_df.empty:
        st.error(f"Não foram encontradas ações SPADL para o time {team_id}.")
        return pd.DataFrame() 

    atomic_spadl_df = atomicspadl.convert_to_atomic(spadl_df)
    atomic_spadl_df = atomic_spadl_df[atomic_spadl_df['team_id'] == team_id]
    actions = futmetria.gera_a(atomic_spadl_df)
    
    eventVaep = vaep(spadl_df)
    aVaep = actions.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )

    # --- PASSO 3: SALVAR OS NOVOS DADOS E RETORNAR ---
    try:
        aVaep.to_parquet(output_path, index=False)
        placeholder.success(f"Processamento concluído!")
    except Exception as e:
        placeholder.error(f"Ocorreu um erro ao salvar o arquivo Parquet: {e}")

    return aVaep

@st.cache_data(show_spinner="Carregando dados de jogadores...")
def load_players_data():
    players_path = DATA_DIR / PLAYERS_FILE
    if not players_path.is_file():
        st.error(f"Arquivo de jogadores não encontrado: '{PLAYERS_FILE}' em '{DATA_DIR}'.")
        st.stop()

    players = pd.read_json(players_path)
    players = players[["wyId", "firstName", "lastName"]].copy()
    players["jogNome"] = players["firstName"] + " " + players["lastName"]
    players = players.drop(columns=["firstName", "lastName"])
    players = players.rename(columns={"wyId": "player_id"})
    players["jogNome"] = players["jogNome"].apply(
        lambda x: x.encode().decode("unicode_escape")
    )
    return players

@st.cache_data(show_spinner="Carregando dados de clubes...")
def load_teams_data():
    """Carrega os dados dos times a partir de um arquivo JSON."""
    teams_path = DATA_DIR / TEAMS_FILE
    if not teams_path.is_file():
        st.error(f"Arquivo de clubes não encontrado: '{TEAMS_FILE}' em '{DATA_DIR}'.")
        st.stop()

    teams = pd.read_json(teams_path)
    return teams

@st.cache_resource(show_spinner="Carregando modelos de VAEP...")
def load_ml_models():
    model_path = MODELS_DIR / MODELS_PKL_FILE
    if not model_path.is_file():
        st.error(f"Arquivo de modelo não encontrado: '{MODELS_PKL_FILE}' em '{MODELS_DIR}'. Certifique-se de que o modelo foi treinado e salvo.")
        st.stop()
    return futmetria.carregar_modelos(model_path)

@st.cache_data(show_spinner="Gerando rankings...")
def calculate_player_rankings(_modelos, _aVaep, _players_df):
    rankings = futmetria.get_players_ranking_for_models(_modelos, _aVaep)
    return rankings

# --- CARREGAMENTO GLOBAL DE DADOS E MODELOS ---
# Estas funções serão executadas uma vez (graças ao cache) quando o aplicativo iniciar.
aVaep_global = load_and_process_event_data()
# aVaep_time = load_vaep_time()
players_df_global = load_players_data()
modelos_global = load_ml_models()
teams_df = load_teams_data()
if modelos_global is None:
    st.error("Modelos essenciais não puderam ser carregados. Verifique os arquivos do modelo.")
    st.stop()
rankings_global = calculate_player_rankings(modelos_global, aVaep_global, players_df_global)
# --- FIM DO CARREGAMENTO GLOBAL ---


st.sidebar.title("Navegação")
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.sidebar.button("Página Inicial"):
    st.session_state.page = 'home'
if st.sidebar.button("🥅Análise de Clubes"):
    st.session_state.page = 'team_analysis'
if st.sidebar.button("👕Análise de Jogadores"):
    st.session_state.page = 'player_analysis'

### --- HOME PAGE --- ###
if st.session_state.page == 'home':
    
    # --- TÍTULO ---
    st.title("⚽ Ciência de Dados Aplicada ao Futebol")
    st.markdown("---")

    # --- INTRODUÇÃO ---
    st.header("Estratégias de Times e Detecção de Padrões de Jogo")
    st.markdown("""
    Para entender as estratégias e a eficácia de times e jogadores de futebol, é preciso ir além das estatísticas tradicionais de gols e assistências. Uma ação aparentemente simples, como um passe no meio-campo, pode ser o ponto de partida para um gol, enquanto um cruzamento aparentemente perigoso pode ser uma jogada de baixa probabilidade. 
    Nossa metodologia foi desenhada para capturar essa complexidade, combinando duas abordagens poderosas: a detecção de padrões de jogo (clusters) com o SoccerMix e a valorização de cada ação individual com o VAEP.    """)
    st.markdown("---")

    # --- PILARES DA METODOLOGIA (Em colunas para comparação) ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Valorizando Ações com VAEP")
        st.markdown("""
        **VAEP (Valuing Actions by Estimating Probabilities)** é um framework que atribui um valor numérico a cada ação com bola. Ele mede como cada passe, drible ou chute altera a probabilidade do time marcar ou sofrer um gol.
        
        - ✅ **Avaliação Holística:** Valoriza todas as ações, não só as que resultam em gol.
        - 🌎 **Análise Contextual:** O valor de um passe depende da sua localização e do momento do jogo.
        - 💎 **Identificação de Talentos:** Revela jogadores que contribuem de formas que as estatísticas tradicionais ignoram.
        """)

    with col2:
        st.subheader("Detectando Padrões com SoccerMix")
        st.markdown("""
        **SoccerMix** é um algoritmo de clusterização que analisa milhares de ações e as agrupa em "padrões de jogo" ou **ações prototípicas**. Ele nos ajuda a construir um agrupamentos de jogadas-padrão.
        
        - 🎨 **Identifica o Estilo:** Mostra se um time prefere passes curtos, cruzamentos ou lançamentos longos.
        - 🗺️ **Define Zonas de Influência:** Mapeia as áreas do campo onde certos padrões são mais comuns.
        - 📖 **Cria um "Dicionário de Jogadas":** Transforma dados brutos em táticas compreensíveis, como "construção pelo meio".
        """)
    
    st.markdown("---")

    # --- A SINERGIA ---
    st.header("Onde Valor Encontra a Tática")
    st.info("""
    Ao combinar **VAEP** e **SoccerMix**, alcançamos uma análise muito mais rica. Não apenas identificamos os **padrões** que um time usa (SoccerMix), mas também medimos a **eficácia** de cada um desses padrões (calculando o VAEP médio por cluster). Isso nos permite comparar estilos de jogo de forma objetiva e identificar quais estratégias realmente geram valor em campo.
    """)

    # --- DETALHES TÉCNICOS (Em um menu expansível) ---
    st.markdown("""
    #### Passo a passo do fluxo de geração das análises:

    1.  **Eventos para SPADL:** Os dados brutos de eventos são convertidos para o formato padronizado SPADL, que representa ações com suas coordenadas no campo.

    2.  **Ações Atômicas:** As ações SPADL são decompostas em componentes mais granulares para uma análise detalhada (ex: um passe é dividido em "recepção" e "passe").

    3.  **Análise de Clusters (SoccerMix):** As ações são agrupadas em clusters com base em suas características, identificando os padrões de jogo recorrentes.

    4.  **Modelagem de Probabilidades:** Modelos de Machine Learning são treinados para prever a probabilidade de um time marcar (`P(score)`) ou sofrer um gol (`P(concede)`) a partir de qualquer estado do jogo.

    5.  **Cálculo do VAEP:** Para cada ação, calculamos o valor como a diferença na probabilidade de gol *após* a ação e *antes* da ação.
        - 📈 **VAEP Positivo:** Ação que aumenta a chance de gol (um passe chave, um desarme crucial).
        - 📉 **VAEP Negativo:** Ação que aumenta a chance de sofrer um gol (um passe errado na defesa).
    
    6.  **Ranking e Análise:** Jogadores e times são ranqueados pela soma de seus valores VAEP, ou pela sua eficácia em diferentes clusters de ações, fornecendo uma visão completa da performance.
    """)

    st.markdown("---")
    st.caption("Desenvolvido para análise de futebol com dados Wyscout, com clusterização Soccermix e métricas VAEP.")

### --- TEAM ANALYSIS --- ###
elif st.session_state.page == 'team_analysis':
    
    st.title("⚽ Análise de Ações e Clusters por Clubes")
    st.markdown("---")

# --- PASSO 1: SELEÇÃO DO TIPO DE ANÁLISE ---
    # Usamos st.radio para apresentar as opções ao usuário
    analysis_type = st.radio(
        label="Escolha qual tipo de análise deseja fazer:",
        options=[
            "Análise de clusterização com VAEP por time",
            "Análise de clusterização com VAEP global",
            "Análise comparativa entre dois clubes",
            "Análise Z-Rank - Time x Liga"
        ],
        horizontal=True, # Deixa os botões na horizontal
        label_visibility="collapsed" # Oculta o label principal para um visual mais limpo
    )
    st.markdown("---")


    # --- PASSO 2: EXIBIR A ANÁLISE SELECIONADA ---

    ### --- OPÇÃO 1: Análise de clusterização com VAEP global ---
    if analysis_type == "Análise de clusterização com VAEP global":
        
        st.subheader("Análise de Ações e Clusters por Clube")
        st.write("Selecione um clube e um tipo de ação para visualizar os padrões de jogo e sua eficácia (VAEP).")

        # Usar colunas para organizar os dropdowns
        col1, col2 = st.columns(2)

        with col1:
            # Dropdown de Clubes
            unique_team_ids_in_data = aVaep_global['team_id'].unique()
            relevant_teams = teams_df[teams_df['wyId'].isin(unique_team_ids_in_data)]
            team_names_for_dropdown = relevant_teams['name'].sort_values().tolist()
            selected_team_name = st.selectbox(
                label="Selecione um clube:",
                options=team_names_for_dropdown
            )

        with col2:
            # Dropdown de Ações (Clusters)
            all_model_names = [model.name for model in modelos_global if hasattr(model, 'name')]
            all_model_names.sort()
            selected_action_name = st.selectbox(
                label="Selecione uma ação:",
                options=all_model_names
            )
        
        # Botão para acionar a análise
        if st.button("Gerar Análise", type="primary"):
            if selected_team_name and selected_action_name:
                with st.spinner("Processando e gerando o gráfico..."):
                    
                    # Lógica para obter dados e modelo
                    selected_team_id = relevant_teams[relevant_teams['name'] == selected_team_name]['wyId'].iloc[0]
                    model_selected_action_list = [modelo for modelo in modelos_global if modelo.name == selected_action_name]
                    
                    if model_selected_action_list:
                        model_selected_action = model_selected_action_list[0]
                        
                        # Chamar a função de plotagem
                        plotagem = futmetria.plot(modelos=[model_selected_action], a=aVaep_global)
                        st.pyplot(plotagem)
                    else:
                        st.error(f"Não foi possível encontrar o modelo para a ação '{selected_action_name}'.")


    #### --- OPÇÃO 2: Análise de clusterização com VAEP por time ---
    elif analysis_type == "Análise de clusterização com VAEP por time":
        st.subheader("Análise de Ações e Clusters por Clube")
        st.write("Selecione um clube e um tipo de ação para visualizar os padrões de jogo e sua eficácia (VAEP).")

        # Usar colunas para organizar os dropdowns
        col1, col2 = st.columns(2)

        with col1:
            # Dropdown de Clubes
            unique_team_ids_in_data = aVaep_global['team_id'].unique()
            relevant_teams = teams_df[teams_df['wyId'].isin(unique_team_ids_in_data)]
            team_names_for_dropdown = relevant_teams['name'].sort_values().tolist()
            selected_team_name = st.selectbox(
                label="Selecione um clube:",
                options=team_names_for_dropdown
            )

        with col2:
            # Dropdown de Ações (Clusters)
            all_model_names = [model.name for model in modelos_global if hasattr(model, 'name')]
            all_model_names.sort()
            selected_action_name = st.selectbox(
                label="Selecione uma ação:",
                options=all_model_names
            )
        
        # Botão para acionar a análise
        if st.button("Gerar Análise", type="primary"):
            if selected_team_name and selected_action_name:
                    
                # Lógica para obter dados e modelo
                selected_team_id = relevant_teams[relevant_teams['name'] == selected_team_name]['wyId'].iloc[0]
                vaep_selected_team = get_team_vaep(selected_team_id)
                model_selected_action_list = [modelo for modelo in modelos_global if modelo.name == selected_action_name]
                
                if model_selected_action_list:
                    model_selected_action = model_selected_action_list[0]
                    
                    # Chamar a função de plotagem
                    plotagem = futmetria.plot(modelos=[model_selected_action], a=vaep_selected_team)
                    st.pyplot(plotagem)
                else:
                    st.error(f"Não foi possível encontrar o modelo para a ação '{selected_action_name}'.")

    ### --- OPÇÃO 3: Análise comparativa entre dois clubes ---
    elif analysis_type == "Análise comparativa entre dois clubes":
        st.subheader("Análise Comparativa entre Clubes")
        st.info("Esta funcionalidade está em desenvolvimento. Aqui você poderá selecionar dois times e comparar seus estilos de jogo e eficácia.")
        # Aqui você adicionaria o código para esta análise no futuro
    
    ### --- OPÇÃO 4: Análise Z-Rank - Time x Liga
    elif analysis_type == "Análise Z-Rank - Time x Liga":
        st.subheader("Análise Comparativa Z-Rank - Time x Liga")
        st.write("Selecione um clube e um tipo de ação para visualizar o quão bom ele é comparado com os demais times da liga")

        # Usar colunas para organizar os dropdowns
        col1, col2 = st.columns(2)

        with col1:
            # Dropdown de Clubes
            unique_team_ids_in_data = aVaep_global['team_id'].unique()
            relevant_teams = teams_df[teams_df['wyId'].isin(unique_team_ids_in_data)]
            team_names_for_dropdown = relevant_teams['name'].sort_values().tolist()
            selected_team_name = st.selectbox(
                label="Selecione um clube:",
                options=team_names_for_dropdown
            )

        with col2:
            # Dropdown de Ações (Clusters)
            all_model_names = [model.name for model in modelos_global if hasattr(model, 'name')]
            all_model_names.sort()
            selected_action_name = st.selectbox(
                label="Selecione uma ação:",
                options=all_model_names
            )
        
        # Botão para acionar a análise
        if st.button("Gerar Análise", type="primary"):
            if selected_team_name and selected_action_name:
                with st.spinner("Processando e gerando o gráfico..."):
                    
                    # Lógica para obter dados e modelo
                    selected_team_id = relevant_teams[relevant_teams['name'] == selected_team_name]['wyId'].iloc[0]
                    model_selected_action_list = [modelo for modelo in modelos_global if modelo.name == selected_action_name]
                    
                    if model_selected_action_list:
                        model_selected_action = model_selected_action_list[0]
                        
                        # Chamar a função de plotagem
                        plotagem = futmetria.plot_z_rank(modelos=[model_selected_action], a=aVaep_global, time_id=selected_team_id)
                        st.pyplot(plotagem)
                    else:
                        st.error(f"Não foi possível encontrar o modelo para a ação '{selected_action_name}'.")
        
    
    st.markdown("---")
    st.caption("Desenvolvido para análise de futebol com dados Wyscout, com clusterização soccermix e métricas VAEP.")

### --- PLAYER ANALYSIS --- ###
elif st.session_state.page == 'player_analysis':
    st.title("⚽ Análise de Ações e Ranking de Jogadores")
    st.markdown("---")

    st.header("📊 Top Jogadores por VAEP")

    percentile_threshold = st.sidebar.slider(
        "Selecione o Percentil para Filtrar:",
        min_value=0.01,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Define o limiar de VAEP para considerar os 'melhores' jogadores."
    )

    overall_best_players_display = futmetria.rank_players_overall(
        rankings_global, # Usando a variável global
        modelos_global, # Usando a variável global
        percentile=percentile_threshold,
    )
    player_ranking_df = futmetria.create_player_ranking_df(
        overall_best_players_display,
        players_df_global, # Usando a variável global
    )

    if not player_ranking_df.empty:
        columns_to_display = [
            "rank",
            "jogNome",
            "player_id",
            "clusters_count",
            "clusters_percentage",
        ]

        display_df = player_ranking_df.copy()
        valid_columns_to_display = [col for col in columns_to_display if col in display_df.columns]
        display_df = display_df[valid_columns_to_display]

        st.dataframe(display_df, use_container_width=True)

        st.markdown(f"**Total de jogadores no ranking (acima do percentil {percentile_threshold*100:.0f}%):** {len(player_ranking_df)}")

        st.markdown("---")

        st.subheader("🔍 Detalhes de Jogador Individual")

        player_names = ["Selecione um jogador..."] + sorted(player_ranking_df["jogNome"].tolist())
        selected_player_name = st.selectbox("Escolha um jogador para ver os detalhes:", player_names)

        if selected_player_name != "Selecione um jogador...":
            selected_player_row = player_ranking_df.loc[player_ranking_df["jogNome"] == selected_player_name].iloc[0]

            st.write(f"### Detalhes de {selected_player_name}")
            st.write(f"**ID do Jogador:** {selected_player_row['player_id']}")
            st.write(f"**Rank:** {selected_player_row['rank']}")
            st.write(f"**Contagem de Clusters:** {selected_player_row['clusters_count']}")
            st.write(f"**Porcentagem de Clusters:** {selected_player_row['clusters_percentage']:.2f}%")

            st.subheader(f"📈 Contribuição de Ações de {selected_player_name} por Tipo")

            all_model_names = [model.name for model in modelos_global if hasattr(model, 'name')]

            available_actions_for_player = []
            if isinstance(selected_player_row.get("clusters"), dict):
                for model_name in all_model_names:
                    if model_name in selected_player_row["clusters"]:
                        if selected_player_row["clusters"][model_name]:
                            available_actions_for_player.append(model_name)

            available_actions_for_player = sorted(available_actions_for_player)

            default_selection = ["receival"] if "receival" in available_actions_for_player else []

            if available_actions_for_player:
                selected_action_types = st.multiselect(
                    "Selecione o(s) tipo(s) de ação para plotar:",
                    options=available_actions_for_player,
                    default=default_selection,
                    help="Cada tipo de ação corresponde a um modelo e será plotado em um gráfico separado."
                )

                if selected_action_types:
                    player_figures = futmetria.plot_player_rankings(
                        selected_player_row,
                        modelos_global,
                        num_players=len(player_ranking_df),
                        wanted_actions=selected_action_types,
                    )

                    if player_figures:
                        for action_type, fig_plot in player_figures.items():
                            st.write(f"#### Gráfico para: {action_type}")
                            st.pyplot(fig_plot)
                            plt.close(fig_plot)
                    else:
                        st.warning("Não foi possível gerar gráficos para os tipos de ação selecionados com este jogador.")
                else:
                    st.info("Por favor, selecione um ou mais tipos de ação para visualizar o gráfico.")
            else:
                st.info("Este jogador não possui dados para gerar gráficos de ações ou os dados estão vazios para todos os tipos de ação.")
                st.warning("Experimente selecionar outro jogador.")

        else:
            st.info("Selecione um jogador na caixa acima para ver seus detalhes e contribuição de ações.")

    else:
        st.warning("Nenhum jogador encontrado para o percentil selecionado. Tente ajustar o percentil na barra lateral.")
    
    st.markdown("---")
    st.caption("Desenvolvido para análise de futebol com dados Wyscout, com clusterização soccermix e métricas VAEP.")

