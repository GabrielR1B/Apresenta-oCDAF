import warnings
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import socceraction.atomic.spadl as atomicspadl

import futmetria
from vaep import vaep

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(layout="wide", page_title="Análise de Futebol com VAEP")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
MODELS_DIR = BASE_DIR / "modelos"

EVENTS_FILE = "events_England.json"
MATCHES_FILE = "matches_England.json"
PLAYERS_FILE = "players.json"
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
        st.info("Carregando dados SPADL pré-processados do arquivo CSV...")
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
players_df_global = load_players_data()
modelos_global = load_ml_models()
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
if st.sidebar.button("👕Análise de Jogadores"):
    st.session_state.page = 'player_analysis'


if st.session_state.page == 'home':
    st.title("⚽Ciência  de Dados Aplicada ao Futebol")
    st.markdown("---")
    st.header("Estratégias de Times e Detecção de Padrões de Jogo")
    st.write("""
    A metodologia VAEP (Valuing Actions by Estimating Probabilities) é uma abordagem avançada para avaliar a contribuição de cada ação individual de um jogador de futebol para o resultado final de uma partida. Ao contrário de métricas tradicionais que apenas contam eventos (passes, desarmes, chutes), o VAEP atribui um valor a cada ação com base em como ela muda a probabilidade de o time marcar um gol ou sofrer um gol.

    **Como funciona:**

    1.  **Eventos para SPADL:** Os dados brutos de eventos do futebol (como os fornecidos pelo StatsBomb) são convertidos para o formato SPADL (Space-adjusted Player Action Data Language). Este formato padroniza as ações e suas coordenadas no campo.
    2.  **Ações Atômicas:** As ações SPADL são decompostas em "ações atômicas" mais granulares (por exemplo, um passe bem-sucedido pode ser decomposto em "recepção" e "passe").
    3.  **Modelagem de Probabilidades:** Modelos de Machine Learning (geralmente redes neurais ou modelos de regressão) são treinados para prever a probabilidade de um time marcar ou sofrer um gol a partir de qualquer estado do jogo (posição da bola, jogadores, etc.).
    4.  **Cálculo do VAEP:** Para cada ação, o valor VAEP é calculado como a diferença entre a probabilidade de gol (ou não sofrer gol) *após* a ação e a probabilidade de gol *antes* da ação.
        * Um passe que abre a defesa e aumenta drasticamente a chance de gol terá um VAEP alto e positivo.
        * Um passe errado que resulta em perda de posse e aumenta a chance do adversário marcar terá um VAEP negativo.
        * Um desarme crucial na defesa terá um VAEP positivo (reduz a chance do adversário marcar).
    5.  **Análise de Clusters:** As ações dos jogadores são agrupadas em clusters com base em suas características espaciais e contextuais. Isso permite identificar padrões de jogo e as "zonas de influência" onde os jogadores são mais eficazes em diferentes tipos de ações.
    6.  **Ranking de Jogadores:** Os jogadores são ranqueados com base na soma ou média de seus valores VAEP, ou na sua contribuição em diferentes tipos de ações e clusters, fornecendo uma visão mais profunda de sua performance.

    **Benefícios do VAEP:**

    * **Avaliação Holística:** Valoriza todas as ações, não apenas as que resultam diretamente em gols.
    * **Contextualização:** Leva em conta a situação do jogo, a posição no campo e a pressão dos adversários.
    * **Identificação de Talentos:** Ajuda a identificar jogadores que contribuem significativamente para o time, mesmo que não apareçam nas estatísticas tradicionais de gols e assistências.
    """)
    st.markdown("---")
    st.caption("Desenvolvido para análise de futebol com dados StatsBomb e métricas VAEP.")

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
    st.caption("Desenvolvido para análise de futebol com dados StatsBomb e métricas VAEP.")
