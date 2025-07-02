import pickle
from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.atomic.vaep.features as fs
import socceraction.spadl as spadl
import socceraction.vaep.labels as lab
import tqdm
from dask.distributed import Client
from matplotlib.patheffects import withStroke
from mplsoccer import Pitch
from socceraction.vaep import features as ft

import mixture as mix
from vaep import vaep
from vis import add_ellips


def simplify(actions):
    a = actions.fillna(0)

    corner_idx = a.type_name.str.contains("corner")
    a["type_name"] = a["type_name"].mask(corner_idx, "corner")

    freekick_idx = a.type_name.str.contains("freekick")
    a["type_name"] = a["type_name"].mask(freekick_idx, "freekick")

    keeper_idx = a.type_name.str.contains("keeper")
    a["type_name"] = a["type_name"].mask(keeper_idx, "keeper_catch")

    a["x"] = a.x.mask(a.type_name == "goalkick", 5)
    a["y"] = a.y.mask(a.type_name == "goalkick", 32.5)

    a["x"] = a.x.mask(a.type_name == "goal", 105)
    a["y"] = a.y.mask(a.type_name == "goal", 34)
    a["x"] = a.x.mask(a.type_name == "shot_penalty", 94.5)
    a["y"] = a.y.mask(a.type_name == "shot_penalty", 34)

    def fix_out_y(y):
        return 68 if 68 - y < y else 0

    a["y"] = a.y.mask(a.type_name == "out", a.y.apply(fix_out_y))

    return a


def add_noise(a):
    a["x"] = a.x + np.random.normal(0, 1, len(a))
    a["y"] = a.y + np.random.normal(0, 1, len(a))
    # a["dx"] = a.dx + (a.dx != 0).apply(int) * np.random.normal(0,1,len(a))
    # a["dy"] = a.dy + (a.dy != 0).apply(int) * np.random.normal(0,1,len(a))
    a["dx"] = a.dx + np.random.normal(0, 1, len(a))
    a["dy"] = a.dy + np.random.normal(0, 1, len(a))
    return a


def mirrordouble(a):
    mirror_a = a.copy()
    mirror_a["y"] = 68 - mirror_a["y"]
    mirror_a["dy"] = -mirror_a["dy"]
    return pd.concat([a, mirror_a], axis=0)


def learn_mixture(X, W, experiment):
    model = mix.MixtureModel(
        n_components=experiment["n_components"], distribution=experiment["distribution"]
    )
    model = model.fit(X, weights=W[experiment["name"]].values)
    if model:
        model.name = experiment["name"]
    return model


def learn_mixture_models(X, W, experiments, distribution=mix.Gauss, verbose=True):
    models = []
    if verbose:
        experiments = tqdm.tqdm(experiments)
    for experiment in experiments:
        model = learn_mixture(X, W, experiment)
        if model:
            models.append(model)
    return models


def learn_mixture_models_dask(X, W, experiments, host="134.58.41.55:8786"):
    with Client(host) as client:
        client.upload_file("../mixture.py")
        X_fut = client.scatter(X, broadcast=True)
        W_fut = client.scatter(W, broadcast=True)

        futures = []
        for experiment in experiments:
            key = f"{experiment['name']}_{experiment['n_components']}"
            future = client.submit(learn_mixture, X_fut, W_fut, experiment, key=key)
            futures.append(future)
        models = client.gather(futures)
        return list(m for m in models if m)


def inspect_strategies(candidates, max_components=10):
    for strategy in [
        mix.ilp_select_models_bic,
        mix.select_models_solo_bic,
        mix.ilp_select_models_bic_triangle,
        lambda can: mix.ilp_select_models_max(can, max_components),
    ]:
        models = strategy(candidates)
        print("total: ", sum(m.n_components for m in models))
        print(
            {m.name: m.n_components for m in strategy(candidates) if m.n_components > 1}
        )


def gera_a(eventos):
    a = eventos.copy()
    a = a.merge(atomicspadl.actiontypes_df(), how="left").merge(
        spadl.bodyparts_df(), how="left"
    )
    a = simplify(a)
    a = add_noise(a)
    a = pd.concat([a, fs.movement_polar(a)], axis=1)  # add polar movement direction

    return a


def gera_cat_pesos(a):
    cat_model = mix.CategoricalModel()
    cat_model.fit(a["type_name"])
    cat_weights = cat_model.predict_proba(a["type_name"])
    return cat_weights


def gera_exp(cat_weights):
    experiments = list(
        [
            dict(name=name, n_components=i, distribution=mix.MultiGauss)
            for name in set(cat_weights.columns)
            for i in range(1, 10)
        ]
    )
    experiments += list(
        [
            dict(name=name, n_components=i, distribution=mix.MultiGauss)
            for name in ["pass", "dribble", "receival"]
            for i in range(10, 30)
        ]
    )

    return experiments


# ____________________________________________________________________plot__________________________________________________________________________________#
#                                                                                                                                                          #
#                                                                                                                                                          #
#                                                                                                                                                          #
#                                                                                                                                                          #
#                                                                                                                                                          #
# __________________________________________________________________________________________________________________________________________________________#


def plot(modelos, a):
    rem = [
        "red_card",
        "keeper_catch",
        "out",
        "goalkick",
        "yellow_card",
        "foul",
        "freekick",
    ]
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # VAEP médio e contagem de ações por cluster
        cluster_vaep = evento.groupby("cluster")["vaep_value"].mean()
        cluster_counts = evento["cluster"].value_counts()

        # Normalizadores
        vaep_min, vaep_max = cluster_vaep.min(), cluster_vaep.max()
        vaep_norm = mcolors.Normalize(vmin=vaep_min, vmax=vaep_max)
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        cmap = plt.get_cmap("RdYlGn")  # vermelho → amarelo → verde

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(
            f"Setas e elipses coloridas por VAEP médio - {m.name}", fontsize=16
        )

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # Cor baseada no VAEP médio
            vaep_avg = cluster_vaep[cluster_id]
            color = cmap(vaep_norm(vaep_avg))

            # Largura da seta baseada na quantidade de ações
            count = cluster_counts[cluster_id]
            lw = 1 + 5 * count_norm(count)  # escala 1 a 6

            # Médias de posição
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5  # Quanto você quer esticar (em metros no campo)
            x_end_stretched = x_end + stretch_factor * dx / norm
            y_end_stretched = y_end + stretch_factor * dy / norm

            # Contorno preto (ligeiramente mais grosso)
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida por cima
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

        ### --- ADICIONANDO A LEGENDA DE CORES (A MUDANÇA PRINCIPAL) --- ###
        sm = cm.ScalarMappable(cmap=cmap, norm=vaep_norm)
        sm.set_array([])

        # Adicionar a barra de legenda à figura
        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.04
        )

        # Definir o rótulo da barra de cores
        cbar.set_label(
            "Qualidade da Ação dado pela métrica VAEP\nVerde = Positivo / Vermelho = Negativo",
            fontsize=12,
            labelpad=15,
        )

        return fig


def plot_z(modelos, a, time_id, rem):
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"Z-Score VAEP por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # Cálculo do z-score para cada time nesse cluster
            team_means = cluster_data.groupby("team_id")["vaep_value"].mean()
            mean_all = team_means.mean()
            std_all = team_means.std()

            if std_all == 0 or time_id not in team_means:
                z = 0
            else:
                z = (team_means[time_id] - mean_all) / std_all

            # Normalizador específico para esse cluster
            z_min = (team_means - mean_all).min() / std_all if std_all > 0 else -1
            z_max = (team_means - mean_all).max() / std_all if std_all > 0 else 1
            zscore_norm = mcolors.TwoSlopeNorm(vmin=z_min, vcenter=0, vmax=z_max)

            cmap = cm.get_cmap("RdYlGn")
            color = cmap(zscore_norm(z))

            # Largura da seta
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida por cima
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

        plt.show()


def plot_z_rank(modelos, a, time_id):
    rem = [
        "red_card",
        "keeper_catch",
        "out",
        "goalkick",
        "yellow_card",
        "foul",
        "freekick",
    ]
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"Ranking VAEP por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # VAEP médio por time no cluster
            team_means = (
                cluster_data.groupby("team_id")["vaep_value"]
                .mean()
                .sort_values(ascending=False)
            )
            teams = team_means.index.tolist()

            if time_id in teams:
                rank = teams.index(time_id) + 1  # posição no ranking (1 = melhor)
            else:
                rank = None  # time não presente nesse cluster

            num_teams = len(teams)

            # Normalizador com melhor time em verde e pior em vermelho
            rank_norm = mcolors.Normalize(vmin=1, vmax=num_teams)
            cmap = plt.get_cmap("RdYlGn_r")

            color = (
                cmap(rank_norm(rank)) if rank else (0.5, 0.5, 0.5, 1.0)
            )  # cinza se time ausente

            # Largura da seta baseada em quantidade
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

            # Texto com a posição do time no ranking
            if rank:
                ax.text(
                    gauss.mean[0],
                    gauss.mean[1],
                    f"{rank}",
                    color="white",
                    fontsize=12,
                    ha="center",
                    va="center",
                    path_effects=[withStroke(linewidth=3, foreground="black")],
                )

        plt.show()


def plot_z_jog(modelos, a, jog_id, rem):
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"Z-Score VAEP por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # VAEP médio por jogador no cluster
            jog_mean = cluster_data.groupby("player_id")["vaep_value"].mean()
            mean_all = jog_mean.mean()
            std_all = jog_mean.std()

            if std_all == 0 or jog_id not in jog_mean or pd.isna(jog_mean[jog_id]):
                z = 0
            else:
                z = (jog_mean[jog_id] - mean_all) / std_all

            # Normalizador seguro
            z_min = ((jog_mean - mean_all).min() / std_all) if std_all > 0 else -1
            z_max = ((jog_mean - mean_all).max() / std_all) if std_all > 0 else 1
            if z_min > z_max:
                z_min, z_max = -1, 1

            # Colormap com cinza no centro
            cmap = cm.get_cmap("RdYlGn")
            zscore_norm = mcolors.TwoSlopeNorm(vmin=z_min, vcenter=0, vmax=z_max)
            color = cmap(zscore_norm(z))

            # Largura da seta
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida por cima
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

        plt.show()


def plot_z_rank_jog(modelos, a, jog_id, rem):
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"Ranking VAEP por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # VAEP médio por jogador no cluster
            jog_means = (
                cluster_data.groupby("player_id")["vaep_value"]
                .mean()
                .sort_values(ascending=False)
            )
            jogs = jog_means.index.tolist()

            if jog_id in jogs:
                rank = jogs.index(jog_id) + 1  # posição no ranking (1 = melhor)
            else:
                rank = None  # jogador não presente nesse cluster

            num_jogs = len(jogs)

            # Normalizador com melhor jogador em verde e pior em vermelho
            rank_norm = mcolors.Normalize(vmin=1, vmax=num_jogs)
            cmap = cm.get_cmap("RdYlGn_r")

            color = (
                cmap(rank_norm(rank)) if rank else (0.5, 0.5, 0.5, 1.0)
            )  # cinza se jogador ausente

            # Largura da seta baseada em quantidade
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

            # Texto com a posição do time no ranking
            if rank:
                ax.text(
                    gauss.mean[0],
                    gauss.mean[1],
                    f"{rank}",
                    color="white",
                    fontsize=12,
                    ha="center",
                    va="center",
                    path_effects=[withStroke(linewidth=3, foreground="black")],
                )

        plt.show()


def plot_xM(modelos, a, jog_id, rem):
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"% de VAEP vs Média por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # VAEP médio do cluster
            mean_cluster = cluster_data["vaep_value"].mean()

            # VAEP médio do jogador
            jog_cluster_data = cluster_data[cluster_data["player_id"] == jog_id]
            mean_jogador = jog_cluster_data["vaep_value"].mean()

            if pd.isna(mean_jogador) or mean_cluster == 0:
                perc = 0
            else:
                perc = ((mean_jogador / mean_cluster) - 1) * 100

            # Normalizador de percentuais para colormap
            perc_min = -100
            perc_max = 100
            perc_norm = mcolors.TwoSlopeNorm(vmin=perc_min, vcenter=0, vmax=perc_max)

            # Colormap: vermelho para abaixo da média, verde acima
            cmap = cm.get_cmap("RdYlGn")
            color = cmap(perc_norm(perc))

            # Largura da seta
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida por cima
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

            # Texto do valor percentual no cluster (arredondado)
            ax.text(
                gauss.mean[0],
                gauss.mean[1],
                f"{perc:.1f}%",
                color="white",
                fontsize=11,
                ha="center",
                va="center",
                path_effects=[withStroke(linewidth=3, foreground="black")],
            )

        plt.show()


def plot_xM_rank(modelos, a, jog_id, rem):
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"Ranking VAEP % por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # VAEP médio geral do cluster
            mean_cluster = cluster_data["vaep_value"].mean()

            # VAEP médio de todos os jogadores no cluster
            jog_means = cluster_data.groupby("player_id")["vaep_value"].mean()
            perc_players = ((jog_means / mean_cluster) - 1) * 100

            # Ranking dos jogadores (1 = melhor, mais acima da média)
            perc_sorted = perc_players.sort_values(ascending=False)
            jogadores = perc_sorted.index.tolist()

            if jog_id in jogadores:
                rank = jogadores.index(jog_id) + 1
                perc = perc_players[jog_id]
            else:
                rank = None
                perc = 0

            # Normalizador de ranks para colormap
            num_players = len(jogadores)
            rank_norm = mcolors.Normalize(vmin=1, vmax=num_players)
            cmap = cm.get_cmap("RdYlGn_r")

            color = cmap(rank_norm(rank)) if rank else (0.5, 0.5, 0.5, 1.0)

            # Largura da seta
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida por cima
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

            # Texto com a posição no ranking
            if rank:
                ax.text(
                    gauss.mean[0],
                    gauss.mean[1],
                    f"{rank}",
                    color="white",
                    fontsize=11,
                    ha="center",
                    va="center",
                    path_effects=[withStroke(linewidth=3, foreground="black")],
                )

        plt.show()


def plot_xM_rank_jogs(modelos, a, jogs, team_id, rem):  # falhas a se concertar
    for m in modelos:
        if m.name in rem:
            continue
        evento = a[a.type_name == m.name].copy()
        evento["x_end"] = evento["x"] + evento["dx"]
        evento["y_end"] = evento["y"] + evento["dy"]

        # Previsão dos clusters
        pos = evento[["x", "y"]]
        probs = m.predict_proba(pos)
        evento["cluster"] = probs.argmax(axis=1)

        # Contagem de ações por cluster
        cluster_counts = evento["cluster"].value_counts()
        count_norm = mcolors.Normalize(
            vmin=cluster_counts.min(), vmax=cluster_counts.max()
        )

        # Campo
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        fig, ax = pitch.draw(figsize=(12, 8))
        ax.set_title(f"Melhor Jogador por Cluster - {m.name}", fontsize=16)

        for cluster_id in sorted(evento["cluster"].unique()):
            cluster_data = evento[evento["cluster"] == cluster_id]

            # VAEP médio por jogador e por time
            jog_means = cluster_data.groupby("player_id")["vaep_value"].mean().dropna()
            team_cluster_data = cluster_data[cluster_data["team_id"] == team_id]
            mean_team = team_cluster_data["vaep_value"].mean()

            # Evita divisão por zero
            if pd.isna(mean_team) or mean_team == 0 or jog_means.empty:
                best_id = None
                color = (0.5, 0.5, 0.5, 1.0)  # cinza neutro
            else:
                # Melhor jogador do cluster
                perc_players = ((jog_means / mean_team) - 1) * 100
                best_id = perc_players.idxmax()
                max_diff = perc_players.max()
                min_diff = perc_players.min()

                # Normalizador e cor
                diff_norm = mcolors.Normalize(vmin=min_diff, vmax=max_diff)
                cmap = cm.get_cmap("RdYlGn")
                color = cmap(diff_norm(max_diff))

            # Largura da seta
            count = cluster_counts.get(cluster_id, 0)
            lw = 1 + 5 * count_norm(count)

            # Coordenadas médias
            x_start = cluster_data["x"].mean()
            y_start = cluster_data["y"].mean()
            x_end = cluster_data["x_end"].mean()
            y_end = cluster_data["y_end"].mean()

            dx = x_end - x_start
            dy = y_end - y_start
            norm_vec = np.sqrt(dx**2 + dy**2)
            stretch_factor = 0.5
            x_end_stretched = x_end + stretch_factor * dx / norm_vec
            y_end_stretched = y_end + stretch_factor * dy / norm_vec

            # Contorno preto
            ax.annotate(
                "",
                xy=(x_end_stretched, y_end_stretched),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw + 3),
            )

            # Seta colorida
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )

            # Elipse
            gauss = m.submodels[cluster_id]
            add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=1)

            # Nome do jogador
            if best_id is not None and best_id in jogs.player_id.values:
                nome = jogs.loc[jogs.player_id == best_id, "jogNome"].values[0]
            else:
                nome = "Desconhecido"

            # Texto no centro do cluster
            ax.text(
                gauss.mean[0],
                gauss.mean[1],
                nome,
                color="white",
                fontsize=10,
                ha="center",
                va="center",
                path_effects=[withStroke(linewidth=3, foreground="black")],
            )

        plt.show()


# ____________________________________________________________________main__________________________________________________________________________________#
#                                                                                                                                                          #
#                                                                                                                                                          #
#                                                                                                                                                          #
#                                                                                                                                                          #
#                                                                                                                                                          #
# __________________________________________________________________________________________________________________________________________________________#


def time_analize(spadl_df, timeId, rem=[]):

    atomic = atomicspadl.convert_to_atomic(spadl_df)
    eventos = atomic[atomic.team_id == timeId]

    a = gera_a(eventos)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = a.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))
    plot(modelos=modelos, a=aVaep, rem=rem)


def time_analize_vaep_time(spadl_df, timeId, matches, rem=[]):

    partidas_time = []

    for _, partida in matches.iterrows():
        if partida.teamId == timeId:
            partidas_time.append(partida.matchId)

    time_spadl = spadl_df[spadl_df.game_id.isin(partidas_time)].reset_index(drop=True)

    atomic = atomicspadl.convert_to_atomic(spadl_df)
    eventos = atomic[atomic.team_id == timeId]

    a = gera_a(eventos)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(time_spadl)
    aVaep = a.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))
    plot(modelos=modelos, a=aVaep, rem=rem)


def filter_last_thirds_events(
    spadl_df, n_last_thirds=2, pitch_length=105, is_atomic=False
):
    """
    Filtra os eventos para manter apenas eventos que ocorreram nos
    últimos terços do campo.
    Parâmetros:
        spadl_df (DataFrame): DataFrame com os eventos do SPADL.
        n_last_third (int): Número de terços finais a serem mantidos.
    """

    # Calcula a largura de cada terço
    third_length = pitch_length / 3

    # Define os limites dos terços finais
    lower_bound = pitch_length - (n_last_thirds * third_length)
    upper_bound = pitch_length

    # Filtra os eventos que estão dentro dos últimos terços
    if is_atomic:
        filtered_events = spadl_df[
            (spadl_df.start_x >= lower_bound) & (spadl_df.start_x <= upper_bound)
        ]
    else:
        filtered_events = spadl_df[
            (spadl_df.start_x >= lower_bound) & (spadl_df.start_x <= upper_bound)
        ]

    return filtered_events.reset_index(drop=True)


def plot_events(events, pitch=None):
    if pitch is None:
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    pitch.draw(ax=ax)

    for _, event in events.iterrows():
        x, y = event["start_x"], event["start_y"]
        pitch.scatter(x, y, ax=ax, s=50, color="blue", label=event["type_name"])

    plt.title(f"Analyzing events")
    plt.show()


def time_analize_z(spadl_df, timeId, rem=[], filter_last_thirds=0, plot_actions=False):

    if filter_last_thirds > 0:
        eventos = filter_last_thirds_events(spadl_df, n_last_thirds=filter_last_thirds)

    if plot_actions:
        plot_events(eventos)

    eventos = atomicspadl.convert_to_atomic(spadl_df)

    time_df = eventos[eventos.team_id == timeId]

    simples_eventos = gera_a(eventos=eventos)

    a = gera_a(time_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = simples_eventos.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_z(modelos=modelos, a=aVaep, time_id=timeId, rem=rem)


def time_analize_z_rank(spadl_df, timeId, rem=[]):

    eventos = atomicspadl.convert_to_atomic(spadl_df)
    time_df = eventos[eventos.team_id == timeId]

    simples_eventos = gera_a(eventos=eventos)

    a = gera_a(time_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = simples_eventos.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_z_rank(modelos=modelos, a=aVaep, time_id=timeId, rem=rem)


def jog_analize(spadl_df, jogId, rem=[]):

    atomic = atomicspadl.convert_to_atomic(spadl_df)
    eventos = atomic[atomic.player_id == jogId]

    a = gera_a(eventos)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = a.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))
    plot(modelos=modelos, a=aVaep, rem=rem)


def jog_analize_vaep_time(spadl_df, timeId, jogId, matches, rem=[]):

    partidas_time = []

    for _, partida in matches.iterrows():
        if partida.teamId == timeId:
            partidas_time.append(partida.matchId)

    time_spadl = spadl_df[spadl_df.game_id.isin(partidas_time)].reset_index(drop=True)

    atomic = atomicspadl.convert_to_atomic(spadl_df)
    eventos = atomic[atomic.player_id == jogId]

    a = gera_a(eventos)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(time_spadl)
    aVaep = a.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))
    plot(modelos=modelos, a=aVaep, rem=rem)


def jog_analize_z(spadl_df, jogId, rem=[]):

    eventos = atomicspadl.convert_to_atomic(spadl_df)
    jog_df = eventos[eventos.player_id == jogId]

    simples_eventos = gera_a(eventos=eventos)

    a = gera_a(jog_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = simples_eventos.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_z_jog(modelos=modelos, a=aVaep, jog_id=jogId, rem=rem)


def jog_analise_z_rank(spadl_df, jogId, rem=[]):
    eventos = atomicspadl.convert_to_atomic(spadl_df)
    jog_df = eventos[eventos.player_id == jogId]

    simples_eventos = gera_a(eventos=eventos)

    a = gera_a(jog_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = simples_eventos.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_z_rank_jog(modelos=modelos, a=aVaep, jog_id=jogId, rem=rem)


def jog_xM(spadl_df, jogId, rem=[]):

    eventos = atomicspadl.convert_to_atomic(spadl_df)
    jog_df = eventos[eventos.player_id == jogId]

    simples_eventos = gera_a(eventos=eventos)

    a = gera_a(jog_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = simples_eventos.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_xM(modelos=modelos, a=aVaep, jog_id=jogId, rem=rem)


def jog_xM_rank(spadl_df, jogId, rem=[]):

    eventos = atomicspadl.convert_to_atomic(spadl_df)
    jog_df = eventos[eventos.player_id == jogId]

    simples_eventos = gera_a(eventos=eventos)

    a = gera_a(jog_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = simples_eventos.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_xM_rank(modelos=modelos, a=aVaep, jog_id=jogId, rem=rem)


def jog_xM_time(spadl_df, jogId, teamId, rem=[]):

    eventos = atomicspadl.convert_to_atomic(spadl_df)
    time_df = eventos[eventos.team_id == teamId]

    eventos_simples = gera_a(eventos=eventos)

    a = gera_a(time_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = eventos_simples.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_xM_rank(modelos=modelos, a=aVaep, jog_id=jogId, rem=rem)


def jogs_xM_time(spadl_df, jogs, teamId, rem=[]):  # falhas a se concertar

    eventos = atomicspadl.convert_to_atomic(spadl_df)
    time_df = eventos[eventos.team_id == teamId]

    eventos_simples = gera_a(eventos=eventos)

    a = gera_a(time_df)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)
    len(loc_candidates)
    inspect_strategies(loc_candidates, 50)

    eventVaep = vaep(spadl_df)
    aVaep = eventos_simples.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    plot_xM_rank_jogs(modelos=modelos, a=aVaep, jogs=jogs, team_id=teamId, rem=rem)


def get_players_ranking_for_cluster(cluster_data, rank_by="zscore"):
    """
    Obtém o ranking dos jogadores para um cluster de ações.
    """

    # VAEP médio geral do cluster
    cluster_mean_vaep = cluster_data["vaep_value"].mean()
    cluster_std_vaep = cluster_data["vaep_value"].std()

    # VAEP médio de todos os jogadores no cluster
    player_means = cluster_data.groupby("player_id")["vaep_value"].mean()

    player_ranking = defaultdict(dict)
    for player_id, vaep in player_means.items():
        if pd.isna(vaep) or cluster_mean_vaep == 0:
            player_ranking[player_id] = None
        else:
            # Calcula o z score do vaep para o jogador
            zscore = (vaep - cluster_mean_vaep) / cluster_std_vaep

            # Calcula o valor percentual em relação à média do cluster
            perc = ((vaep / cluster_mean_vaep) - 1) * 100

            player_ranking[player_id]["zscore"] = zscore
            player_ranking[player_id]["perc"] = perc

    # Remove jogadores sem ações no cluster
    player_ranking = {k: v for k, v in player_ranking.items() if v is not None}

    # Ordena os jogadores pelo percentual em relação à média do cluster
    sorted_ranking = sorted(
        player_ranking.items(), key=lambda x: x[1][rank_by], reverse=True
    )
    return {
        player_id: {
            "zscore": data["zscore"],
            "perc": data["perc"],
            "rank": idx + 1,
        }
        for idx, (player_id, data) in enumerate(sorted_ranking)
    }


def get_players_ranking_for_model(model, a, rank_by="zscore"):
    """
    Obtém o ranking dos jogadores para um modelo específico.
    """
    evento = a[a.type_name == model.name].copy()
    evento["x_end"] = evento["x"] + evento["dx"]
    evento["y_end"] = evento["y"] + evento["dy"]

    # Previsão dos clusters
    pos = evento[["x", "y"]]
    probs = model.predict_proba(pos)
    evento["cluster"] = probs.argmax(axis=1)

    # Agrupa por cluster e obtém o ranking dos jogadores
    player_rankings = {}
    for cluster_id in sorted(evento["cluster"].unique()):
        cluster_data = evento[evento["cluster"] == cluster_id]
        player_rankings[cluster_id] = get_players_ranking_for_cluster(
            cluster_data, rank_by
        )

    return player_rankings


def get_players_ranking_for_models(models, a, rank_by="zscore"):
    """
    Obtém o ranking dos jogadores para vários modelos.
    """
    all_rankings = {}
    for model in models:
        all_rankings[model.name] = get_players_ranking_for_model(model, a, rank_by)
    return all_rankings


def rank_players_overall(
    player_rankings, modelos, wanted_actions=None, threshold=2, compare_value="zscore"
):
    """
    Classifica os jogadores com base no número de vezes que superam o percentil especificado
    em ações específicas.
    """

    # Se  não for especificado, usa todas as ações dos modelos
    if wanted_actions is None:
        wanted_actions = [model.name for model in modelos]

    total_clusters = sum(
        len(player_rankings[model.name])
        for model in modelos
        if model.name in wanted_actions
    )

    player_figures_at_top = {}

    # Inicializa o dicionário para contar quantas vezes cada jogador está no topo
    for model_name, rankings in player_rankings.items():
        if wanted_actions is not None and model_name not in wanted_actions:
            continue

        for cluster_id, rankings in rankings.items():
            for player_id, data in rankings.items():
                if compare_value in ("zscore", "perc"):
                    if data[compare_value] >= threshold:
                        if player_id not in player_figures_at_top:
                            player_figures_at_top[player_id] = {
                                "count": 0,
                                "clusters": defaultdict(dict),
                            }
                        player_figures_at_top[player_id]["count"] += 1
                        player_figures_at_top[player_id]["clusters"][model_name][
                            cluster_id
                        ] = data
                elif compare_value == "rank":
                    if data[compare_value] <= threshold:
                        if player_id not in player_figures_at_top:
                            player_figures_at_top[player_id] = {
                                "count": 0,
                                "clusters": defaultdict(dict),
                            }
                        player_figures_at_top[player_id]["count"] += 1
                        player_figures_at_top[player_id]["clusters"][model_name][
                            cluster_id
                        ] = data

    # Filtra jogadores que estão no topo de pelo menos um cluster
    top_players = {
        player_id: {
            "count": data["count"],
            "percentage": data["count"] / total_clusters * 100,
            "clusters": data["clusters"],
        }
        for player_id, data in player_figures_at_top.items()
        if data["count"] > 0
    }

    if top_players.get(0) is not None:
        del top_players[0]

    # Ordena os jogadores pelo número de vezes que aparecem no topo
    sorted_top_players = sorted(
        top_players.items(), key=lambda x: x[1]["percentage"], reverse=True
    )

    ranked_top_players = []
    for idx, (player_id, data) in enumerate(sorted_top_players):
        ranked_top_players.append(
            {
                "rank": idx + 1,
                "player_id": player_id,
                "count": data["count"],
                "percentage": data["percentage"],
                "clusters": data["clusters"],
            }
        )

    return ranked_top_players


def get_overall_best_players(spadl_df, wanted_actions=None, percentile=0.75):
    """
    Obtém os jogadores que superam um certo percentil de VAEP em ações específicas.
    """
    print("Convertendo eventos para formato atômico...")
    eventos = atomicspadl.convert_to_atomic(spadl_df)

    print("Gerando eventos simples...")
    eventos_simples = gera_a(eventos=eventos)

    print("Gerando matriz de ações e pesos...")
    a = gera_a(eventos)
    cat_weights = gera_cat_pesos(a)
    experiments = gera_exp(cat_weights)

    print("Aprendendo modelos de mistura...")
    X = a[["x", "y"]]
    loc_candidates = learn_mixture_models(X, cat_weights, experiments)

    print("Calculando VAEP dos eventos...")
    eventVaep = vaep(spadl_df)

    print("Unindo eventos com VAEP...")
    aVaep = eventos_simples.merge(
        eventVaep[["original_event_id", "vaep_value"]],
        on="original_event_id",
        how="left",
    )
    modelos = list(mix.ilp_select_models_bic(loc_candidates))

    print("Obtendo rankings dos jogadores para os modelos...")
    player_rankings = get_players_ranking_for_models(modelos, aVaep)

    ranked_top_players = rank_players_overall(
        player_rankings,
        modelos,
        wanted_actions=wanted_actions,
        percentile=percentile,
    )

    return ranked_top_players


def plot_player_rankings(
    player_ranking,
    modelos,
    num_players=10,
    wanted_actions=None,
    show_score="zscore",
):
    # Dicionário para armazenar as figuras geradas (uma por ação selecionada)
    # Assim, o Streamlit pode exibir múltiplas figuras.
    figures = {}

    def _get_cluster_text(score, rank):
        if show_score == "zscore":
            return f"{rank}\n({score:.2f})"
        elif show_score == "perc":
            return f"{rank}\n({score:.1f}%)"

    for model in modelos:
        if wanted_actions is not None and model.name not in wanted_actions:
            continue

        if model.name not in player_ranking["clusters"]:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        pitch = Pitch(
            pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2
        )
        pitch.draw(ax=ax)

        ax.set_title(
            f"Rankings do Jogador - {player_ranking['jogNome']} - {model.name}",
            fontsize=16,
        )

        for cluster_id, gauss in enumerate(model.submodels):
            mean = gauss.mean
            cov = gauss.cov
            if cluster_id in player_ranking["clusters"][model.name]:
                score = player_ranking["clusters"][model.name][cluster_id][show_score]
                rank = player_ranking["clusters"][model.name][cluster_id]["rank"]
                rank_norm = mcolors.Normalize(vmin=1, vmax=num_players)
                cmap = plt.get_cmap("RdYlGn_r")
                color = cmap(rank_norm(rank))
                ax.text(
                    mean[0],
                    mean[1],
                    _get_cluster_text(score, rank),
                    color="white",
                    fontsize=11,
                    ha="center",
                    va="center",
                    path_effects=[withStroke(linewidth=3, foreground="black")],
                )
                add_ellips(ax, mean, cov, color=color, alpha=0.5)
            else:
                color = (0.5, 0.5, 0.5, 1.0)  # cinza neutro
                add_ellips(ax, mean, cov, color=color, alpha=0.5)

        # Armazena a figura no dicionário, usando o nome do modelo como chave
        figures[model.name] = fig

    # Retorna o dicionário de figuras
    return figures


def create_player_ranking_df(
    overall_best_players, players_df
):  # Renomeei 'players' para 'players_df' para clareza
    player_ranking = []

    for obp in overall_best_players:
        player_id = obp["player_id"]
        player_info = players_df.loc[players_df["player_id"] == player_id]
        player_name = (
            player_info["jogNome"].values[0]
            if not player_info.empty
            else "Unknown Player"
        )

        player_ranking.append(
            {
                "player_id": player_id,
                "rank": obp["rank"],
                "jogNome": player_name,  # <-- CORREÇÃO AQUI! Mudei de "player_name" para "jogNome"
                "clusters_count": obp["count"],
                "clusters_percentage": obp["percentage"],
                "clusters": obp["clusters"],
                # Removi as colunas "total_vaep" e "avg_vaep_per_90" como você solicitou.
            },
        )

    player_ranking_df = pd.DataFrame(player_ranking)

    return player_ranking_df


def salvar_modelos(modelos, caminho_arquivo):
    """
    Salva uma lista de modelos em um arquivo usando pickle.

    Parâmetros:
    -----------
    modelos : list
        A lista de modelos a ser salva.
    caminho_arquivo : str
        O nome do arquivo onde os modelos serão salvos (ex: 'modelos.pkl').
    """
    try:
        with open(caminho_arquivo, "wb") as f:
            pickle.dump(modelos, f)
        print(f"Modelos salvos com sucesso em '{caminho_arquivo}'")
    except Exception as e:
        print(f"Ocorreu um erro ao salvar os modelos: {e}")


def carregar_modelos(caminho_arquivo):
    """
    Carrega uma lista de modelos de um arquivo usando pickle.

    Parâmetros:
    -----------
    caminho_arquivo : str
        O nome do arquivo de onde os modelos serão carregados (ex: 'modelos.pkl').

    Retorna:
    --------
    list
        A lista de modelos carregada do arquivo, ou None se ocorrer um erro.
    """
    try:
        with open(caminho_arquivo, "rb") as f:
            modelos = pickle.load(f)
        print(f"Modelos carregados com sucesso de '{caminho_arquivo}'")
        return modelos
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar os modelos: {e}")
        return None


def load_events(path):
    events = pd.read_json(path_or_buf=path)
    # pré processamento em colunas da tabela de eventos para facilitar a conversão p/ SPADL
    events = events.rename(
        columns={
            "id": "event_id",
            "eventId": "type_id",
            "subEventId": "subtype_id",
            "teamId": "team_id",
            "playerId": "player_id",
            "matchId": "game_id",
        }
    )
    events["milliseconds"] = events["eventSec"] * 1000
    events["period_id"] = events["matchPeriod"].replace({"1H": 1, "2H": 2})

    return events


def load_matches(path):
    matches = pd.read_json(path_or_buf=path)
    # as informações dos times de cada partida estão em um dicionário dentro da coluna 'teamsData', então vamos separar essas informações
    team_matches = []
    for i in range(len(matches)):
        match = pd.DataFrame(matches.loc[i, "teamsData"]).T
        match["matchId"] = matches.loc[i, "wyId"]
        team_matches.append(match)
    team_matches = pd.concat(team_matches).reset_index(drop=True)

    return team_matches


def spadl_transform(events, matches):
    spadl_df = []
    game_ids = events.game_id.unique().tolist()
    for g in tqdm.tqdm(game_ids):
        match_events = events.loc[events.game_id == g]
        match_home_id = matches.loc[
            (matches.matchId == g) & (matches.side == "home"), "teamId"
        ].values[0]
        match_actions = spadl.wyscout.convert_to_actions(
            events=match_events, home_team_id=match_home_id
        )
        match_actions = spadl.play_left_to_right(
            actions=match_actions, home_team_id=match_home_id
        )
        match_actions = spadl.add_names(match_actions)
        spadl_df.append(match_actions)
    spadl_df = pd.concat(spadl_df).reset_index(drop=True)

    return spadl_df
