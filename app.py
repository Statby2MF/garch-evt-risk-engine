"""
Dashboard Streamlit — GARCH-EVT Risk Engine.

Interface interactive pour explorer :
    - Les données brutes (prix, rendements, volatilité)
    - L'ajustement EVT (GPD / POT)
    - La VaR/ES dynamiques (GARCH-EVT)
    - Le backtesting Bâle III (Kupiec, Christoffersen, DQ)
    - La comparaison des 4 modèles de VaR

Usage :
    streamlit run app.py

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""
from src.report_generator import generate_report
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_manager import get_returns, get_prices, CRYPTOS
from src.garch_model import fit_garch, forecast_volatility
from src.evt_model import (
    fit_gpd_pot, gpd_quantile, gpd_es, mean_residual_life,
)
from src.dynamic_var import rolling_var_es
from src.benchmark_models import var_historical, var_normal, var_garch_normal
from src.backtesting import run_full_backtest, backtest_table


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GARCH-EVT Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Style CSS personnalisé
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1em;
        color: #666;
        text-align: center;
        margin-bottom: 2em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #1f77b4;
    }
    .winner {
        background-color: #d4edda;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cache des calculs lourds
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def compute_garch_fit(returns: pd.Series, dist: str = "ged"):
    """Cache du fit GARCH."""
    return fit_garch(returns, model_type="garch", dist=dist)


@st.cache_data(show_spinner=False, ttl=3600)
def compute_evt_fit(z_values: tuple, threshold: float = 0.90):
    """Cache du fit EVT (reçoit un tuple pour hash)."""
    z = np.array(z_values)
    return fit_gpd_pot(z, threshold_quantile=threshold, tail="lower")


@st.cache_data(show_spinner=False, ttl=3600)
def compute_all_var(
    returns: pd.Series,
    window: int,
    p: float,
    refit_every: int,
    ticker: str,
):
    """Cache de tous les calculs de VaR (le plus lourd)."""
    models = {}

    models["Historique"] = var_historical(returns, window=window, p=p)
    models["Normale"] = var_normal(returns, window=window, p=p)
    models["GARCH-Normal"] = var_garch_normal(
        returns, window=window, p=p, refit_every=refit_every
    )

    result_evt = rolling_var_es(
        returns, window=window, p=p, refit_every=refit_every,
        model_type="garch", dist="ged",
        threshold_quantile=0.90, verbose=False,
    )
    models["GARCH-EVT"] = result_evt.var

    return models


@st.cache_data(show_spinner=False, ttl=3600)
def compute_backtest_table(returns: pd.Series, models: dict, p: float):
    """Cache du backtest."""
    table = backtest_table(returns, models, p=p)
    table["score_bale"] = table["kupiec_p"] + table["cc_p"]
    table = table.sort_values("score_bale", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


@st.cache_data(show_spinner=False, ttl=3600)
def cached_report_path(ticker: str, p: float, window: int, refit_every: int) -> str:
    """
    Génère le PDF et met en cache le chemin du fichier.
    Le cache évite de regénérer le même rapport à chaque clic.
    """
    path = generate_report(
        ticker=ticker,
        p=p,
        window=window,
        refit_every=refit_every,
        verbose=False,   # silencieux dans Streamlit
    )
    return str(path)
# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">📊 GARCH-EVT Risk Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Mesure dynamique du risque extrême sur les cryptomonnaies — '
    'GARCH(1,1) + Théorie des Valeurs Extrêmes (GPD/POT)</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    ticker = st.selectbox(
        "🪙 Cryptomonnaie",
        options=list(CRYPTOS.keys()),
        format_func=lambda x: f"{CRYPTOS[x]} ({x})",
        index=0,
    )

    p = st.select_slider(
        "📊 Niveau de confiance",
        options=[0.95, 0.99, 0.995],
        value=0.99,
        format_func=lambda x: f"{x:.1%}",
    )

    window = st.select_slider(
        "🪟 Fenêtre glissante (jours)",
        options=[500, 750, 1000, 1250, 1500],
        value=1000,
    )

    refit_every = st.select_slider(
        "🔄 Refit tous les N jours",
        options=[25, 50, 100, 200],
        value=50,
    )

    st.markdown("---")

    force_recompute = st.button(
        "🔄 Recalculer (vider le cache)",
        use_container_width=True,
    )
    if force_recompute:
        st.cache_data.clear()
        st.success("Cache vidé.")

    st.markdown("---")
    st.markdown(
        "**📚 Références**\n"
        "- McNeil & Frey (2000)\n"
        "- Bollerslev (1986)\n"
        "- Christoffersen (1998)\n"
        "- Kupiec (1995)"
    )
    st.markdown("---")
    st.markdown("### 📄 Rapport PDF")

    if st.button("📄 Générer le rapport PDF", use_container_width=True):
        with st.spinner(f"⏳ Génération du rapport pour {ticker}… "
                        f"(30-60 sec)"):
            try:
                pdf_path = cached_report_path(ticker, p, window, refit_every)
                st.session_state["pdf_path"] = pdf_path
                st.success("✅ Rapport prêt !")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    # Bouton de téléchargement si le PDF existe
    if "pdf_path" in st.session_state:
        pdf_path_obj = Path(st.session_state["pdf_path"])
        if pdf_path_obj.exists():
            with open(pdf_path_obj, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger le rapport",
                    data=f.read(),
                    file_name=pdf_path_obj.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.caption(f"📁 {pdf_path_obj.name}")
    st.caption("Statby2Mf")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

try:
    returns = get_returns(ticker)
    prices = get_prices(ticker)
except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Onglets principaux
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Données & Volatilité",
    "🎯 EVT (GPD)",
    "🔥 VaR & ES Dynamiques",
    "✅ Backtesting",
    "🏆 Comparaison finale",
])


# ===========================================================================
# ONGLET 1 — Données & Volatilité
# ===========================================================================

with tab1:
    st.header("📈 Données & Volatilité conditionnelle")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(returns):,}")
    col2.metric("Période", f"{(returns.index[-1] - returns.index[0]).days} jours")
    col3.metric("Rendement moyen", f"{returns.mean():.3f}% / jour")
    col4.metric("Volatilité inconditionnelle", f"{returns.std():.3f}% / jour")

    # --- Prix + rendements ---
    st.subheader("Prix et log-rendements")

    with st.spinner("Calcul du fit GARCH…"):
        fit = compute_garch_fit(returns, dist="ged")

    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Prix de clôture", "Log-rendements (%)"),
        vertical_spacing=0.1,
    )
    fig1.add_trace(
        go.Scatter(x=prices.index, y=prices.values, name="Prix",
                   line=dict(color="#1f77b4", width=1.5)),
        row=1, col=1,
    )
    fig1.add_trace(
        go.Scatter(x=returns.index, y=returns.values, name="Rendements",
                   line=dict(color="gray", width=0.5)),
        row=2, col=1,
    )
    fig1.update_layout(height=550, showlegend=False,
                       margin=dict(t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)

    # --- Volatilité conditionnelle ---
    st.subheader("Volatilité conditionnelle GARCH(1,1)")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=returns.index, y=returns.values,
        name="Rendements", line=dict(color="lightgray", width=0.5),
    ))
    fig2.add_trace(go.Scatter(
        x=fit.conditional_vol.index,
        y=fit.conditional_vol.values,
        name="σ_t (GARCH)", line=dict(color="#d62728", width=1.5),
    ))
    fig2.update_layout(
        height=400, title="σ_t conditionnelle",
        xaxis_title="Date", yaxis_title="Volatilité (%)",
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Résumé du fit GARCH ---
    st.subheader("Paramètres GARCH(1,1)-GED")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("α (ARCH)", f"{fit.alpha:.4f}")
    col2.metric("β (GARCH)", f"{fit.beta:.4f}")
    col3.metric("α + β (persistance)", f"{fit.persistence:.4f}")
    col4.metric("AIC", f"{fit.aic:.2f}")

    with st.expander("Voir les détails complets du fit"):
        st.text(fit.summary())


# ===========================================================================
# ONGLET 2 — EVT (GPD)
# ===========================================================================

with tab2:
    st.header("🎯 Théorie des Valeurs Extrêmes — Fit GPD")

    with st.spinner("Calcul du fit EVT…"):
        z_values = tuple(fit.residuals_std.values)
        evt = compute_evt_fit(z_values, threshold=0.90)

    # --- KPIs EVT ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ξ (shape)", f"{evt.xi:.4f}",
                help="ξ > 0 → queue épaisse (Fréchet)")
    col2.metric("β (scale)", f"{evt.beta:.4f}")
    col3.metric("Seuil u", f"{evt.threshold:.4f}")
    col4.metric("Excès observés", f"{evt.n_exceed} ({evt.exceedance_rate:.1%})")

    st.info(f"**Type de queue** : {evt.tail_type}")

    # --- Distribution des résidus + QQ plot ---
    st.subheader("Distribution des résidus standardisés")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fit.residuals_std.values, nbinsx=80,
            name="Résidus z_t", marker_color="#4c72b0",
            opacity=0.7,
        ))
        fig_hist.update_layout(
            title="Histogramme des z_t",
            xaxis_title="z_t", yaxis_title="Fréquence",
            height=350, showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # QQ-plot des excès vs GPD théorique
        from scipy.stats import genpareto
        losses = -fit.residuals_std.values
        exceedances = losses[losses > evt.threshold] - evt.threshold
        exceedances_sorted = np.sort(exceedances)
        n = len(exceedances_sorted)
        theoretical = genpareto.ppf(
            (np.arange(1, n + 1) - 0.5) / n, evt.xi, loc=0, scale=evt.beta
        )

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical, y=exceedances_sorted,
            mode="markers", name="Excès",
            marker=dict(color="#d62728", size=6),
        ))
        lim = max(theoretical.max(), exceedances_sorted.max())
        fig_qq.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim],
            mode="lines", name="y=x",
            line=dict(color="black", dash="dash"),
        ))
        fig_qq.update_layout(
            title="QQ-plot (excès vs GPD)",
            xaxis_title="Quantiles GPD théoriques",
            yaxis_title="Quantiles empiriques",
            height=350,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # --- Mean Residual Life plot ---
    st.subheader("Mean Residual Life plot")
    st.caption("Outil de diagnostic pour le choix du seuil u. "
               "La zone linéaire indique un seuil approprié.")

    mrl = mean_residual_life(fit.residuals_std.values,
                              tail="lower", n_thresholds=30)

    fig_mrl = go.Figure()
    fig_mrl.add_trace(go.Scatter(
        x=mrl["threshold"], y=mrl["mean_excess"],
        mode="lines+markers", name="Mean Excess",
        line=dict(color="#2ca02c"),
    ))
    fig_mrl.add_trace(go.Scatter(
        x=pd.concat([mrl["threshold"], mrl["threshold"][::-1]]),
        y=pd.concat([mrl["conf_high"], mrl["conf_low"][::-1]]),
        fill="toself", fillcolor="rgba(44,160,44,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="IC 95%",
    ))
    fig_mrl.add_vline(x=evt.threshold, line_dash="dash",
                       line_color="red",
                       annotation_text=f"u = {evt.threshold:.3f}")
    fig_mrl.update_layout(
        height=400,
        xaxis_title="Seuil u", yaxis_title="Mean excess",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_mrl, use_container_width=True)

    # --- Quantiles EVT ---
    st.subheader("Quantiles EVT (pertes standardisées)")
    rows = []
    for level in [0.95, 0.99, 0.995, 0.999]:
        if level > evt.threshold_quantile:
            rows.append({
                "Niveau": f"{level:.1%}",
                "VaR_std": f"{gpd_quantile(evt, level):.4f}",
                "ES_std": f"{gpd_es(evt, level):.4f}",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===========================================================================
# ONGLET 3 — VaR & ES Dynamiques
# ===========================================================================

with tab3:
    st.header("🔥 VaR & ES dynamiques (GARCH-EVT)")

    with st.spinner("Calcul du rolling backtest (peut prendre 2-3 min)…"):
        models = compute_all_var(returns, window, p, refit_every, ticker)

    # --- VaR des 4 modèles ---
    st.subheader("Comparaison des VaR")

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=returns.index, y=returns.values,
        name="Rendements", line=dict(color="lightgray", width=0.5),
    ))
    colors = {
        "Historique": "#1f77b4",
        "Normale": "#ff7f0e",
        "GARCH-Normal": "#2ca02c",
        "GARCH-EVT": "#d62728",
    }
    for name, var in models.items():
        fig_var.add_trace(go.Scatter(
            x=var.index, y=var.values, name=name,
            line=dict(color=colors[name],
                      width=2.0 if name == "GARCH-EVT" else 1.2),
        ))
    fig_var.update_layout(
        height=500,
        xaxis_title="Date", yaxis_title="VaR (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # --- KPIs sur la dernière VaR ---
    st.subheader(f"VaR au {returns.index[-1].date()}")
    col1, col2, col3, col4 = st.columns(4)
    for col, (name, var) in zip([col1, col2, col3, col4], models.items()):
        last = var.dropna().iloc[-1]
        col.metric(name, f"{last:.3f}%")

    # --- Stats descriptives ---
    st.subheader("Statistiques descriptives des VaR")
    stats = []
    for name, var in models.items():
        v = var.dropna()
        stats.append({
            "Modèle": name,
            "Moyenne": f"{v.mean():.3f}%",
            "Min (pire)": f"{v.min():.3f}%",
            "Max (meilleur)": f"{v.max():.3f}%",
            "Écart-type": f"{v.std():.3f}%",
        })
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)


# ===========================================================================
# ONGLET 4 — Backtesting
# ===========================================================================

with tab4:
    st.header("✅ Backtesting Bâle III")

    with st.spinner("Calcul des tests statistiques…"):
        models = compute_all_var(returns, window, p, refit_every, ticker)
        table = compute_backtest_table(returns, models, p)

    # --- Tableau principal ---
    st.subheader("Résultats des tests")

    display = table[[
        "rank", "model", "n_viol", "rate",
        "kupiec_p", "ind_p", "cc_p", "score_bale",
    ]].copy()
    display.columns = [
        "Rang", "Modèle", "Viol.", "Taux",
        "Kupiec p", "IND p", "CC p", "Score Bâle III",
    ]
    display["Taux"] = display["Taux"].apply(lambda x: f"{x:.4%}")
    display["Kupiec p"] = display["Kupiec p"].apply(lambda x: f"{x:.4f}")
    display["IND p"] = display["IND p"].apply(lambda x: f"{x:.4f}")
    display["CC p"] = display["CC p"].apply(lambda x: f"{x:.4f}")
    display["Score Bâle III"] = display["Score Bâle III"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Graphique des p-values ---
    st.subheader("p-values des tests Bâle III")

    fig_pv = go.Figure()
    for test_col, test_name in [
        ("kupiec_p", "Kupiec"),
        ("ind_p", "Christoffersen IND"),
        ("cc_p", "Christoffersen CC"),
    ]:
        fig_pv.add_trace(go.Bar(
            x=table["model"], y=table[test_col], name=test_name,
        ))
    fig_pv.add_hline(y=0.05, line_dash="dash", line_color="red",
                      annotation_text="Seuil α = 0.05")
    fig_pv.update_layout(
        barmode="group", height=400,
        yaxis_title="p-value",
        xaxis_title="Modèle",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_pv, use_container_width=True)

    # --- Violations ---
    st.subheader("Violations par modèle")

    fig_viol = make_subplots(
        rows=len(models), cols=1,
        shared_xaxes=True,
        subplot_titles=list(models.keys()),
        vertical_spacing=0.05,
    )
    for i, (name, var) in enumerate(models.items(), start=1):
        mask = var.notna()
        r = returns[mask]
        v = var[mask]
        violations = r < v

        fig_viol.add_trace(
            go.Scatter(x=r.index, y=r.values, name=name,
                       line=dict(color="lightgray", width=0.5),
                       showlegend=False),
            row=i, col=1,
        )
        fig_viol.add_trace(
            go.Scatter(x=v.index, y=v.values, name=f"{name} VaR",
                       line=dict(color="blue", width=1),
                       showlegend=False),
            row=i, col=1,
        )
        fig_viol.add_trace(
            go.Scatter(x=r[violations].index, y=r[violations].values,
                       mode="markers", marker=dict(color="red", size=8),
                       name=f"{name} Violations",
                       showlegend=False),
            row=i, col=1,
        )

    fig_viol.update_layout(height=200 * len(models),
                            margin=dict(t=40, b=20))
    st.plotly_chart(fig_viol, use_container_width=True)


# ===========================================================================
# ONGLET 5 — Comparaison finale
# ===========================================================================

with tab5:
    st.header("🏆 Comparaison finale")

    with st.spinner("Consolidation des résultats…"):
        models = compute_all_var(returns, window, p, refit_every, ticker)
        table = compute_backtest_table(returns, models, p)

    best = table.iloc[0]

    # --- Carte du gagnant ---
    st.markdown(
        f"""
        <div class="winner">
            <h2>🥇 Meilleur modèle : {best['model']}</h2>
            <p><b>Taux de violations</b> : {best['rate']:.4%} (cible : {1-p:.2%})</p>
            <p><b>Score Bâle III</b> : {best['score_bale']:.4f}</p>
            <p><b>Tests Bâle III passés</b> : {sum(table.iloc[0][['kupiec_p','ind_p','cc_p']] > 0.05)}/3</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Classement ---
    st.subheader("Classement par score Bâle III")

    fig_score = go.Figure()
    colors_bar = [
        "#28a745" if m == "GARCH-EVT" else "#4c72b0"
        for m in table["model"]
    ]
    fig_score.add_trace(go.Bar(
        x=table["score_bale"], y=table["model"],
        orientation="h", marker=dict(color=colors_bar),
        text=table["score_bale"].round(3),
        textposition="outside",
    ))
    fig_score.update_layout(
        height=350,
        xaxis_title="Score Bâle III (Kupiec + CC)",
        yaxis_title="",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # --- Synthèse ---
    st.subheader("📝 Synthèse")

    if best["model"] == "GARCH-EVT":
        st.success(
            f"**GARCH-EVT** remporte la comparaison avec un score Bâle III de "
            f"{best['score_bale']:.3f}, soit "
            f"**{best['score_bale'] / table.iloc[1]['score_bale']:.1f}×** "
            f"le score du 2e ({table.iloc[1]['model']}). "
            f"Le taux de violations de {best['rate']:.4%} est "
            f"quasi-optimal (cible : {1-p:.2%})."
        )

    st.markdown("""
    ### 📖 Interprétation

    - **Kupiec (POF)** teste si le taux de violations = 1 - niveau de confiance
    - **Christoffersen IND** teste l'indépendance des violations (pas de clustering)
    - **Christoffersen CC** combine POF + IND
    - **Score Bâle III** = Kupiec + CC (standard réglementaire)

    ### 🎯 Conclusion

    La supériorité empirique de **GARCH-EVT** sur les modèles paramétriques
    classiques (Normale, GARCH-Normal) confirme l'intérêt de la théorie des
    valeurs extrêmes pour la mesure du risque de queue sur les cryptomonnaies.
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "📊 **GARCH-EVT Risk Engine** — Statby2Mf — "
    "M2 Statistique, UGB Saint-Louis — "
    "Références : McNeil & Frey (2000), Bollerslev (1986), Christoffersen (1998)"
)
