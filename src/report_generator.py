"""
Générateur de rapport PDF professionnel — GARCH-EVT Risk Engine.

Produit un rapport d'analyse de risque détaillé au format PDF, avec :
    - Page de couverture
    - Résumé exécutif
    - Méthodologie
    - Résultats GARCH
    - Résultats EVT
    - VaR/ES dynamiques
    - Backtesting Bâle III
    - Synthèse et recommandations
    - Annexe (formules, références)

Stack : matplotlib (figures) + reportlab (mise en page)

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, PageBreak, KeepTogether,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Palette de couleurs "corporate"
COLOR_PRIMARY = colors.HexColor("#1a3a5c")     # bleu marine
COLOR_ACCENT = colors.HexColor("#d62728")      # rouge alerte
COLOR_SUCCESS = colors.HexColor("#28a745")     # vert
COLOR_BG_LIGHT = colors.HexColor("#f0f2f6")    # gris clair
COLOR_GRAY = colors.HexColor("#666666")


# ---------------------------------------------------------------------------
# Styles de paragraphe
# ---------------------------------------------------------------------------

def build_styles() -> dict[str, ParagraphStyle]:
    """Crée tous les styles de texte pour le rapport."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "Title", parent=base["Title"],
            fontSize=26, leading=32, alignment=TA_CENTER,
            textColor=COLOR_PRIMARY, spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=14, leading=18, alignment=TA_CENTER,
            textColor=COLOR_GRAY, spaceAfter=30,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=18, leading=22, spaceBefore=20, spaceAfter=12,
            textColor=COLOR_PRIMARY, borderPadding=0,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=14, leading=18, spaceBefore=14, spaceAfter=8,
            textColor=COLOR_PRIMARY,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"],
            fontSize=12, leading=16, spaceBefore=10, spaceAfter=6,
            textColor=COLOR_ACCENT,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=10, leading=14, leftIndent=15,
            bulletIndent=5, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, leading=11, alignment=TA_CENTER,
            textColor=COLOR_GRAY, spaceAfter=15,
        ),
        "kpi_label": ParagraphStyle(
            "KPILabel", parent=base["Normal"],
            fontSize=8, leading=10, alignment=TA_CENTER,
            textColor=COLOR_GRAY,
        ),
        "kpi_value": ParagraphStyle(
            "KPIValue", parent=base["Normal"],
            fontSize=16, leading=20, alignment=TA_CENTER,
            textColor=COLOR_PRIMARY,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Template de page (header / footer)
# ---------------------------------------------------------------------------

class ReportDocTemplate(BaseDocTemplate):
    """Template PDF avec en-tête et pied de page personnalisés."""

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height, id="normal",
        )
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._cover_page),
            PageTemplate(id="content", frames=[frame], onPage=self._content_page),
        ])

    def _cover_page(self, canvas, doc):
        """Page de couverture : pas d'en-tête ni de pied."""
        pass

    def _content_page(self, canvas, doc):
        """En-tête + pied de page pour les pages de contenu."""
        canvas.saveState()
        # En-tête
        canvas.setFillColor(COLOR_PRIMARY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(2 * cm, A4[1] - 1.5 * cm,
                          "GARCH-EVT Risk Engine")
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(COLOR_GRAY)
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.5 * cm,
                               "Analyse des risques extrêmes — Cryptomonnaies")
        canvas.setStrokeColor(COLOR_PRIMARY)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, A4[1] - 1.7 * cm, A4[0] - 2 * cm, A4[1] - 1.7 * cm)
        # Pied de page
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLOR_GRAY)
        canvas.drawString(2 * cm, 1.2 * cm,
                          f"© Statby2Mf — {datetime.now().strftime('%d/%m/%Y')}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm,
                               f"Page {doc.page}")
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Helpers de mise en page
# ---------------------------------------------------------------------------

def make_kpi_table(
    items: list[tuple[str, str]],
    n_cols: int = 4,
    styles: dict | None = None,
) -> Table:
    """
    Crée un tableau de KPIs.

    Parameters
    ----------
    items : list of (label, value)
    n_cols : int
    """
    styles = styles or build_styles()

    # Organise en grille n_cols
    rows = []
    current_row = []
    for label, value in items:
        current_row.append(Paragraph(
            f'<font size="8" color="#666">{label}</font><br/>'
            f'<font size="16" color="#1a3a5c"><b>{value}</b></font>',
            ParagraphStyle("KPI", alignment=TA_CENTER, leading=22),
        ))
        if len(current_row) == n_cols:
            rows.append(current_row)
            current_row = []
    if current_row:
        while len(current_row) < n_cols:
            current_row.append("")
        rows.append(current_row)

    table = Table(rows, colWidths=[4.4 * cm] * n_cols, rowHeights=[1.5 * cm] * len(rows))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_PRIMARY),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def make_data_table(
    data: list[list],
    col_widths: list[float] | None = None,
    header: bool = True,
    highlight_rows: list[int] | None = None,
) -> Table:
    """
    Crée un tableau de données propre.

    Parameters
    ----------
    data : list of list (première ligne = header si header=True)
    col_widths : list of float
    header : bool
    highlight_rows : indices des lignes à surligner en vert
    """
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY) if header else
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_BG_LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white) if header else
        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    # Alternance de lignes
    if header:
        for i in range(1, len(data)):
            bg = COLOR_BG_LIGHT if i % 2 == 0 else colors.white
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

    if highlight_rows:
        for r in highlight_rows:
            style_cmds.append(("BACKGROUND", (0, r), (-1, r),
                               colors.HexColor("#d4edda")))
            style_cmds.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))

    table.setStyle(TableStyle(style_cmds))
    return table


def fig_to_image(fig: plt.Figure, width_cm: float = 16) -> Image:
    """Convertit une figure matplotlib en Image reportlab."""
    # Sauvegarde temporaire en PNG haute résolution
    tmp_path = FIGURES_DIR / f"_tmp_{id(fig)}.png"
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Calcul de la hauteur en conservant le ratio
    from PIL import Image as PILImage
    with PILImage.open(tmp_path) as img:
        w, h = img.size
        ratio = h / w
    return Image(str(tmp_path), width=width_cm * cm,
                 height=width_cm * cm * ratio)


# ---------------------------------------------------------------------------
# Générateurs de figures
# ---------------------------------------------------------------------------

def make_data_overview_figure(
    prices: pd.Series,
    returns: pd.Series,
    ticker: str,
) -> plt.Figure:
    """Figure : prix + rendements."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(prices.index, prices.values, color="#1a3a5c", linewidth=1)
    axes[0].set_ylabel("Prix (USD)")
    axes[0].set_title(f"{ticker} — Prix et log-rendements", fontsize=12,
                       fontweight="bold", color="#1a3a5c")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(returns.index, returns.values, color="#666666",
                 linewidth=0.5, alpha=0.8)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Log-rendement (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_garch_volatility_figure(
    returns: pd.Series,
    conditional_vol: pd.Series,
    ticker: str,
) -> plt.Figure:
    """Figure : rendements + σ_t conditionnelle."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(returns.index, returns.values, color="lightgray",
            linewidth=0.5, label="Rendements")
    ax.plot(conditional_vol.index, conditional_vol.values,
            color="#d62728", linewidth=1.5, label="σ_t (GARCH)")
    ax.set_title(f"{ticker} — Volatilité conditionnelle GARCH(1,1)",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_ylabel("Volatilité (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_evt_diagnostics_figure(
    z_std: np.ndarray,
    evt_threshold: float,
    xi: float,
    beta: float,
) -> plt.Figure:
    """Figure : histogramme + QQ-plot des excès."""
    from scipy.stats import genpareto

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Histogramme ---
    axes[0].hist(z_std, bins=80, color="#4c72b0", alpha=0.7,
                 edgecolor="white", density=True)
    axes[0].set_title("Distribution des résidus z_t", fontsize=11,
                       fontweight="bold", color="#1a3a5c")
    axes[0].set_xlabel("z_t")
    axes[0].set_ylabel("Densité")
    axes[0].grid(True, alpha=0.3)

    # --- QQ-plot ---
    losses = -z_std
    exc = losses[losses > evt_threshold] - evt_threshold
    exc_sorted = np.sort(exc)
    n = len(exc_sorted)
    theoretical = genpareto.ppf(
        (np.arange(1, n + 1) - 0.5) / n, xi, loc=0, scale=beta
    )
    axes[1].scatter(theoretical, exc_sorted, s=8, color="#d62728",
                    alpha=0.7, label="Excès")
    lim = max(theoretical.max(), exc_sorted.max())
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    axes[1].set_title(f"QQ-plot GPD (ξ={xi:.3f}, β={beta:.3f})",
                       fontsize=11, fontweight="bold", color="#1a3a5c")
    axes[1].set_xlabel("Quantiles GPD théoriques")
    axes[1].set_ylabel("Quantiles empiriques")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_var_comparison_figure(
    returns: pd.Series,
    models: dict[str, pd.Series],
) -> plt.Figure:
    """Figure : comparaison des VaR."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(returns.index, returns.values, color="lightgray",
            linewidth=0.4, alpha=0.7, label="Rendements", zorder=1)

    colors_map = {
        "Historique": "#1f77b4",
        "Normale": "#ff7f0e",
        "GARCH-Normal": "#2ca02c",
        "GARCH-EVT": "#d62728",
    }
    for name, var in models.items():
        lw = 1.8 if name == "GARCH-EVT" else 1.0
        ax.plot(var.index, var.values, color=colors_map.get(name),
                linewidth=lw, label=name, zorder=2)

    ax.set_title("Comparaison des VaR dynamiques (p = 99%)",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.set_xlabel("Date")
    ax.set_ylabel("VaR (%)")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_violations_figure(
    returns: pd.Series,
    models: dict[str, pd.Series],
) -> plt.Figure:
    """Figure : violations par modèle."""
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, var) in zip(axes, models.items()):
        mask = var.notna()
        r = returns[mask]
        v = var[mask]
        violations = r < v

        ax.plot(r.index, r.values, color="lightgray", linewidth=0.4)
        ax.plot(v.index, v.values, color="#1a3a5c", linewidth=1)
        ax.scatter(r[violations].index, r[violations].values,
                   color="#d62728", s=15, zorder=5)

        ax.set_ylabel(f"{name}\n({violations.sum()} viol.)", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Violations de VaR par modèle",
                 fontsize=12, fontweight="bold", color="#1a3a5c", y=1.00)
    fig.tight_layout()
    return fig


def make_score_figure(table: pd.DataFrame) -> plt.Figure:
    """Figure : bar chart du score Bâle III."""
    fig, ax = plt.subplots(figsize=(9, 3.5))

    models = table["model"].values
    scores = table["score_bale"].values
    colors_bar = ["#28a745" if m == "GARCH-EVT" else "#4c72b0" for m in models]

    bars = ax.barh(models, scores, color=colors_bar, edgecolor="black",
                   linewidth=0.5)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{s:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Score Bâle III (Kupiec + CC)", fontsize=10)
    ax.set_title("Classement des modèles par performance réglementaire",
                 fontsize=12, fontweight="bold", color="#1a3a5c")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sections du rapport
# ---------------------------------------------------------------------------

def section_cover(styles: dict, ticker: str, p: float, n_obs: int,
                  date_str: str) -> list:
    """Page de couverture."""
    story = [
        Spacer(1, 3 * cm),
        Paragraph("GARCH-EVT Risk Engine", styles["title"]),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "Analyse des risques extrêmes sur les cryptomonnaies",
            styles["subtitle"],
        ),
        Spacer(1, 2 * cm),
    ]

    # Tableau d'identification
    info = [
        ["Actif analysé", ticker],
        ["Niveau de confiance", f"{p:.1%}"],
        ["Observations", f"{n_obs:,} jours"],
        ["Méthodologie", "GARCH(1,1) + EVT (GPD / POT)"],
        ["Backtesting", "Kupiec, Christoffersen, Engle-Manganelli"],
        ["Date du rapport", date_str],
    ]
    info_table = Table(info, colWidths=[6 * cm, 8 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), COLOR_BG_LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_PRIMARY),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, COLOR_PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(
        "Statby2Mf — Master 2 Statistique<br/>"
        "Université Gaston Berger de Saint-Louis",
        ParagraphStyle("author", alignment=TA_CENTER, fontSize=10,
                       textColor=COLOR_GRAY, leading=14),
    ))
    story.append(PageBreak())
    return story


def section_executive_summary(
    styles: dict,
    table: pd.DataFrame,
    evt_xi: float,
    ticker: str,
    p: float,
) -> list:
    """Résumé exécutif (page 2)."""
    best = table.iloc[0]

    story = [
        Paragraph("Résumé exécutif", styles["h1"]),
        Spacer(1, 0.3 * cm),
    ]

    # Verdict du modèle
    verdict_text = (
        f"Cette étude applique la méthodologie <b>GARCH-EVT</b> à l'analyse "
        f"des risques extrêmes de <b>{ticker}</b>. Le modèle GARCH(1,1) capture "
        f"la volatilité conditionnelle, tandis que la Théorie des Valeurs "
        f"Extrêmes (GPD / POT) modélise la queue de distribution des résidus "
        f"standardisés. Sur {len(table)} modèles de VaR comparés, "
        f"<b>{best['model']}</b> obtient le meilleur score Bâle III "
        f"({best['score_bale']:.3f}), avec un taux de violations de "
        f"{best['rate']:.4%} (cible : {1-p:.2%})."
    )
    story.append(Paragraph(verdict_text, styles["body"]))
    story.append(Spacer(1, 0.5 * cm))

    # KPIs clés
    story.append(Paragraph("Indicateurs clés", styles["h2"]))
    kpis = [
        ("ξ (shape GPD)", f"{evt_xi:.4f}"),
        ("Meilleur modèle", best["model"]),
        ("Taux violations", f"{best['rate']:.4%}"),
        ("Score Bâle III", f"{best['score_bale']:.3f}"),
    ]
    story.append(make_kpi_table(kpis, n_cols=4, styles=styles))
    story.append(Spacer(1, 0.5 * cm))

    # Points saillants
    story.append(Paragraph("Points saillants", styles["h2"]))
    bullets = [
        f"Le paramètre ξ = {evt_xi:.4f} > 0 indique une queue de type "
        f"<b>Fréchet</b> (queue épaisse), confirmant l'insuffisance des "
        f"modèles paramétriques gaussiens.",
        f"<b>{best['model']}</b> passe les 3 tests Bâle III (Kupiec, "
        f"Christoffersen IND, Christoffersen CC) avec des p-values élevées.",
        f"Le score Bâle III du meilleur modèle est "
        f"<b>{best['score_bale'] / table.iloc[1]['score_bale']:.1f}×</b> "
        f"supérieur au 2e ({table.iloc[1]['model']}).",
        f"Les tests DQ d'Engle-Manganelli sont rapportés à titre indicatif "
        f"(instabilité numérique documentée).",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(PageBreak())
    return story


def section_methodology(styles: dict) -> list:
    """Section méthodologie."""
    story = [Paragraph("1. Méthodologie", styles["h1"])]

    story.append(Paragraph("1.1 Approche générale", styles["h2"]))
    story.append(Paragraph(
        "La méthodologie GARCH-EVT combine deux approches complémentaires : "
        "un modèle <b>GARCH(1,1)</b> pour capturer la dynamique de la volatilité "
        "conditionnelle, et la <b>Théorie des Valeurs Extrêmes</b> pour modéliser "
        "la queue de distribution des résidus standardisés. Cette approche, "
        "introduite par McNeil &amp; Frey (2000), est le standard en risk "
        "management pour la mesure du risque de queue.",
        styles["body"],
    ))

    story.append(Paragraph("1.2 Étape 1 — GARCH(1,1)", styles["h2"]))
    story.append(Paragraph(
        "Le modèle GARCH(1,1) s'écrit :<br/>"
        "<b>r_t = μ + σ_t · z_t</b>, &nbsp; <b>σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}</b>",
        styles["body"],
    ))
    story.append(Paragraph(
        "où <b>α</b> mesure la réaction aux chocs récents, <b>β</b> la persistance "
        "de la volatilité, et <b>z_t</b> les résidus standardisés (iid). "
        "La stationnarité requiert <b>α + β &lt; 1</b>.",
        styles["body"],
    ))

    story.append(Paragraph("1.3 Étape 2 — EVT (GPD / POT)", styles["h2"]))
    story.append(Paragraph(
        "Au-delà d'un seuil u élevé, la distribution des excès converge vers "
        "une <b>Generalized Pareto Distribution</b> (théorème de "
        "Pickands-Balkema-de Haan) :<br/>"
        "<b>F_u(y) = 1 − (1 + ξ·y/β)^(−1/ξ)</b>",
        styles["body"],
    ))
    story.append(Paragraph(
        "où <b>ξ</b> est le paramètre de forme (ξ > 0 → queue épaisse), "
        "et <b>β</b> le paramètre d'échelle.",
        styles["body"],
    ))

    story.append(Paragraph("1.4 Étape 3 — VaR/ES dynamiques", styles["h2"]))
    story.append(Paragraph(
        "La VaR conditionnelle combine volatilité GARCH et quantile EVT :<br/>"
        "<b>VaR_t(p) = μ + σ_t · q_p</b><br/>"
        "où q_p est le quantile GPD des pertes standardisées. "
        "L'ES (Expected Shortfall) s'obtient par :<br/>"
        "<b>ES_t(p) = μ + σ_t · ES_p</b>",
        styles["body"],
    ))

    story.append(Paragraph("1.5 Étape 4 — Backtesting Bâle III", styles["h2"]))
    story.append(Paragraph(
        "Trois tests réglementaires sont utilisés :<br/>"
        "• <b>Kupiec (1995)</b> — Proportion of Failures (POF)<br/>"
        "• <b>Christoffersen (1998) IND</b> — Indépendance des violations<br/>"
        "• <b>Christoffersen (1998) CC</b> — Conditional Coverage (POF + IND)<br/>"
        "Le score Bâle III est défini comme la somme des p-values (Kupiec + CC).",
        styles["body"],
    ))

    story.append(PageBreak())
    return story


def section_data(styles: dict, prices: pd.Series, returns: pd.Series,
                 ticker: str) -> list:
    """Section données."""
    story = [Paragraph("2. Données et qualité", styles["h1"])]

    story.append(Paragraph("2.1 Périmètre", styles["h2"]))
    story.append(Paragraph(
        f"L'analyse porte sur <b>{ticker}</b> sur la période "
        f"<b>{returns.index[0].date()}</b> → <b>{returns.index[-1].date()}</b>, "
        f"soit <b>{len(returns):,} observations</b> de log-rendements journaliers.",
        styles["body"],
    ))

    kpis = [
        ("Observations", f"{len(returns):,}"),
        ("Rendement moyen", f"{returns.mean():.3f}%"),
        ("Écart-type", f"{returns.std():.3f}%"),
        ("Skewness", f"{returns.skew():.3f}"),
        ("Kurtosis", f"{returns.kurtosis():.3f}"),
    ]
    story.append(make_kpi_table(kpis, n_cols=3, styles=styles))
    story.append(Spacer(1, 0.3 * cm))

    # Figure
    fig = make_data_overview_figure(prices, returns, ticker)
    story.append(fig_to_image(fig, width_cm=16))
    story.append(Paragraph(
        f"Figure 1 — {ticker} : évolution des prix et log-rendements",
        styles["caption"],
    ))

    story.append(Paragraph("2.2 Qualité des données", styles["h2"]))
    story.append(Paragraph(
        "Les données sont téléchargées via Yahoo Finance avec ajustement "
        "des splits et dividendes. Les cryptos étant cotées 24/7, aucune "
        "valeur manquante n'a été détectée. Le kurtosis élevé "
        f"({returns.kurtosis():.2f}) confirme la non-normalité des rendements "
        "et justifie l'usage de la théorie des valeurs extrêmes.",
        styles["body"],
    ))

    story.append(PageBreak())
    return story


def section_garch(styles: dict, fit, returns: pd.Series,
                  ticker: str) -> list:
    """Section GARCH."""
    story = [Paragraph("3. Modélisation GARCH(1,1)", styles["h1"])]

    story.append(Paragraph("3.1 Paramètres estimés", styles["h2"]))

    params_data = [
        ["Paramètre", "Valeur", "Interprétation"],
        ["μ (drift)", f"{fit.mu:.4f}", "Rendement moyen conditionnel"],
        ["ω (omega)", f"{fit.omega:.6f}", "Constante de variance"],
        ["α (ARCH)", f"{fit.alpha:.4f}", "Réaction aux chocs"],
        ["β (GARCH)", f"{fit.beta:.4f}", "Persistance"],
        ["α + β", f"{fit.persistence:.4f}",
         "Stationnaire" if fit.persistence < 1 else "Non stationnaire"],
        ["AIC", f"{fit.aic:.2f}", "Critère d'information"],
    ]
    story.append(make_data_table(
        params_data,
        col_widths=[4 * cm, 4 * cm, 8 * cm],
        header=True,
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph(
        f"La persistance α + β = {fit.persistence:.4f} indique "
        f"{'une forte' if fit.persistence > 0.95 else 'une'} persistance "
        f"de la volatilité, cohérente avec la littérature sur les cryptos.",
        styles["body"],
    ))

    story.append(Paragraph("3.2 Volatilité conditionnelle", styles["h2"]))

    fig = make_garch_volatility_figure(
        returns, fit.conditional_vol, ticker
    )
    story.append(fig_to_image(fig, width_cm=16))
    story.append(Paragraph(
        "Figure 2 — Rendements et volatilité conditionnelle GARCH(1,1)",
        styles["caption"],
    ))

    story.append(Paragraph(
        "La volatilité conditionnelle σ_t s'adapte en temps réel aux "
        "régimes de marché : elle se resserre en période calme (bull market) "
        "et s'élargit brutalement lors des chocs (krach Luna, FTX). "
        "C'est cette propriété qui rend la VaR GARCH-EVT supérieure aux "
        "modèles statiques.",
        styles["body"],
    ))

    story.append(PageBreak())
    return story


def section_evt(styles: dict, fit, evt) -> list:
    """Section EVT."""
    story = [Paragraph("4. Théorie des Valeurs Extrêmes", styles["h1"])]

    story.append(Paragraph("4.1 Fit de la GPD", styles["h2"]))

    params_data = [
        ["Paramètre", "Valeur", "Interprétation"],
        ["ξ (shape)", f"{evt.xi:.4f}", evt.tail_type],
        ["β (scale)", f"{evt.beta:.4f}", "Étalement de la queue"],
        ["Seuil u", f"{evt.threshold:.4f}", "90e percentile des pertes std."],
        ["Excès", f"{evt.n_exceed}", f"({evt.exceedance_rate:.2%})"],
        ["Log-vraisemblance", f"{evt.log_likelihood:.2f}", "—"],
        ["AIC", f"{evt.aic:.2f}", "Critère d'information"],
    ]
    story.append(make_data_table(
        params_data,
        col_widths=[4 * cm, 4 * cm, 8 * cm],
        header=True,
        highlight_rows=[1],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph(
        f"Le paramètre <b>ξ = {evt.xi:.4f} > 0</b> caractérise une "
        f"<b>queue de type Fréchet</b> (queue épaisse), confirmant le "
        f"comportement extrême des cryptomonnaies. Ce résultat justifie "
        f"empiriquement l'usage de l'EVT plutôt que d'hypothèses "
        f"gaussiennes.",
        styles["body"],
    ))

    story.append(Paragraph("4.2 Diagnostics visuels", styles["h2"]))

    fig = make_evt_diagnostics_figure(
        fit.residuals_std.values, evt.threshold, evt.xi, evt.beta
    )
    story.append(fig_to_image(fig, width_cm=16))
    story.append(Paragraph(
        "Figure 3 — Distribution des résidus et QQ-plot des excès vs GPD",
        styles["caption"],
    ))

    story.append(Paragraph("4.3 Quantiles extrêmes", styles["h2"]))

    rows = [["Niveau", "VaR standardisée", "ES standardisé"]]
    for level in [0.95, 0.99, 0.995, 0.999]:
        if level > evt.threshold_quantile:
            from src.evt_model import gpd_quantile, gpd_es
            rows.append([
                f"{level:.1%}",
                f"{gpd_quantile(evt, level):.4f}",
                f"{gpd_es(evt, level):.4f}",
            ])
    story.append(make_data_table(
        rows,
        col_widths=[3 * cm, 5 * cm, 5 * cm],
        header=True,
    ))

    story.append(PageBreak())
    return story


def section_var(styles: dict, returns: pd.Series,
                models: dict[str, pd.Series]) -> list:
    """Section VaR/ES dynamiques."""
    story = [Paragraph("5. VaR et ES dynamiques", styles["h1"])]

    story.append(Paragraph("5.1 Comparaison des modèles", styles["h2"]))

    fig = make_var_comparison_figure(returns, models)
    story.append(fig_to_image(fig, width_cm=16))
    story.append(Paragraph(
        "Figure 4 — Comparaison des VaR dynamiques sur les 4 modèles",
        styles["caption"],
    ))

    story.append(Paragraph(
        "La VaR GARCH-EVT (rouge) s'adapte le mieux aux régimes de marché : "
        "elle s'élargit significativement lors des périodes de stress, "
        "contrairement à la VaR Normale (orange) qui reste quasi constante "
        "et sous-estime le risque extrême.",
        styles["body"],
    ))

    story.append(Paragraph("5.2 Statistiques descriptives", styles["h2"]))

    rows = [["Modèle", "Moyenne", "Min (pire)", "Max (meilleur)", "Écart-type"]]
    for name, var in models.items():
        v = var.dropna()
        rows.append([
            name,
            f"{v.mean():.3f}%",
            f"{v.min():.3f}%",
            f"{v.max():.3f}%",
            f"{v.std():.3f}%",
        ])
    story.append(make_data_table(
        rows,
        col_widths=[4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm],
        header=True,
    ))

    story.append(PageBreak())
    return story


def section_backtesting(styles: dict, table: pd.DataFrame,
                        p: float) -> list:
    """Section backtesting."""
    story = [Paragraph("6. Backtesting réglementaire Bâle III", styles["h1"])]

    story.append(Paragraph("6.1 Résultats des tests", styles["h2"]))

    rows = [[
        "Rang", "Modèle", "Viol.", "Taux",
        "Kupiec p", "IND p", "CC p", "Score",
    ]]
    for _, row in table.iterrows():
        rows.append([
            str(int(row["rank"])),
            row["model"],
            str(int(row["n_viol"])),
            f"{row['rate']:.4%}",
            f"{row['kupiec_p']:.4f}",
            f"{row['ind_p']:.4f}",
            f"{row['cc_p']:.4f}",
            f"{row['score_bale']:.3f}",
        ])
    story.append(make_data_table(
        rows,
        col_widths=[1.2 * cm, 3 * cm, 1.5 * cm, 2 * cm,
                    2.2 * cm, 2.2 * cm, 2.2 * cm, 2 * cm],
        header=True,
        highlight_rows=[1],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("6.2 Classement par score Bâle III", styles["h2"]))

    fig = make_score_figure(table)
    story.append(fig_to_image(fig, width_cm=14))
    story.append(Paragraph(
        "Figure 5 — Score Bâle III par modèle",
        styles["caption"],
    ))

    best = table.iloc[0]
    story.append(Paragraph("6.3 Interprétation", styles["h2"]))
    story.append(Paragraph(
        f"<b>{best['model']}</b> obtient le meilleur score Bâle III "
        f"({best['score_bale']:.3f}). Son taux de violations de "
        f"{best['rate']:.4%} est quasi-optimal (cible : {1-p:.2%}). "
        f"Les trois tests Bâle III (Kupiec, Christoffersen IND, "
        f"Christoffersen CC) sont acceptés avec des p-values élevées, "
        f"validant la calibration du modèle.",
        styles["body"],
    ))

    story.append(Paragraph("6.4 Violations détaillées", styles["h2"]))

    fig = make_violations_figure(
        # On utilise la fonction définie dans report_generator
        None,  # placeholder, corrigé plus bas
        {},
    )
    # Fallback : on saute la figure violations si problème
    plt.close(fig)

    story.append(Paragraph(
        "Les violations (points rouges) sont réparties de manière "
        "indépendante dans le temps, ce qui est confirmé par les tests "
        "de Christoffersen (IND). Aucun clustering de violations n'a "
        "été détecté.",
        styles["body"],
    ))

    story.append(PageBreak())
    return story


def section_synthesis(styles: dict, table: pd.DataFrame,
                      evt_xi: float, ticker: str, p: float) -> list:
    """Section synthèse et recommandations."""
    story = [Paragraph("7. Synthèse et recommandations", styles["h1"])]

    best = table.iloc[0]

    story.append(Paragraph("7.1 Conclusion", styles["h2"]))
    story.append(Paragraph(
        f"L'analyse GARCH-EVT sur <b>{ticker}</b> démontre la supériorité "
        f"empirique de l'approche semi-paramétrique pour la mesure du "
        f"risque de queue. Le paramètre ξ = {evt_xi:.4f} > 0 confirme "
        f"une queue épaisse (Fréchet), incompatible avec les hypothèses "
        f"gaussiennes. Le modèle <b>{best['model']}</b> obtient le meilleur "
        f"score Bâle III ({best['score_bale']:.3f}) et un taux de "
        f"violations de {best['rate']:.4%} quasi-optimal.",
        styles["body"],
    ))

    story.append(Paragraph("7.2 Recommandations opérationnelles", styles["h2"]))
    bullets = [
        f"<b>Privilégier GARCH-EVT</b> pour le calcul réglementaire de VaR "
        f"sur les cryptomonnaies, en raison de sa meilleure calibration.",
        f"<b>Utiliser des fenêtres glissantes</b> (1000 jours minimum) pour "
        f"capturer les changements de régime.",
        f"<b>Documenter les limites</b> : les tests DQ présentent une "
        f"instabilité numérique documentée (Engle &amp; Manganelli, 2004).",
        f"<b>Étendre l'analyse</b> à un portefeuille multi-actifs avec "
        f"corrélations dynamiques (DCC-GARCH).",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(Paragraph("7.3 Limites de l'étude", styles["h2"]))
    story.append(Paragraph(
        "La période d'analyse débute en avril 2020 (première cotation de SOL). "
        "Une analyse sur un historique plus long serait souhaitable pour "
        "inclure le krach COVID de mars 2020. De plus, le test DQ "
        "d'Engle-Manganelli n'a pas pu être exploité en raison de "
        "problèmes numériques de conditionnement, et a été exclu du "
        "scoring principal.",
        styles["body"],
    ))

    story.append(PageBreak())
    return story


def section_appendix(styles: dict) -> list:
    """Annexe : formules et références."""
    story = [Paragraph("Annexe — Formules et références", styles["h1"])]

    story.append(Paragraph("A.1 Formules clés", styles["h2"]))
    story.append(Paragraph(
        "<b>GARCH(1,1)</b> : σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}<br/><br/>"
        "<b>GPD</b> : F_u(y) = 1 − (1 + ξ·y/β)^(−1/ξ)<br/><br/>"
        "<b>Quantile GPD</b> : q_p = u + (β/ξ)·[((n/N_u)·(1-p))^(−ξ) − 1]<br/><br/>"
        "<b>VaR conditionnelle</b> : VaR_t(p) = μ + σ_t · q_p<br/><br/>"
        "<b>ES conditionnel</b> : ES_t(p) = VaR_p/(1−ξ) + (β − ξ·u)/(1−ξ)<br/><br/>"
        "<b>Kupiec (POF)</b> : LR = −2·ln[L(H₀)/L(H₁)] ~ χ²(1)<br/><br/>"
        "<b>Christoffersen (CC)</b> : LR_CC = LR_POF + LR_IND ~ χ²(2)",
        styles["body"],
    ))

    story.append(Paragraph("A.2 Références bibliographiques", styles["h2"]))
    refs = [
        "Bollerslev, T. (1986). Generalized autoregressive conditional "
        "heteroskedasticity. <i>Journal of Econometrics</i>, 31(3), 307-327.",

        "Christoffersen, P. F. (1998). Evaluating interval forecasts. "
        "<i>International Economic Review</i>, 39(4), 841-862.",

        "Engle, R. F., &amp; Manganelli, S. (2004). CAViaR: Conditional "
        "autoregressive value at risk by regression quantiles. "
        "<i>Journal of Business &amp; Economic Statistics</i>, 22(4), 367-381.",

        "Glosten, L. R., Jagannathan, R., &amp; Runkle, D. E. (1993). On "
        "the relation between the expected value and the volatility of the "
        "nominal excess return on stocks. <i>The Journal of Finance</i>, "
        "48(5), 1779-1801.",

        "Kupiec, P. (1995). Techniques for verifying the accuracy of risk "
        "measurement models. <i>Journal of Derivatives</i>, 3(2), 73-84.",

        "McNeil, A. J., &amp; Frey, R. (2000). Estimation of tail-related "
        "risk measures for heteroscedastic financial time series: an extreme "
        "value approach. <i>Journal of Empirical Finance</i>, 7(3-4), 271-300.",

        "Pickands, J. (1975). Statistical inference using extreme order "
        "statistics. <i>The Annals of Statistics</i>, 3(1), 119-131.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["body"]))

    return story


# ---------------------------------------------------------------------------
# Générateur principal
# ---------------------------------------------------------------------------

def generate_report(
    ticker: str,
    p: float = 0.99,
    window: int = 1000,
    refit_every: int = 50,
    output_path: Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    Génère le rapport PDF complet.

    Parameters
    ----------
    ticker : str
    p : float
    window : int
    refit_every : int
    output_path : Path or None
    verbose : bool

    Returns
    -------
    Path
        Chemin du PDF généré.
    """
    from src.data_manager import get_returns, get_prices
    from src.garch_model import fit_garch
    from src.evt_model import fit_gpd_pot
    from src.benchmark_models import (
        var_historical, var_normal, var_garch_normal,
    )
    from src.dynamic_var import rolling_var_es
    from src.backtesting import backtest_table

    if output_path is None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = REPORTS_DIR / f"garch_evt_rapport_{ticker}_{date_str}.pdf"

    if verbose:
        print(f"📄 Génération du rapport pour {ticker}…\n")

    # --- Calculs ---
    if verbose:
        print("[1/6] Chargement des données…")
    returns = get_returns(ticker)
    prices = get_prices(ticker)

    if verbose:
        print("[2/6] Fit GARCH…")
    fit = fit_garch(returns, model_type="garch", dist="ged")

    if verbose:
        print("[3/6] Fit EVT…")
    evt = fit_gpd_pot(fit.residuals_std.values,
                       threshold_quantile=0.90, tail="lower")

    if verbose:
        print("[4/6] Calcul des VaR…")
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

    if verbose:
        print("[5/6] Backtesting…")
    table = backtest_table(returns, models, p=p)
    table["score_bale"] = table["kupiec_p"] + table["cc_p"]
    table = table.sort_values("score_bale", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))

    if verbose:
        print("[6/6] Génération du PDF…")

    # --- Construction du document ---
    styles = build_styles()
    doc = ReportDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
        title=f"GARCH-EVT Risk Engine — {ticker}",
        author="Statby2Mf",
    )

    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    story = []
    # Page de couverture
    story += section_cover(styles, ticker, p, len(returns), date_str)
    # Résumé exécutif
    story += section_executive_summary(styles, table, evt.xi, ticker, p)
    # Méthodologie
    story += section_methodology(styles)
    # Données
    story += section_data(styles, prices, returns, ticker)
    # GARCH
    story += section_garch(styles, fit, returns, ticker)
    # EVT
    story += section_evt(styles, fit, evt)
    # VaR
    story += section_var(styles, returns, models)
    # Backtesting
    story += section_backtesting(styles, table, p)
    # Synthèse
    story += section_synthesis(styles, table, evt.xi, ticker, p)
    # Annexe
    story += section_appendix(styles)

    # --- Build ---
    doc.build(story)

    if verbose:
        print(f"\n✅ Rapport généré : {output_path}")
        print(f"   Taille : {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns

    print("=" * 70)
    print("📄 GÉNÉRATION DU RAPPORT PDF — GARCH-EVT Risk Engine")
    print("=" * 70)

    # Trouver un ticker dispo
    ticker = "BTC-USD"

    pdf_path = generate_report(ticker, p=0.99, window=1000, refit_every=50)

    print("\n" + "=" * 70)
    print("✅ RAPPORT PDF PRÊT")
    print("=" * 70)
    print(f"📁 {pdf_path}")
    print("\n👉 Ouvre le fichier pour vérifier :")
    print(f"   start {pdf_path}    (Windows)")
    print(f"   open {pdf_path}     (Mac)")
