"""
Modèle EVT (Extreme Value Theory) via la méthode POT.

Applique la théorie des valeurs extrêmes aux résidus standardisés z_t
issus de GARCH (voir garch_model.py).

Théorème fondateur (Pickands-Balkema-de Haan, 1974-75) :
    Au-delà d'un seuil élevé u, la distribution des excès
    (X - u | X > u) converge vers une Generalized Pareto Distribution (GPD) :
        F_u(y) = 1 - (1 + ξ·y/β)^(-1/ξ)

Références :
    - Pickands, J. (1975). Statistical inference using extreme order statistics.
    - Balkema, A. A., & de Haan, L. (1974). Residual life time at great age.
    - McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures...

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import genpareto


# Type de queue
TailType = Literal["lower", "upper"]


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass
class EVTFit:
    """
    Résultat complet d'un ajustement GPD via POT.

    Attributes
    ----------
    xi : float
        Paramètre de forme (shape). ξ > 0 → queue épaisse (Fréchet).
    beta : float
        Paramètre d'échelle (scale). β > 0.
    threshold : float
        Seuil u (dans l'échelle des PERTES, i.e., -z_t pour tail='lower').
    n_exceed : int
        Nombre d'excès (observations au-delà du seuil).
    n_total : int
        Nombre total d'observations.
    threshold_quantile : float
        Quantile utilisé pour fixer u (ex : 0.90).
    tail : str
        'lower' (pertes) ou 'upper' (gains).
    log_likelihood : float
        Log-vraisemblance au point optimal.
    aic : float
        Akaike Information Criterion.
    """
    xi: float
    beta: float
    threshold: float
    n_exceed: int
    n_total: int
    threshold_quantile: float
    tail: str
    log_likelihood: float
    aic: float

    @property
    def exceedance_rate(self) -> float:
        """Fréquence d'excès : N_u / n."""
        return self.n_exceed / self.n_total

    @property
    def tail_type(self) -> str:
        """Classification de la queue selon ξ."""
        if self.xi > 0.05:
            return "Fréchet (queue épaisse)"
        elif self.xi < -0.05:
            return "Weibull (queue bornée)"
        else:
            return "Gumbel (queue exponentielle)"

    def summary(self) -> str:
        """Résumé texte lisible."""
        return (
            f"EVT — GPD sur queue {self.tail} (POT)\n"
            f"  ξ (shape)        : {self.xi:.4f}  → {self.tail_type}\n"
            f"  β (scale)        : {self.beta:.4f}\n"
            f"  Seuil u          : {self.threshold:.4f}\n"
            f"  Quantile seuil   : {self.threshold_quantile:.2%}\n"
            f"  Excès observés   : {self.n_exceed} / {self.n_total} "
            f"({self.exceedance_rate:.2%})\n"
            f"  Log-vraisemblance: {self.log_likelihood:.2f}\n"
            f"  AIC              : {self.aic:.2f}"
        )


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def to_losses(z: np.ndarray | pd.Series, tail: TailType = "lower") -> np.ndarray:
    """
    Transforme les résidus standardisés z_t en "pertes" positives.

    - tail='lower' : on s'intéresse aux pertes → renvoie -z_t
      (les grosses pertes correspondent aux z_t très négatifs)
    - tail='upper' : on s'intéresse aux gains extrêmes → renvoie z_t

    Parameters
    ----------
    z : array-like
        Résidus standardisés (peuvent contenir des valeurs négatives).
    tail : {'lower', 'upper'}

    Returns
    -------
    np.ndarray
        Pertes positives (pour tail='lower') ou gains (pour tail='upper').
    """
    z_arr = np.asarray(z, dtype=float)
    if tail == "lower":
        return -z_arr
    return z_arr


# ---------------------------------------------------------------------------
# Ajustement GPD
# ---------------------------------------------------------------------------

def fit_gpd_pot(
    z: np.ndarray | pd.Series,
    threshold_quantile: float = 0.90,
    tail: TailType = "lower",
) -> EVTFit:
    """
    Ajuste une GPD par la méthode Peaks-Over-Threshold.

    Étapes :
        1. Transforme z en pertes (si tail='lower')
        2. Choisit le seuil u = quantile(threshold_quantile)
        3. Extrait les excès (pertes > u) - u
        4. Fit une GPD(ξ, β) par maximum de vraisemblance

    Parameters
    ----------
    z : array-like
        Résidus standardisés issus de GARCH (typiquement ~ 2000 obs).
    threshold_quantile : float
        Quantile pour le seuil u. Typiquement 0.90 à 0.95.
        Plus haut = moins d'excès mais plus "extrême".
    tail : {'lower', 'upper'}
        'lower' = pertes (recommandé pour la VaR), 'upper' = gains.

    Returns
    -------
    EVTFit

    Raises
    ------
    ValueError
        Si moins de 20 excès (fit GPD instable).
    """
    # 1. Transformation
    losses = to_losses(z, tail=tail)
    n_total = len(losses)

    # 2. Seuil
    u = float(np.quantile(losses, threshold_quantile))

    # 3. Excès
    exceedances = losses[losses > u] - u
    n_exceed = len(exceedances)

    if n_exceed < 20:
        raise ValueError(
            f"Pas assez d'excès ({n_exceed} < 20). "
            f"Baisse threshold_quantile (actuel : {threshold_quantile})."
        )

    # 4. Fit GPD via scipy
    # floc=0 : on force loc=0 (les excès sont ≥ 0 par construction)
    xi, loc, beta = genpareto.fit(exceedances, floc=0)

    # Log-vraisemblance
    log_lik = float(np.sum(genpareto.logpdf(exceedances, xi, loc=0, scale=beta)))
    # AIC = -2·logL + 2·k, avec k=2 paramètres (xi, beta)
    aic = float(-2 * log_lik + 2 * 2)

    return EVTFit(
        xi=float(xi),
        beta=float(beta),
        threshold=u,
        n_exceed=n_exceed,
        n_total=n_total,
        threshold_quantile=threshold_quantile,
        tail=tail,
        log_likelihood=log_lik,
        aic=aic,
    )


# ---------------------------------------------------------------------------
# Quantiles et probabilités
# ---------------------------------------------------------------------------

def gpd_quantile(evt: EVTFit, p: float) -> float:
    """
    Quantile de la GPD au niveau de probabilité p (p proche de 1).

    Formule (McNeil & Frey 2000, eq. 15) :
        q_p = u + (β/ξ) · [ ((n/N_u)·(1-p))^(-ξ) - 1 ]

    Cas limite ξ → 0 :
        q_p = u + β · ln((n/N_u) / (1-p))

    ⚠️ Convention retenue : n/N_u (PAS N_u/n) — vérifiée par le fait que
    VaR_99% > VaR_95% quand ξ > 0.

    Parameters
    ----------
    evt : EVTFit
    p : float
        Niveau (ex : 0.99, 0.999). Doit être > threshold_quantile.

    Returns
    -------
    float
        Quantile dans l'échelle des PERTES (valeur positive).
    """
    if p <= evt.threshold_quantile:
        raise ValueError(
            f"p={p} doit être > threshold_quantile={evt.threshold_quantile}"
        )

    xi, beta, u = evt.xi, evt.beta, evt.threshold
    n, Nu = evt.n_total, evt.n_exceed

    ratio = (n / Nu) * (1 - p)   # ← n/N_u (inverse !)

    if abs(xi) < 1e-6:
        # Cas Gumbel : limite ξ → 0
        return u + beta * np.log(1 / ratio)

    return u + (beta / xi) * (ratio ** (-xi) - 1)


def gpd_es(evt: EVTFit, p: float) -> float:
    """
    Expected Shortfall (ES) au niveau p via la GPD.

    Formule (McNeil & Frey 2000, eq. 16) :
        ES_p = VaR_p / (1 - ξ) + (β - ξ·u) / (1 - ξ)

    Valide seulement si ξ < 1 (sinon ES = ∞).

    Parameters
    ----------
    evt : EVTFit
    p : float
        Niveau de probabilité (ex : 0.99).

    Returns
    -------
    float
        ES dans l'échelle des pertes (valeur positive).
    """
    if evt.xi >= 1:
        return np.inf
    var_p = gpd_quantile(evt, p)
    return (var_p + evt.beta - evt.xi * evt.threshold) / (1 - evt.xi)


# ---------------------------------------------------------------------------
# Diagnostic — choix du seuil
# ---------------------------------------------------------------------------

def mean_residual_life(
    z: np.ndarray | pd.Series,
    tail: TailType = "lower",
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """
    Mean Residual Life plot — outil VISUEL pour choisir le seuil.

    Pour chaque seuil u candidat, calcule : E[X - u | X > u].
    Théoriquement, cette fonction est LINÉAIRE en u au-delà du seuil optimal.
    On cherche donc la plage où le graphique est à peu près linéaire.

    Parameters
    ----------
    z : array-like
    tail : {'lower', 'upper'}
    n_thresholds : int
        Nombre de seuils testés.

    Returns
    -------
    pd.DataFrame
        Colonnes : threshold, mean_excess, n_exceed, conf_low, conf_high.
    """
    losses = to_losses(z, tail=tail)
    # Seuils candidats : entre 50e et 99e percentile
    thresholds = np.quantile(losses, np.linspace(0.5, 0.99, n_thresholds))

    rows = []
    for u in thresholds:
        excess = losses[losses > u] - u
        n_exc = len(excess)
        if n_exc < 10:
            continue
        mean_exc = excess.mean()
        std_exc = excess.std()
        # IC approximatif à 95% (théorème central limite)
        se = std_exc / np.sqrt(n_exc)
        rows.append({
            "threshold": u,
            "mean_excess": mean_exc,
            "n_exceed": n_exc,
            "conf_low": mean_exc - 1.96 * se,
            "conf_high": mean_exc + 1.96 * se,
        })
    return pd.DataFrame(rows)


def compare_thresholds(
    z: np.ndarray | pd.Series,
    quantiles: tuple[float, ...] = (0.85, 0.90, 0.95),
    tail: TailType = "lower",
) -> pd.DataFrame:
    """
    Ajuste la GPD à plusieurs seuils et compare les paramètres (ξ, β).

    Un bon seuil donne des ξ stables. Si ξ varie beaucoup → seuil mal choisi.

    Returns
    -------
    pd.DataFrame
        Colonnes : quantile, threshold, xi, beta, n_exceed, log_likelihood, aic.
    """
    rows = []
    for q in quantiles:
        try:
            evt = fit_gpd_pot(z, threshold_quantile=q, tail=tail)
            rows.append({
                "quantile": q,
                "threshold": evt.threshold,
                "xi": evt.xi,
                "beta": evt.beta,
                "n_exceed": evt.n_exceed,
                "log_likelihood": evt.log_likelihood,
                "aic": evt.aic,
            })
        except Exception as e:
            rows.append({
                "quantile": q,
                "threshold": np.nan,
                "xi": np.nan,
                "beta": np.nan,
                "n_exceed": np.nan,
                "log_likelihood": np.nan,
                "aic": np.nan,
                "error": str(e)[:40],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns
    from src.garch_model import fit_garch

    print("🔍 Test du module EVT (GPD / POT)\n")

    # 1. On récupère les résidus standardisés
    btc = get_returns("BTC-USD")
    print("📊 Ajustement GARCH-GED (meilleur AIC sur BTC)…")
    fit = fit_garch(btc, model_type="garch", dist="ged")
    z = fit.residuals_std.values
    print(f"   Résidus z_t : {len(z)} observations\n")

    # 2. Ajustement EVT
    print("=" * 70)
    print("📉 Ajustement GPD sur les PERTES (tail='lower') — seuil 90%")
    print("=" * 70)
    evt = fit_gpd_pot(z, threshold_quantile=0.90, tail="lower")
    print(evt.summary())

    # 3. Quantiles extrêmes
    print("\n🎯 Quantiles EVT (dans l'échelle des pertes standardisées) :")
    for p in [0.95, 0.99, 0.995, 0.999]:
        try:
            q = gpd_quantile(evt, p)
            es = gpd_es(evt, p)
            print(f"   p = {p:.3%}  →  VaR_std = {q:.4f}, ES_std = {es:.4f}")
        except ValueError as e:
            print(f"   p = {p:.3%}  →  {e}")

    # 4. Comparaison de seuils
    print("\n" + "=" * 70)
    print("🔬 Robustesse au choix du seuil")
    print("=" * 70)
    cmp = compare_thresholds(z, quantiles=(0.85, 0.90, 0.95))
    print(cmp.to_string(index=False))

    # 5. Mean Residual Life
    print("\n" + "=" * 70)
    print("📈 Mean Residual Life (extrait)")
    print("=" * 70)
    mrl = mean_residual_life(z, tail="lower", n_thresholds=10)
    print(mrl.round(3).to_string(index=False))
