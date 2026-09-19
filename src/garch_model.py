"""
Modèle GARCH(1,1) pour la volatilité conditionnelle.

Ce module ajuste un modèle GARCH(1,1) sur les log-rendements d'un actif
et fournit :
    - Les paramètres estimés (ω, α, β, μ)
    - La volatilité conditionnelle σ_t
    - Les résidus standardisés z_t = ε_t / σ_t  → INPUT pour l'EVT
    - Les prévisions de volatilité σ_{T+h}

Références :
    - Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
    - McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures...

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass
class GarchFit:
    """
    Résultat complet d'un ajustement GARCH.

    Attributes
    ----------
    returns : pd.Series
        Log-rendements originaux (en %).
    conditional_vol : pd.Series
        σ_t — volatilité conditionnelle (en %, même échelle que returns).
    residuals : pd.Series
        ε_t = r_t - μ — résidus bruts.
    residuals_std : pd.Series
        z_t = ε_t / σ_t — résidus STANDARDISÉS (iid).
        → C'est CE vecteur qu'on donnera à l'EVT.
    mu : float
        Rendement moyen estimé.
    omega : float
        Constante de l'équation de variance.
    alpha : float
        Coefficient ARCH (réaction aux chocs).
    beta : float
        Coefficient GARCH (persistance).
    dist : str
        Distribution utilisée pour z_t.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    log_likelihood : float
        Log-vraisemblance au point optimal.
    model_result : object
        Objet `arch.univariate.base.ARCHModelResult` complet (pour diagnostic).
    """

    returns: pd.Series
    conditional_vol: pd.Series
    residuals: pd.Series
    residuals_std: pd.Series
    mu: float
    omega: float
    alpha: float
    beta: float
    dist: str
    aic: float
    bic: float
    log_likelihood: float
    model_result: object = field(repr=False)

    @property
    def persistence(self) -> float:
        """α + β — mesure la persistance de la volatilité. Doit être < 1."""
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """
        Volatilité de long terme (inconditionnelle) :
            σ_LR = sqrt( ω / (1 - α - β) )
        """
        denom = 1 - self.persistence
        if denom <= 0:
            return np.nan
        return float(np.sqrt(self.omega / denom))

    def summary(self) -> str:
        """Résumé texte lisible."""
        return (
            f"GARCH(1,1) — {self.dist}\n"
            f"  μ (mean)         : {self.mu:.4f}\n"
            f"  ω (omega)        : {self.omega:.6f}\n"
            f"  α (alpha)        : {self.alpha:.4f}\n"
            f"  β (beta)         : {self.beta:.4f}\n"
            f"  α + β            : {self.persistence:.4f}  "
            f"({'stationnaire' if self.persistence < 1 else 'NON stationnaire !'})\n"
            f"  σ long terme     : {self.long_run_vol:.4f}%\n"
            f"  AIC / BIC        : {self.aic:.2f} / {self.bic:.2f}\n"
            f"  Log-vraisemblance: {self.log_likelihood:.2f}\n"
            f"  N observations   : {len(self.returns)}"
        )


# ---------------------------------------------------------------------------
# Ajustement
# ---------------------------------------------------------------------------

def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "skewt",
    mean: str = "Constant",
) -> GarchFit:
    """
    Ajuste un modèle GARCH(p, q) sur les log-rendements.

    Parameters
    ----------
    returns : pd.Series
        Log-rendements en pourcentage (× 100). PAS de NaN.
    p : int
        Ordre ARCH (défaut : 1).
    q : int
        Ordre GARCH (défaut : 1).
    dist : str
        Distribution de z_t. Options : 'normal', 't', 'skewt', 'ged'.
        → 'skewt' (skewed Student-t) recommandé pour crypto.
    mean : str
        Modèle de moyenne : 'Constant', 'Zero', 'AR', 'ARX'.

    Returns
    -------
    GarchFit
        Résultat complet avec résidus standardisés prêts pour l'EVT.

    Raises
    ------
    ValueError
        Si les rendements contiennent des NaN ou sont trop courts.
    """
    # --- Validation ---
    if returns.isna().any():
        raise ValueError("Les rendements contiennent des NaN. Nettoie d'abord.")
    if len(returns) < 100:
        raise ValueError(f"Pas assez d'observations ({len(returns)} < 100).")

    # --- Construction & fit ---
    # rescale=False : nos rendements sont déjà en % → pas besoin de re-scaler
    am = arch_model(
        returns,
        vol="Garch",
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)

    # --- Extraction ---
    cond_vol = res.conditional_volatility
    residuals = res.resid
    resid_std = residuals / cond_vol

    # Nettoyage des NaN éventuels dans les résidus standardisés
    mask = resid_std.notna()
    resid_std = resid_std[mask]

    # --- Paramètres ---
    params = res.params
    fit = GarchFit(
        returns=returns,
        conditional_vol=cond_vol,
        residuals=residuals,
        residuals_std=resid_std,
        mu=float(params.get("mu", 0.0)),
        omega=float(params.get("omega", np.nan)),
        alpha=float(params.get(f"alpha[{1}]", np.nan)) if p >= 1 else 0.0,
        beta=float(params.get(f"beta[{1}]", np.nan)) if q >= 1 else 0.0,
        dist=dist,
        aic=float(res.aic),
        bic=float(res.bic),
        log_likelihood=float(res.loglikelihood),
        model_result=res,
    )
    return fit


# ---------------------------------------------------------------------------
# Prévisions
# ---------------------------------------------------------------------------

def forecast_volatility(fit: GarchFit, horizon: int = 1) -> np.ndarray:
    """
    Prévoit σ_{T+1}, ..., σ_{T+h} (h = horizon).

    C'est la brique clé pour la VaR DYNAMIQUE :
    demain, σ change → la VaR change.

    Parameters
    ----------
    fit : GarchFit
    horizon : int
        Nombre de pas de temps à prévoir.

    Returns
    -------
    np.ndarray
        Array de taille (horizon,) contenant σ_{T+1..T+h} en %.
    """
    fct = fit.model_result.forecast(horizon=horizon, reindex=False)
    variances = fct.variance.values[-1, :]
    return np.sqrt(variances)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ljung_box_test(fit: GarchFit, lags: int = 10) -> dict:
    """
    Test de Ljung-Box sur les résidus standardisés AU CARRÉ.

    But : vérifier que le GARCH a bien capturé toute la structure
          de dépendance dans la variance.

    H0 : pas d'autocorrélation dans les z_t² jusqu'au lag `lags`.
    → Si p-value > 0.05, le modèle est bien spécifié.

    Parameters
    ----------
    fit : GarchFit
    lags : int
        Nombre de retards.

    Returns
    -------
    dict
        {'stat': ..., 'pvalue': ..., 'lags': ...}
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    squared = (fit.residuals_std ** 2).dropna()
    result = acorr_ljungbox(squared, lags=[lags], return_df=True)
    return {
        "stat": float(result["lb_stat"].iloc[0]),
        "pvalue": float(result["lb_pvalue"].iloc[0]),
        "lags": lags,
    }


def compare_distributions(
    returns: pd.Series,
    distributions: tuple[str, ...] = ("normal", "t", "skewt", "ged"),
) -> pd.DataFrame:
    """
    Ajuste GARCH(1,1) avec plusieurs distributions et compare AIC/BIC.

    Utile pour JUSTIFIER ton choix de distribution dans le mémoire.

    Parameters
    ----------
    returns : pd.Series
    distributions : tuple of str

    Returns
    -------
    pd.DataFrame
        Tableau avec colonnes : dist, log_likelihood, aic, bic, alpha, beta.
    """
    rows = []
    for dist in distributions:
        try:
            fit = fit_garch(returns, dist=dist)
            rows.append({
                "dist": dist,
                "log_likelihood": fit.log_likelihood,
                "aic": fit.aic,
                "bic": fit.bic,
                "alpha": fit.alpha,
                "beta": fit.beta,
                "persistence": fit.persistence,
            })
        except Exception as e:
            rows.append({
                "dist": dist,
                "log_likelihood": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "persistence": np.nan,
                "error": str(e)[:50],
            })
    df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns

    print("🔍 Test du module GARCH\n")
    btc = get_returns("BTC-USD")

    print(f"📊 Ajustement GARCH(1,1)-skewt sur BTC ({len(btc)} obs)…\n")
    fit = fit_garch(btc)

    print(fit.summary())

    print("\n📉 Test de Ljung-Box sur résidus² (lag=10) :")
    lb = ljung_box_test(fit, lags=10)
    verdict = "✅ OK" if lb["pvalue"] > 0.05 else "⚠️ structure résiduelle"
    print(f"   Stat = {lb['stat']:.2f}, p-value = {lb['pvalue']:.4f} → {verdict}")

    print("\n🔮 Prévision σ pour les 5 prochains jours :")
    sigmas = forecast_volatility(fit, horizon=5)
    for h, s in enumerate(sigmas, start=1):
        print(f"   σ_(T+{h}) = {s:.3f}%")

    print("\n🏆 Comparaison des distributions (AIC) :")
    comparison = compare_distributions(btc)
    print(comparison.to_string(index=False))
