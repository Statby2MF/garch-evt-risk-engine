"""
Modèles GARCH pour la volatilité conditionnelle.

Ce module supporte DEUX types de modèles :
    - GARCH(1,1)   : modèle symétrique classique
    - GJR-GARCH    : version asymétrique (effet de levier)

Il fournit :
    - Les paramètres estimés (ω, α, β, γ pour GJR, μ)
    - La volatilité conditionnelle σ_t
    - Les résidus standardisés z_t = ε_t / σ_t  → INPUT pour l'EVT
    - Les prévisions de volatilité σ_{T+h}

Références :
    - Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
    - Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between
      the expected value and the volatility of the nominal excess return on stocks.
    - McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures...

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model


# Types autorisés
ModelType = Literal["garch", "gjr"]


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass
class GarchFit:
    """
    Résultat complet d'un ajustement GARCH ou GJR-GARCH.

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
        Coefficient ARCH (réaction aux chocs, symétrique).
    gamma : float
        Coefficient d'asymétrie GJR (0 si model_type='garch').
        γ > 0 → effet de levier (chocs négatifs amplifient plus la vol).
    beta : float
        Coefficient GARCH (persistance).
    model_type : str
        'garch' ou 'gjr'.
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
    gamma: float
    beta: float
    model_type: str
    dist: str
    aic: float
    bic: float
    log_likelihood: float
    model_result: object = field(repr=False)

    @property
    def persistence(self) -> float:
        """
        Persistance de la volatilité.

        - Pour GARCH    : α + β
        - Pour GJR-GARCH: α + γ/2 + β
          (γ/2 car les chocs négatifs sont, en moyenne, 1 sur 2)
        """
        if self.model_type == "gjr":
            return self.alpha + self.gamma / 2 + self.beta
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """
        Volatilité de long terme (inconditionnelle) :
            σ_LR = sqrt( ω / (1 - persistence) )
        """
        denom = 1 - self.persistence
        if denom <= 0:
            return np.nan
        return float(np.sqrt(self.omega / denom))

    @property
    def leverage_effect(self) -> bool:
        """True si GJR-GARCH avec γ significativement positif (> 0)."""
        return self.model_type == "gjr" and self.gamma > 0

    def summary(self) -> str:
        """Résumé texte lisible."""
        model_label = "GJR-GARCH(1,1,1)" if self.model_type == "gjr" else "GARCH(1,1)"
        lines = [
            f"{model_label} — {self.dist}",
            f"  μ (mean)         : {self.mu:.4f}",
            f"  ω (omega)        : {self.omega:.6f}",
            f"  α (alpha)        : {self.alpha:.4f}",
        ]
        if self.model_type == "gjr":
            lines.append(f"  γ (gamma)        : {self.gamma:.4f}"
                         f"  {'(effet de levier ✓)' if self.leverage_effect else ''}")
        lines.extend([
            f"  β (beta)         : {self.beta:.4f}",
            f"  Persistance      : {self.persistence:.4f}  "
            f"({'stationnaire' if self.persistence < 1 else 'NON stationnaire !'})",
            f"  σ long terme     : {self.long_run_vol:.4f}%",
            f"  AIC / BIC        : {self.aic:.2f} / {self.bic:.2f}",
            f"  Log-vraisemblance: {self.log_likelihood:.2f}",
            f"  N observations   : {len(self.returns)}",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ajustement
# ---------------------------------------------------------------------------

def fit_garch(
    returns: pd.Series,
    model_type: ModelType = "garch",
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "skewt",
    mean: str = "Constant",
) -> GarchFit:
    """
    Ajuste un modèle GARCH(p,q) ou GJR-GARCH(p,o,q).

    Parameters
    ----------
    returns : pd.Series
        Log-rendements en pourcentage (× 100). PAS de NaN.
    model_type : {'garch', 'gjr'}
        - 'garch' : GARCH(p,q) symétrique
        - 'gjr'   : GJR-GARCH(p,o,q) avec terme d'asymétrie
    p : int
        Ordre ARCH (défaut : 1).
    o : int
        Ordre du terme asymétrique (GJR uniquement, défaut : 1).
    q : int
        Ordre GARCH (défaut : 1).
    dist : str
        Distribution de z_t : 'normal', 't', 'skewt', 'ged'.
    mean : str
        Modèle de moyenne : 'Constant', 'Zero', 'AR', 'ARX'.

    Returns
    -------
    GarchFit
        Résultat complet avec résidus standardisés prêts pour l'EVT.

    Raises
    ------
    ValueError
        Si les rendements contiennent des NaN, sont trop courts,
        ou si model_type est invalide.
    """
    # --- Validation ---
    if returns.isna().any():
        raise ValueError("Les rendements contiennent des NaN. Nettoie d'abord.")
    if len(returns) < 100:
        raise ValueError(f"Pas assez d'observations ({len(returns)} < 100).")
    if model_type not in ("garch", "gjr"):
        raise ValueError(f"model_type doit être 'garch' ou 'gjr', reçu '{model_type}'.")

    # --- Construction du modèle ---
    if model_type == "gjr":
        am = arch_model(
            returns,
            vol="Garch",
            p=p,
            o=o,      # ← terme asymétrique GJR
            q=q,
            dist=dist,
            mean=mean,
            rescale=False,
        )
    else:  # garch
        am = arch_model(
            returns,
            vol="Garch",
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            rescale=False,
        )

    # --- Fit ---
    res = am.fit(disp="off", show_warning=False)

    # --- Extraction ---
    cond_vol = res.conditional_volatility
    residuals = res.resid
    resid_std = (residuals / cond_vol).dropna()

    # --- Paramètres (avec .get pour gérer l'absence de gamma) ---
    params = res.params

    # Gestion robuste des noms de paramètres selon le modèle
    alpha_key = f"alpha[{1}]" if p >= 1 else None
    beta_key = f"beta[{1}]" if q >= 1 else None
    gamma_key = f"gamma[{1}]" if model_type == "gjr" and o >= 1 else None

    fit = GarchFit(
        returns=returns,
        conditional_vol=cond_vol,
        residuals=residuals,
        residuals_std=resid_std,
        mu=float(params.get("mu", 0.0)),
        omega=float(params.get("omega", np.nan)),
        alpha=float(params.get(alpha_key, 0.0)) if alpha_key else 0.0,
        gamma=float(params.get(gamma_key, 0.0)) if gamma_key else 0.0,
        beta=float(params.get(beta_key, 0.0)) if beta_key else 0.0,
        model_type=model_type,
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

    But : vérifier que le modèle a bien capturé toute la structure
          de dépendance dans la variance.

    H0 : pas d'autocorrélation dans les z_t² jusqu'au lag `lags`.
    → Si p-value > 0.05, le modèle est bien spécifié.

    Parameters
    ----------
    fit : GarchFit
    lags : int

    Returns
    -------
    dict
        {'stat': ..., 'pvalue': ..., 'lags': ..., 'verdict': str}
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    squared = (fit.residuals_std ** 2).dropna()
    result = acorr_ljungbox(squared, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    pval = float(result["lb_pvalue"].iloc[0])

    return {
        "stat": stat,
        "pvalue": pval,
        "lags": lags,
        "verdict": "✅ OK" if pval > 0.05 else "⚠️ structure résiduelle",
    }


def compare_models(
    returns: pd.Series,
    model_types: tuple[str, ...] = ("garch", "gjr"),
    distributions: tuple[str, ...] = ("normal", "t", "skewt", "ged"),
) -> pd.DataFrame:
    """
    Benchmark croisé : {GARCH, GJR} × {normal, t, skewt, ged}.

    Utile pour JUSTIFIER ton choix de modèle final dans le mémoire.

    Parameters
    ----------
    returns : pd.Series
    model_types : tuple of str
    distributions : tuple of str

    Returns
    -------
    pd.DataFrame
        Tableau trié par AIC croissant.
        Colonnes : model, dist, log_likelihood, aic, bic, alpha, gamma, beta,
                   persistence, leverage.
    """
    rows = []
    for mt in model_types:
        for dist in distributions:
            try:
                fit = fit_garch(returns, model_type=mt, dist=dist)
                rows.append({
                    "model": mt,
                    "dist": dist,
                    "log_likelihood": fit.log_likelihood,
                    "aic": fit.aic,
                    "bic": fit.bic,
                    "alpha": fit.alpha,
                    "gamma": fit.gamma,
                    "beta": fit.beta,
                    "persistence": fit.persistence,
                    "leverage": fit.leverage_effect,
                })
            except Exception as e:
                rows.append({
                    "model": mt,
                    "dist": dist,
                    "log_likelihood": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "alpha": np.nan,
                    "gamma": np.nan,
                    "beta": np.nan,
                    "persistence": np.nan,
                    "leverage": False,
                    "error": str(e)[:60],
                })
    df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns

    print("🔍 Test du module GARCH / GJR-GARCH\n")
    btc = get_returns("BTC-USD")

    # --- GARCH(1,1) ---
    print("=" * 70)
    print("📊 [1/2] GARCH(1,1)-skewt sur BTC")
    print("=" * 70)
    fit_garch_sym = fit_garch(btc, model_type="garch", dist="skewt")
    print(fit_garch_sym.summary())

    lb = ljung_box_test(fit_garch_sym, lags=10)
    print(f"\n📉 Ljung-Box (lag=10) : stat={lb['stat']:.2f}, "
          f"p={lb['pvalue']:.4f} → {lb['verdict']}")

    # --- GJR-GARCH ---
    print("\n" + "=" * 70)
    print("📊 [2/2] GJR-GARCH(1,1,1)-skewt sur BTC")
    print("=" * 70)
    fit_gjr = fit_garch(btc, model_type="gjr", dist="skewt")
    print(fit_gjr.summary())

    lb_gjr = ljung_box_test(fit_gjr, lags=10)
    print(f"\n📉 Ljung-Box (lag=10) : stat={lb_gjr['stat']:.2f}, "
          f"p={lb_gjr['pvalue']:.4f} → {lb_gjr['verdict']}")

    # --- Comparaison globale ---
    print("\n" + "=" * 70)
    print("🏆 BENCHMARK : {GARCH, GJR} × {normal, t, skewt, ged}")
    print("=" * 70)
    comparison = compare_models(btc)
    print(comparison.to_string(index=False))

    # --- Meilleur modèle ---
    best = comparison.iloc[0]
    print(f"\n🥇 MEILLEUR MODÈLE (AIC) : {best['model'].upper()} — {best['dist']}")
    print(f"   AIC = {best['aic']:.2f}")
    if best["model"] == "gjr":
        print(f"   γ = {best['gamma']:.4f} → "
              f"{'effet de levier confirmé ✓' if best['gamma'] > 0 else 'pas d effet de levier'}")
