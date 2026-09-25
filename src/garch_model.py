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
    - Bollerslev, T. (1986).
    - Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993).
    - McNeil, A. J., & Frey, R. (2000).

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model


ModelType = Literal["garch", "gjr"]


@dataclass
class GarchFit:
    """Résultat complet d'un ajustement GARCH ou GJR-GARCH."""

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
        """Persistance de la volatilité (α + β ou α + γ/2 + β pour GJR)."""
        if self.model_type == "gjr":
            return self.alpha + self.gamma / 2 + self.beta
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """Volatilité inconditionnelle, ou NaN si quasi-IGARCH."""
        TOL = 1e-4
        denom = 1 - self.persistence
        if denom < TOL:
            return np.nan
        return float(np.sqrt(self.omega / denom))

    @property
    def leverage_effect(self) -> bool:
        """True si GJR avec γ > 0."""
        return self.model_type == "gjr" and self.gamma > 0

    def summary(self) -> str:
        """Résumé texte lisible."""
        model_label = "GJR-GARCH(1,1,1)" if self.model_type == "gjr" else "GARCH(1,1)"
        is_stationary = self.persistence < 1 - 1e-4
        lines = [
            f"{model_label} — {self.dist}",
            f"  μ (mean)         : {self.mu:.4f}",
            f"  ω (omega)        : {self.omega:.6f}",
            f"  α (alpha)        : {self.alpha:.4f}",
        ]
        if self.model_type == "gjr":
            tag = "  (effet de levier ✓)" if self.leverage_effect else ""
            lines.append(f"  γ (gamma)        : {self.gamma:.4f}{tag}")
        lines.extend([
            f"  β (beta)         : {self.beta:.4f}",
            f"  Persistance      : {self.persistence:.4f}  "
            f"({'stationnaire' if is_stationary else 'quasi-IGARCH'})",
            f"  σ long terme     : {self.long_run_vol:.4f}%",
            f"  AIC / BIC        : {self.aic:.2f} / {self.bic:.2f}",
            f"  Log-vraisemblance: {self.log_likelihood:.2f}",
            f"  N observations   : {len(self.returns)}",
        ])
        return "\n".join(lines)


def fit_garch(
    returns: pd.Series,
    model_type: ModelType = "garch",
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "skewt",
    mean: str = "Constant",
) -> GarchFit:
    """Ajuste un GARCH(p,q) ou un GJR-GARCH(p,o,q)."""
    if returns.isna().any():
        raise ValueError("Les rendements contiennent des NaN.")
    if len(returns) < 100:
        raise ValueError(f"Pas assez d'observations ({len(returns)} < 100).")
    if model_type not in ("garch", "gjr"):
        raise ValueError(f"model_type doit être 'garch' ou 'gjr', reçu '{model_type}'.")

    if model_type == "gjr":
        am = arch_model(returns, vol="Garch", p=p, o=o, q=q,
                        dist=dist, mean=mean, rescale=False)
    else:
        am = arch_model(returns, vol="Garch", p=p, q=q,
                        dist=dist, mean=mean, rescale=False)

    res = am.fit(disp="off", show_warning=False)

    cond_vol = res.conditional_volatility
    residuals = res.resid
    resid_std = (residuals / cond_vol).dropna()

    params = res.params
    alpha_key = f"alpha[{1}]" if p >= 1 else None
    beta_key = f"beta[{1}]" if q >= 1 else None
    gamma_key = f"gamma[{1}]" if model_type == "gjr" and o >= 1 else None

    return GarchFit(
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


def forecast_volatility(fit: GarchFit, horizon: int = 1) -> np.ndarray:
    """Prévoit σ_{T+1}, ..., σ_{T+h}."""
    fct = fit.model_result.forecast(horizon=horizon, reindex=False)
    return np.sqrt(fct.variance.values[-1, :])


def ljung_box_test(fit: GarchFit, lags: int = 10) -> dict:
    """Test de Ljung-Box sur les résidus standardisés au carré."""
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
    """Benchmark croisé : {GARCH, GJR} × {normal, t, skewt, ged}."""
    rows = []
    for mt in model_types:
        for dist in distributions:
            try:
                fit = fit_garch(returns, model_type=mt, dist=dist)
                rows.append({
                    "model": mt, "dist": dist,
                    "log_likelihood": fit.log_likelihood,
                    "aic": fit.aic, "bic": fit.bic,
                    "alpha": fit.alpha, "gamma": fit.gamma, "beta": fit.beta,
                    "persistence": fit.persistence,
                    "leverage": fit.leverage_effect,
                })
            except Exception as e:
                rows.append({
                    "model": mt, "dist": dist,
                    "log_likelihood": np.nan, "aic": np.nan, "bic": np.nan,
                    "alpha": np.nan, "gamma": np.nan, "beta": np.nan,
                    "persistence": np.nan, "leverage": False,
                    "error": str(e)[:60],
                })
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


if __name__ == "__main__":
    from src.data_manager import get_returns

    print("🔍 Test du module GARCH / GJR-GARCH\n")
    btc = get_returns("BTC-USD")

    print("=" * 70)
    print("📊 [1/2] GARCH(1,1)-skewt sur BTC")
    print("=" * 70)
    fit_sym = fit_garch(btc, model_type="garch", dist="skewt")
    print(fit_sym.summary())
    lb = ljung_box_test(fit_sym)
    print(f"\n📉 Ljung-Box (lag=10) : stat={lb['stat']:.2f}, "
          f"p={lb['pvalue']:.4f} → {lb['verdict']}")

    print("\n" + "=" * 70)
    print("📊 [2/2] GJR-GARCH(1,1,1)-skewt sur BTC")
    print("=" * 70)
    fit_gjr = fit_garch(btc, model_type="gjr", dist="skewt")
    print(fit_gjr.summary())
    lb_gjr = ljung_box_test(fit_gjr)
    print(f"\n📉 Ljung-Box (lag=10) : stat={lb_gjr['stat']:.2f}, "
          f"p={lb_gjr['pvalue']:.4f} → {lb_gjr['verdict']}")

    print("\n" + "=" * 70)
    print("🏆 BENCHMARK : {GARCH, GJR} × {normal, t, skewt, ged}")
    print("=" * 70)
    comparison = compare_models(btc)
    print(comparison.to_string(index=False))

    best = comparison.iloc[0]
    print(f"\n🥇 MEILLEUR MODÈLE (AIC) : {best['model'].upper()} — {best['dist']}")
    print(f"   AIC = {best['aic']:.2f}")
