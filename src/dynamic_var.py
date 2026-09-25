"""
VaR et ES dynamiques via GARCH-EVT.

Combine :
    - GARCH(1,1) / GJR-GARCH  → volatilité conditionnelle σ_t
    - EVT (GPD / POT)          → quantiles extrêmes des résidus z_t

Formules (McNeil & Frey 2000) :
    VaR_t(p) = μ + σ_t · (-q_p)          où q_p = quantile GPD des pertes
    ES_t(p)  = μ + σ_t · (-ES_p)         où ES_p = ES GPD des pertes

Méthode de backtesting :
    - Fenêtre glissante (rolling window)
    - Refit du modèle tous les `refit_every` points
    - Projection σ_t jour par jour entre les refits

Références :
    - McNeil, A. J., & Frey, R. (2000).

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.garch_model import fit_garch, GarchFit
from src.evt_model import fit_gpd_pot, gpd_quantile, gpd_es, EVTFit


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass
class DynamicRiskResult:
    """
    Résultat du calcul de VaR/ES dynamiques sur une période.

    Attributes
    ----------
    returns : pd.Series
        Rendements originaux (en %).
    var : pd.Series
        VaR conditionnelle (négative, en %, même échelle que returns).
    es : pd.Series
        ES conditionnel (négatif, en %).
    sigma : pd.Series
        σ_t conditionnelle utilisée à chaque date.
    confidence_level : float
        Niveau de confiance (ex : 0.99).
    """
    returns: pd.Series
    var: pd.Series
    es: pd.Series
    sigma: pd.Series
    confidence_level: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit en DataFrame pour affichage / sauvegarde."""
        return pd.DataFrame({
            "returns": self.returns,
            "var": self.var,
            "es": self.es,
            "sigma": self.sigma,
        })

    def violations(self) -> pd.Series:
        """
        Série booléenne : True si le rendement a dépassé la VaR
        (i.e., une violation s'est produite).
        """
        return (self.returns < self.var).astype(int).rename("violation")


# ---------------------------------------------------------------------------
# VaR/ES pour un point dans le temps
# ---------------------------------------------------------------------------

def compute_var_es_at_t(
    sigma_t: float,
    mu: float,
    evt: EVTFit,
    p: float,
) -> tuple[float, float]:
    """
    Calcule VaR_t et ES_t pour un σ_t donné.

    Parameters
    ----------
    sigma_t : float
        Volatilité conditionnelle prévue à t (même échelle que returns).
    mu : float
        Rendement moyen conditionnel (constant).
    evt : EVTFit
        Fit GPD sur les résidus standardisés (tail='lower').
    p : float
        Niveau de confiance (ex : 0.99).

    Returns
    -------
    (var_t, es_t) : tuple of float
        VaR et ES en %, négatifs (convention financière : perte = négatif).
    """
    # Quantile GPD des pertes (valeur positive, dans l'échelle des -z)
    q_std = gpd_quantile(evt, p)
    es_std = gpd_es(evt, p)

    # VaR en % (perte → signe négatif)
    var_t = mu - sigma_t * q_std
    es_t = mu - sigma_t * es_std

    return float(var_t), float(es_t)


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def rolling_var_es(
    returns: pd.Series,
    window: int = 1000,
    p: float = 0.99,
    refit_every: int = 50,
    model_type: str = "garch",
    dist: str = "ged",
    threshold_quantile: float = 0.90,
    verbose: bool = True,
) -> DynamicRiskResult:
    """
    Calcule VaR/ES dynamiques en fenêtre glissante.

    Pour chaque jour t à partir de `window` :
        1. (Optionnel) Refit GARCH + EVT sur [t-window, t[
        2. Prévoit σ_{t+1} à partir du modèle
        3. Calcule VaR_{t+1} = μ - σ_{t+1} · q_p(EVT)
        4. Stocke la VaR pour le jour t

    ⚠️ La VaR calculée pour la date T est la VaR PRÉVUE pour T+1 (ex-ante).
       Cela évite le look-ahead bias.

    Parameters
    ----------
    returns : pd.Series
        Log-rendements en %.
    window : int
        Taille de la fenêtre d'estimation. Défaut : 1000.
    p : float
        Niveau de confiance. Défaut : 0.99.
    refit_every : int
        Refit tous les combien de jours. Défaut : 50.
        (Compromis coût/précision.)
    model_type : str
        'garch' ou 'gjr'.
    dist : str
        Distribution GARCH ('normal', 't', 'skewt', 'ged').
    threshold_quantile : float
        Seuil EVT. Défaut : 0.90.
    verbose : bool
        Affiche la progression.

    Returns
    -------
    DynamicRiskResult
        var, es, sigma indexés comme returns (NaN au début).
    """
    n = len(returns)
    if n <= window:
        raise ValueError(f"Pas assez d'obs ({n}) pour window={window}")

    var_arr = np.full(n, np.nan)
    es_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)

    # Variables d'état du modèle (mises à jour lors des refits)
    current_fit: GarchFit | None = None
    current_evt: EVTFit | None = None
    current_mu: float = 0.0
    steps_since_refit: int = 10**9   # force un refit au premier tour

    # Suivi du σ pour les projections entre refits
    sigma_history: list[float] = []

    if verbose:
        print(f"🔄 Rolling backtest : window={window}, p={p:.2%}, "
              f"refit_every={refit_every}")
        print(f"   Modèle : {model_type}-{dist}, seuil EVT : "
              f"{threshold_quantile:.0%}")
        print(f"   Points à calculer : {n - window}")

    for t in range(window, n):
        # ----- Refit périodique -----
        if steps_since_refit >= refit_every:
            train = returns.iloc[t - window:t]

            try:
                current_fit = fit_garch(
                    train, model_type=model_type, dist=dist
                )
                current_evt = fit_gpd_pot(
                    current_fit.residuals_std.values,
                    threshold_quantile=threshold_quantile,
                    tail="lower",
                )
                current_mu = current_fit.mu
                sigma_history = list(current_fit.conditional_vol.values)

                if verbose and (t - window) % (refit_every * 5) == 0:
                    print(f"   [t={t}] refit OK — "
                          f"ξ={current_evt.xi:.4f}, "
                          f"σ_last={sigma_history[-1]:.3f}")

            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Refit échoué à t={t}: {e}")
                steps_since_refit = refit_every  # retry au prochain point
                continue

            steps_since_refit = 0

        # ----- Prévision de σ_{t+1} -----
        # On utilise la dernière valeur de σ connue dans le train
        sigma_t = sigma_history[-1] if sigma_history else np.nan

        # ----- VaR / ES -----
        if current_evt is not None and not np.isnan(sigma_t):
            var_t, es_t = compute_var_es_at_t(
                sigma_t=sigma_t,
                mu=current_mu,
                evt=current_evt,
                p=p,
            )
            var_arr[t] = var_t
            es_arr[t] = es_t
            sigma_arr[t] = sigma_t

        steps_since_refit += 1

    # Construction du résultat
    idx = returns.index
    result = DynamicRiskResult(
        returns=returns,
        var=pd.Series(var_arr, index=idx, name=f"VaR_{p:.0%}"),
        es=pd.Series(es_arr, index=idx, name=f"ES_{p:.0%}"),
        sigma=pd.Series(sigma_arr, index=idx, name="sigma"),
        confidence_level=p,
    )

    if verbose:
        n_var = result.var.notna().sum()
        print(f"✅ Calculé : {n_var} VaR, {n_var} ES")

    return result


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns

    print("🔍 Test du module dynamic_var (GARCH-EVT)\n")

    btc = get_returns("BTC-USD")
    print(f"📊 Données BTC : {len(btc)} obs\n")

    # Rolling backtest
    result = rolling_var_es(
        btc,
        window=1000,
        p=0.99,
        refit_every=50,
        model_type="garch",
        dist="ged",
        threshold_quantile=0.90,
        verbose=True,
    )

    # --- Statistiques descriptives ---
    df = result.to_dataframe().dropna()
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES DE LA VaR DYNAMIQUE")
    print("=" * 70)
    print(f"Période              : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Observations         : {len(df)}")
    print(f"\nVaR 99% (moyenne)    : {df['var'].mean():.3f}%")
    print(f"VaR 99% (min)        : {df['var'].min():.3f}%  ← pire jour")
    print(f"VaR 99% (max)        : {df['var'].max():.3f}%  ← meilleur jour")
    print(f"ES 99% (moyenne)     : {df['es'].mean():.3f}%")

    # --- Taux de violations (out-of-sample) ---
    viol = result.violations().dropna()
    n_viol = int(viol.sum())
    n_obs = len(viol)
    rate = n_viol / n_obs
    expected = 1 - 0.99

    print(f"\n📉 Taux de violations :")
    print(f"   Violations         : {n_viol} / {n_obs}")
    print(f"   Taux observé       : {rate:.4%}")
    print(f"   Taux attendu       : {expected:.4%}")
    print(f"   Écart              : {(rate - expected):.4%}")

    # --- Écart avec VaR gaussienne ---
    from scipy.stats import norm
    var_gauss_mean = df["sigma"].mean() * (-norm.ppf(0.99))
    print(f"\n🔬 Comparaison VaR :")
    print(f"   VaR 99% GARCH-EVT (moyenne) : {df['var'].mean():.3f}%")
    print(f"   VaR 99% Gaussienne (approx) : {var_gauss_mean:.3f}%")
    print(f"   → GARCH-EVT est "
          f"{'plus' if df['var'].mean() < var_gauss_mean else 'moins'} "
          f"conservateur de "
          f"{abs(df['var'].mean() - var_gauss_mean):.3f}%")
