"""
Modèles de VaR concurrents pour comparaison avec GARCH-EVT.

Ce module fournit 4 modèles de VaR alternatifs :
    1. VaR Historique          (quantile empirique)
    2. VaR Normale             (variance-covariance)
    3. GARCH-Normal            (GARCH + quantile normal)
    4. GARCH-t                 (GARCH + quantile Student-t)

Ces modèles sont utilisés dans le benchmark pour montrer la supériorité
de GARCH-EVT sur le backtesting.

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from src.garch_model import fit_garch
from scipy.stats import gennorm

# ---------------------------------------------------------------------------
# Modèle 1 — VaR Historique
# ---------------------------------------------------------------------------

def var_historical(
    returns: pd.Series,
    window: int = 1000,
    p: float = 0.99,
) -> pd.Series:
    """
    VaR historique en fenêtre glissante.

    VaR_t = quantile empirique (1-p) des `window` derniers rendements.
    Pas de distribution supposée, mais sensible à la fenêtre.

    Parameters
    ----------
    returns : pd.Series
    window : int
    p : float
        Niveau de confiance (ex : 0.99).

    Returns
    -------
    pd.Series
        VaR en % (négatif), indexée comme returns. NaN avant `window`.
    """
    quantile_level = 1 - p
    var = returns.rolling(window=window).quantile(quantile_level)
    var.name = f"VaR_hist_{p:.0%}"
    return var


# ---------------------------------------------------------------------------
# Modèle 2 — VaR Normale (Variance-Covariance)
# ---------------------------------------------------------------------------

def var_normal(
    returns: pd.Series,
    window: int = 1000,
    p: float = 0.99,
) -> pd.Series:
    """
    VaR paramétrique gaussienne (approche RiskMetrics).

    VaR_t = μ_w + σ_w · Φ⁻¹(1-p)

    où μ_w, σ_w sont la moyenne et l'écart-type glissants.

    Parameters
    ----------
    returns : pd.Series
    window : int
    p : float

    Returns
    -------
    pd.Series
    """
    mu = returns.rolling(window=window).mean()
    sigma = returns.rolling(window=window).std()
    z = norm.ppf(1 - p)   # quantile normal (négatif pour p > 0.5)
    var = mu + sigma * z
    var.name = f"VaR_norm_{p:.0%}"
    return var


# ---------------------------------------------------------------------------
# Modèle 3 — GARCH-Normal
# ---------------------------------------------------------------------------

def var_garch_normal(
    returns: pd.Series,
    window: int = 1000,
    p: float = 0.99,
    refit_every: int = 50,
    verbose: bool = False,
) -> pd.Series:
    """
    VaR conditionnelle GARCH(1,1) avec distribution normale.

    VaR_t = μ + σ_t · Φ⁻¹(1-p)

    Parameters
    ----------
    returns : pd.Series
    window : int
    p : float
    refit_every : int
    verbose : bool

    Returns
    -------
    pd.Series
    """
    return _var_garch_generic(
        returns, window, p, refit_every, dist="normal", verbose=verbose
    )


# ---------------------------------------------------------------------------
# Modèle 4 — GARCH-t (Student)
# ---------------------------------------------------------------------------

def var_garch_t(
    returns: pd.Series,
    window: int = 1000,
    p: float = 0.99,
    refit_every: int = 50,
    verbose: bool = False,
) -> pd.Series:
    """
    VaR conditionnelle GARCH(1,1) avec distribution Student-t.

    VaR_t = μ + σ_t · t⁻¹_ν(1-p)

    où ν est le degré de liberté estimé par le fit GARCH-t.

    Parameters
    ----------
    returns : pd.Series
    window : int
    p : float
    refit_every : int
    verbose : bool

    Returns
    -------
    pd.Series
    """
    return _var_garch_generic(
        returns, window, p, refit_every, dist="t", verbose=verbose
    )
def var_garch_ged(
    returns: pd.Series,
    window: int = 1000,
    p: float = 0.99,
    refit_every: int = 50,
    verbose: bool = False,
) -> pd.Series:
    """
    VaR conditionnelle GARCH(1,1) avec distribution GED (Generalized Error).

    VaR_t = μ + σ_t · q_GED(1-p; shape)

    où `shape` est le paramètre de forme estimé par le fit GARCH-GED.
    - shape < 2 → queue plus épaisse que la normale
    - shape = 2 → normale
    - shape > 2 → queue plus fine

    Sur BTC, shape ≈ 0.8 est typique → queue très épaisse SANS kurtosis infini
    (contrairement à Student-t ν = 3.1).

    Parameters
    ----------
    returns : pd.Series
    window : int
    p : float
    refit_every : int
    verbose : bool

    Returns
    -------
    pd.Series
    """
    n = len(returns)
    var_arr = np.full(n, np.nan)

    current_sigma = np.nan
    current_mu = 0.0
    current_shape = 2.0   # fallback : normale
    steps_since_refit = 10**9

    for t in range(window, n):
        if steps_since_refit >= refit_every:
            train = returns.iloc[t - window:t]
            try:
                fit = fit_garch(train, model_type="garch", dist="ged")
                current_sigma = float(fit.conditional_vol.iloc[-1])
                current_mu = float(fit.mu)

                # Paramètre de forme de la GED
                params = fit.model_result.params
                shape_raw = float(params.get("eta", 2.0))

                # ⚠️ Contrainte : shape ∈ [0.5, 4]
                #   - shape < 0.5 → queue trop épaisse (Cauchy-like)
                #   - shape > 4 → queue trop fine (redondant avec normale)
                current_shape = max(0.5, min(shape_raw, 4.0))

                if verbose:
                    print(f"   [t={t}] shape_fit={shape_raw:.3f} "
                          f"→ shape_used={current_shape:.3f}")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Refit échoué à t={t}: {e}")
                steps_since_refit = refit_every
                continue
            steps_since_refit = 0

        # Quantile GED via scipy
        if np.isfinite(current_sigma):
            q = gennorm.ppf(1 - p, beta=current_shape)
            var_arr[t] = current_mu + current_sigma * q

        steps_since_refit += 1

    return pd.Series(var_arr, index=returns.index,
                     name=f"VaR_garch_ged_{p:.0%}")

# ---------------------------------------------------------------------------
# Helper commun — GARCH + quantile paramétrique
# ---------------------------------------------------------------------------

def _var_garch_generic(
    returns: pd.Series,
    window: int,
    p: float,
    refit_every: int,
    dist: str,
    verbose: bool = False,
) -> pd.Series:
    """
    Calcule la VaR conditionnelle GARCH avec un quantile paramétrique
    (normal ou Student-t selon `dist`).

    Returns
    -------
    pd.Series
    """
    n = len(returns)
    var_arr = np.full(n, np.nan)

    current_fit = None
    current_sigma = np.nan
    current_mu = 0.0
    current_nu = np.inf
    steps_since_refit = 10**9

    for t in range(window, n):
        if steps_since_refit >= refit_every:
            train = returns.iloc[t - window:t]
            try:
                current_fit = fit_garch(train, model_type="garch", dist=dist)
                current_sigma = float(current_fit.conditional_vol.iloc[-1])
                current_mu = float(current_fit.mu)

                # Degré de liberté ν pour la Student-t
                if dist == "t":
                    nu_key = "nu"
                    if nu_key in current_fit.model_result.params.index:
                        current_nu = float(
                            current_fit.model_result.params[nu_key]
                        )
                    else:
                        current_nu = 5.0   # fallback
                else:
                    current_nu = np.inf
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Refit échoué à t={t}: {e}")
                steps_since_refit = refit_every
                continue
            steps_since_refit = 0

        # Quantile
        if dist == "t" and np.isfinite(current_nu):
            q = student_t.ppf(1 - p, df=current_nu)
        else:
            q = norm.ppf(1 - p)

        if np.isfinite(current_sigma):
            var_arr[t] = current_mu + current_sigma * q

        steps_since_refit += 1

    return pd.Series(var_arr, index=returns.index, name=f"VaR_garch_{dist}_{p:.0%}")


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_manager import get_returns

    print("🔍 Test des modèles de VaR concurrents\n")

    btc = get_returns("BTC-USD")
    print(f"📊 Données BTC : {len(btc)} obs\n")

    p = 0.99
    window = 1000

    print("=" * 70)
    print("📉 CALCUL DES 5 MODÈLES DE VaR CONCURRENTS")
    print("=" * 70)

    print("\n[1/5] VaR Historique…")
    var_hist = var_historical(btc, window=window, p=p)
    print(f"   ✅ {var_hist.notna().sum()} valeurs, "
          f"moyenne = {var_hist.mean():.3f}%")

    print("\n[2/5] VaR Normale (Variance-Covariance)…")
    var_norm = var_normal(btc, window=window, p=p)
    print(f"   ✅ {var_norm.notna().sum()} valeurs, "
          f"moyenne = {var_norm.mean():.3f}%")

    print("\n[3/5] VaR GARCH-Normal…")
    var_gn = var_garch_normal(btc, window=window, p=p,
                              refit_every=50, verbose=False)
    print(f"   ✅ {var_gn.notna().sum()} valeurs, "
          f"moyenne = {var_gn.mean():.3f}%")

    print("\n[4/5] VaR GARCH-GED…")
    var_gg = var_garch_ged(btc, window=window, p=p,
                           refit_every=50, verbose=False)
    print(f"   ✅ {var_gg.notna().sum()} valeurs, "
          f"moyenne = {var_gg.mean():.3f}%")

    print("\n[5/5] VaR GARCH-EVT (déjà calculé ailleurs)…")
    # On utilise la VaR GARCH-EVT du module dynamic_var
    from src.dynamic_var import rolling_var_es
    result_evt = rolling_var_es(
        btc, window=window, p=p, refit_every=50,
        model_type="garch", dist="ged",
        threshold_quantile=0.90, verbose=False,
    )
    var_evt = result_evt.var
    print(f"   ✅ {var_evt.notna().sum()} valeurs, "
          f"moyenne = {var_evt.mean():.3f}%")

    # --- Comparaison des taux de violations ---
    print("\n" + "=" * 70)
    print("📊 TAUX DE VIOLATIONS (out-of-sample) — p = 99%")
    print("=" * 70)

    models = {
        "Historique": var_hist,
        "Normale": var_norm,
        "GARCH-Normal": var_gn,
        "GARCH-GED": var_gg,
        "GARCH-EVT": var_evt,
    }

    expected = 1 - p
    print(f"\n{'Modèle':<18} {'VaR moy.':>10} {'Viol.':>8} {'Taux':>10} "
          f"{'Écart':>10} {'Verdict':<15}")
    print("-" * 80)

    for name, var in models.items():
        mask = var.notna()
        r = btc[mask]
        v = var[mask]
        hits = (r < v).sum()
        n = len(v)
        rate = hits / n
        ecart = rate - expected

        if abs(ecart) < 0.003:
            verdict = "✅ Optimal"
        elif abs(ecart) < 0.005:
            verdict = "🟢 Bien calibré"
        elif rate < expected:
            verdict = "🟡 Trop prudent"
        else:
            verdict = "🔴 Sous-estime"

        print(f"{name:<18} {v.mean():>9.3f}% {hits:>8d} {rate:>9.4%} "
              f"{ecart:>+9.4%} {verdict:<15}")

    print(f"\nTaux attendu : {expected:.4%}")
    print("\n🏆 Classement par proximité au taux attendu :")
    ranking = []
    for name, var in models.items():
        mask = var.notna()
        hits = (btc[mask] < var[mask]).sum()
        rate = hits / mask.sum()
        ranking.append((name, abs(rate - expected)))
    ranking.sort(key=lambda x: x[1])
    for i, (name, ecart) in enumerate(ranking, start=1):
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
        print(f"   {medal} {name:<18} (écart = {ecart:.4%})")
