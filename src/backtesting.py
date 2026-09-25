"""
Backtesting réglementaire de modèles de VaR.

Implémente 4 tests statistiques standards en risk management :
    1. Kupiec (1995)              : Proportion of Failures (POF)
    2. Christoffersen (1998) IND  : Indépendance des violations
    3. Christoffersen (1998) CC   : Conditional Coverage (POF + IND)
    4. Engle-Manganelli (2004)    : Dynamic Quantile (DQ)

Références :
    - Kupiec, P. (1995). Techniques for verifying the accuracy of risk
      measurement models.
    - Christoffersen, P. F. (1998). Evaluating interval forecasts.
    - Engle, R. F., & Manganelli, S. (2004). CAViaR.

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Résultat d'une batterie de backtests sur un modèle."""
    model_name: str
    p: float
    n_obs: int
    n_violations: int
    violation_rate: float
    expected_rate: float

    # Statistiques des tests
    kupiec_lr: float
    kupiec_pvalue: float
    christoffersen_ind_lr: float
    christoffersen_ind_pvalue: float
    christoffersen_cc_lr: float
    christoffersen_cc_pvalue: float
    dq_stat: float
    dq_pvalue: float

    def verdict_kupiec(self, alpha: float = 0.05) -> str:
        return "✅ OK" if self.kupiec_pvalue > alpha else "❌ Rejeté"

    def verdict_christoffersen_ind(self, alpha: float = 0.05) -> str:
        return "✅ OK" if self.christoffersen_ind_pvalue > alpha else "❌ Rejeté"

    def verdict_christoffersen_cc(self, alpha: float = 0.05) -> str:
        return "✅ OK" if self.christoffersen_cc_pvalue > alpha else "❌ Rejeté"

    def verdict_dq(self, alpha: float = 0.05) -> str:
        return "✅ OK" if self.dq_pvalue > alpha else "❌ Rejeté"

    def summary(self) -> str:
        """Résumé texte lisible."""
        return (
            f"Backtest — {self.model_name} (p={self.p:.1%})\n"
            f"  N obs / violations   : {self.n_obs} / {self.n_violations} "
            f"({self.violation_rate:.4%})\n"
            f"  Taux attendu         : {self.expected_rate:.4%}\n"
            f"  ─────────────────────────────────────\n"
            f"  Kupiec (POF)         : LR={self.kupiec_lr:.3f}, "
            f"p={self.kupiec_pvalue:.4f}  {self.verdict_kupiec()}\n"
            f"  Christoffersen (IND) : LR={self.christoffersen_ind_lr:.3f}, "
            f"p={self.christoffersen_ind_pvalue:.4f}  "
            f"{self.verdict_christoffersen_ind()}\n"
            f"  Christoffersen (CC)  : LR={self.christoffersen_cc_lr:.3f}, "
            f"p={self.christoffersen_cc_pvalue:.4f}  "
            f"{self.verdict_christoffersen_cc()}\n"
            f"  Engle-Manganelli(DQ) : stat={self.dq_stat:.3f}, "
            f"p={self.dq_pvalue:.4f}  {self.verdict_dq()}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_violations(
    returns: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """
    Renvoie la série binaire des violations (1 = violation, 0 = pas).

    Seules les dates où la VaR est définie sont considérées.
    """
    mask = var_series.notna() & returns.notna()
    hits = pd.Series(np.nan, index=returns.index, name="violation")
    hits[mask] = (returns[mask] < var_series[mask]).astype(int)
    return hits.dropna()


# ---------------------------------------------------------------------------
# Test 1 — Kupiec (POF)
# ---------------------------------------------------------------------------

def kupiec_pof(
    n_violations: int,
    n_obs: int,
    p: float,
) -> tuple[float, float]:
    """
    Test de Kupiec (Proportion of Failures).

    Parameters
    ----------
    n_violations : int
        Nombre de violations observées.
    n_obs : int
        Nombre total d'observations.
    p : float
        Niveau de confiance (ex : 0.99).
        Taux attendu de violations = 1 - p.

    Returns
    -------
    (LR, pvalue)
    """
    expected_rate = 1 - p
    if n_violations == 0:
        # Cas limite : toutes les violations absentes
        LR = -2 * (n_obs * np.log(1 - expected_rate))
        return float(LR), float(1 - chi2.cdf(LR, df=1))

    if n_violations == n_obs:
        LR = -2 * (n_obs * np.log(expected_rate))
        return float(LR), float(1 - chi2.cdf(LR, df=1))

    pi_hat = n_violations / n_obs

    # Log-vraisemblance sous H0 (taux = expected_rate)
    ln_L0 = (n_obs - n_violations) * np.log(1 - expected_rate) + \
            n_violations * np.log(expected_rate)

    # Log-vraisemblance sous H1 (taux = pi_hat)
    ln_L1 = (n_obs - n_violations) * np.log(1 - pi_hat) + \
            n_violations * np.log(pi_hat)

    LR = -2 * (ln_L0 - ln_L1)
    pval = 1 - chi2.cdf(LR, df=1)
    return float(LR), float(pval)


# ---------------------------------------------------------------------------
# Test 2 — Christoffersen (IND)
# ---------------------------------------------------------------------------

def christoffersen_independence(
    hits: pd.Series,
) -> tuple[float, float]:
    """
    Test d'indépendance des violations (Christoffersen 1998).

    Parameters
    ----------
    hits : pd.Series
        Série binaire (1 = violation, 0 = pas).

    Returns
    -------
    (LR, pvalue)
    """
    h = hits.astype(int).values
    h_prev = h[:-1]
    h_curr = h[1:]

    n00 = int(np.sum((h_prev == 0) & (h_curr == 0)))
    n01 = int(np.sum((h_prev == 0) & (h_curr == 1)))
    n10 = int(np.sum((h_prev == 1) & (h_curr == 0)))
    n11 = int(np.sum((h_prev == 1) & (h_curr == 1)))

    n0 = n00 + n01
    n1 = n10 + n11
    n_total = n0 + n1

    if n_total == 0 or n0 == 0 or n1 == 0:
        return np.nan, np.nan

    # Probabilités conditionnelles
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / n_total

    # Gestion des cas limites (log 0)
    eps = 1e-10
    pi01 = max(min(pi01, 1 - eps), eps)
    pi11 = max(min(pi11, 1 - eps), eps)
    pi = max(min(pi, 1 - eps), eps)

    ln_L0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ln_L1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + \
            n10 * np.log(1 - pi11) + n11 * np.log(pi11)

    LR = -2 * (ln_L0 - ln_L1)
    pval = 1 - chi2.cdf(LR, df=1)
    return float(LR), float(pval)


# ---------------------------------------------------------------------------
# Test 3 — Christoffersen (CC) = POF + IND
# ---------------------------------------------------------------------------

def christoffersen_cc(
    n_violations: int,
    n_obs: int,
    p: float,
    hits: pd.Series,
) -> tuple[float, float]:
    """
    Test de couverture conditionnelle = POF + IND.
    Distribué selon χ²(2) sous H0.
    """
    lr_pof, _ = kupiec_pof(n_violations, n_obs, p)
    lr_ind, _ = christoffersen_independence(hits)
    lr_cc = lr_pof + lr_ind
    pval = 1 - chi2.cdf(lr_cc, df=2)
    return float(lr_cc), float(pval)


# ---------------------------------------------------------------------------
# Test 4 — Engle-Manganelli (DQ)
# ---------------------------------------------------------------------------

def engle_manganelli_dq(
    hits: pd.Series,
    var_series: pd.Series,
    p: float,
    lags: int = 4,
) -> tuple[float, float]:
    """
    Test Dynamic Quantile (Engle & Manganelli 2004).

    Régression : (I_t - p) = β0 + Σ βi·I_{t-i} + γ·VaR_t + ε_t
    Statistique DQ = β' X'X β / (p(1-p)) ~ χ²(k+2)

    Parameters
    ----------
    hits : pd.Series
        Série binaire des violations.
    var_series : pd.Series
        Série de VaR (alignée avec hits).
    p : float
        Niveau de confiance.
    lags : int
        Nombre de retards (défaut : 4).

    Returns
    -------
    (DQ, pvalue)
    """
    # Alignement
    mask = hits.notna() & var_series.notna()
    h = hits[mask].astype(float).values
    v = var_series[mask].values

    T = len(h)
    if T < lags + 10:
        return np.nan, np.nan

    # Construit la matrice X : [1, I_{t-1}, ..., I_{t-lags}, VaR_t]
    X = np.ones((T, lags + 2))
    for i in range(1, lags + 1):
        X[i:, i] = h[:-i]
    X[:, -1] = v

    # Supprime les `lags` premières lignes (pas de retards dispo)
    X = X[lags:]
    y = h[lags:] - p

    # OLS : β = (X'X)^-1 X'y
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # Statistique DQ
    T_eff = len(y)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    dq = float((beta @ XtX @ beta) / (p * (1 - p)))
    df = lags + 2
    pval = 1 - chi2.cdf(dq, df=df)
    return dq, float(pval)


# ---------------------------------------------------------------------------
# API haut-niveau
# ---------------------------------------------------------------------------

def run_full_backtest(
    model_name: str,
    returns: pd.Series,
    var_series: pd.Series,
    p: float = 0.99,
) -> BacktestResult:
    """
    Lance la batterie complète de tests sur un modèle.

    Parameters
    ----------
    model_name : str
    returns : pd.Series
    var_series : pd.Series
        VaR (même index que returns).
    p : float
        Niveau de confiance.

    Returns
    -------
    BacktestResult
    """
    hits = compute_violations(returns, var_series)
    n_obs = len(hits)
    n_viol = int(hits.sum())
    rate = n_viol / n_obs if n_obs > 0 else np.nan

    kup_lr, kup_p = kupiec_pof(n_viol, n_obs, p)
    ind_lr, ind_p = christoffersen_independence(hits)
    cc_lr, cc_p = christoffersen_cc(n_viol, n_obs, p, hits)
    dq_stat, dq_p = engle_manganelli_dq(
        hits, var_series.reindex(hits.index), p, lags=4
    )

    return BacktestResult(
        model_name=model_name,
        p=p,
        n_obs=n_obs,
        n_violations=n_viol,
        violation_rate=rate,
        expected_rate=1 - p,
        kupiec_lr=kup_lr,
        kupiec_pvalue=kup_p,
        christoffersen_ind_lr=ind_lr,
        christoffersen_ind_pvalue=ind_p,
        christoffersen_cc_lr=cc_lr,
        christoffersen_cc_pvalue=cc_p,
        dq_stat=dq_stat,
        dq_pvalue=dq_p,
    )


def backtest_table(
    returns: pd.Series,
    models: dict[str, pd.Series],
    p: float = 0.99,
) -> pd.DataFrame:
    """
    Backteste plusieurs modèles et renvoie un tableau comparatif.

    Parameters
    ----------
    returns : pd.Series
    models : dict of {name: var_series}
    p : float

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, var in models.items():
        r = run_full_backtest(name, returns, var, p=p)
        rows.append({
            "model": name,
            "n_obs": r.n_obs,
            "n_viol": r.n_violations,
            "rate": r.violation_rate,
            "kupiec_p": r.kupiec_pvalue,
            "ind_p": r.christoffersen_ind_pvalue,
            "cc_p": r.christoffersen_cc_pvalue,
            "dq_p": r.dq_pvalue,
            "kupiec": r.verdict_kupiec(),
            "ind": r.verdict_christoffersen_ind(),
            "cc": r.verdict_christoffersen_cc(),
            "dq": r.verdict_dq(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns
    from src.dynamic_var import rolling_var_es
    from src.benchmark_models import (
        var_historical, var_normal, var_garch_normal, var_garch_ged,
    )

    print("🔍 Test du module backtesting\n")
    btc = get_returns("BTC-USD")
    p = 0.99
    window = 1000

    print("📊 Calcul des VaR pour les 5 modèles (peut prendre 2-3 min)…\n")

    var_hist = var_historical(btc, window=window, p=p)
    var_norm = var_normal(btc, window=window, p=p)
    var_gn = var_garch_normal(btc, window=window, p=p, refit_every=50)
    var_gg = var_garch_ged(btc, window=window, p=p, refit_every=50)

    result_evt = rolling_var_es(
        btc, window=window, p=p, refit_every=50,
        model_type="garch", dist="ged",
        threshold_quantile=0.90, verbose=False,
    )
    var_evt = result_evt.var

    models = {
        "Historique": var_hist,
        "Normale": var_norm,
        "GARCH-Normal": var_gn,
        "GARCH-GED": var_gg,
        "GARCH-EVT": var_evt,
    }

    print("=" * 90)
    print("📊 BACKTESTING COMPLET — p = 99%")
    print("=" * 90)

    for name, var in models.items():
        r = run_full_backtest(name, btc, var, p=p)
        print("\n" + r.summary())

    print("\n" + "=" * 90)
    print("🏆 TABLEAU RÉCAPITULATIF")
    print("=" * 90)
    table = backtest_table(btc, models, p=p)
    print("\n" + table.to_string(index=False))

    print("\n" + "=" * 90)
    print("🥇 CLASSEMENT (par p-value DQ, plus grand = mieux)")
    print("=" * 90)
    ranking = table.sort_values("dq_p", ascending=False).reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, row in ranking.iterrows():
        medal = medals[i] if i < len(medals) else "  "
        print(f"  {medal} {row['model']:<15} "
              f"Kupiec p={row['kupiec_p']:.4f}  "
              f"CC p={row['cc_p']:.4f}  "
              f"DQ p={row['dq_p']:.4f}")
