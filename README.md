# 📊 GARCH-EVT Risk Engine

> **Mesure dynamique du risque extrême sur les cryptomonnaies** — GARCH(1,1) + Théorie des Valeurs Extrêmes (GPD/POT), avec backtesting réglementaire Bâle III et comparaison de 4 modèles de VaR.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red.svg)]()

---

## 🎯 Objectifs

Ce projet implémente un **moteur de mesure du risque extrême** adapté aux cryptomonnaies, combinant :

- **GARCH(1,1) / GJR-GARCH** — modélisation de la volatilité conditionnelle
- **Théorie des Valeurs Extrêmes (EVT)** — modélisation des queues via GPD / POT
- **VaR/ES dynamiques** — adaptation au régime de marché
- **Backtesting Bâle III** — Kupiec, Christoffersen, Engle-Manganelli
- **Benchmark** — comparaison de 4 modèles de VaR (Historique, Normale, GARCH-Normal, GARCH-EVT)
- **Dashboard interactif** — Streamlit + génération PDF automatique

---

## 📸 Aperçu

### Dashboard interactif

![Dashboard](docs/screenshots/dashboard.png)

### Rapport PDF généré automatiquement

![Rapport - Couverture](docs/screenshots/report_cover.png)

![Rapport - Backtesting](docs/screenshots/report_backtesting.png)

---

## 🏆 Résultats clés

### Sur BTC (Bitcoin, 2345 obs, out-of-sample 1345 jours, p = 99%)

| Rang | Modèle | Taux violations | Kupiec p | CC p | Score Bâle III |
|------|--------|-----------------|----------|------|----------------|
| 🥇 | **GARCH-EVT** | **0.97%** | **0.901** | **0.874** | **1.775** |
| 🥈 | Normale | 0.74% | 0.321 | 0.567 | 0.887 |
| 🥉 | GARCH-Normal | 1.41% | 0.152 | 0.273 | 0.425 |
| 4 | Historique | 0.37% | 0.008 | 0.029 | 0.036 |

**Conclusion** : Sur BTC, GARCH-EVT obtient le **meilleur score Bâle III** (1.775), **2× supérieur** au meilleur concurrent. Le paramètre ξ = 0.08 (Fréchet) confirme une queue épaisse, justifiant l'approche EVT.

---

## 🏗️ Architecture
