<p align="center">
  <img src="assets/hex-logo.png" alt="statML" width="180"/>
</p>

<h1 align="center">Advanced Statistics and Machine Learning<br/>for Health Research</h1>

<p align="center">
  <em>A practical course for health researchers</em><br/><br/>
  <a href="https://mlhealthcourse.github.io/"><strong>Read the book online</strong></a><br/><br/>
  <img src="https://img.shields.io/badge/licence-CC%20BY%204.0-blue" alt="CC BY 4.0"/>
  <img src="https://img.shields.io/badge/code-R%20%2B%20Python-green" alt="R + Python"/>
  <img src="https://img.shields.io/badge/Quarto-book-orange" alt="Quarto Book"/>
</p>

---

## What this course covers

| Part | Chapters | Topics |
| ---- | -------- | ------ |
| **Pre-Course: Foundations** | 1–4 | Environment setup, probability, statistical inference, regression foundations |
| **Advanced Statistical Methods** | 5–9 | Splines, penalised regression, survival analysis, mixed-effects models, missing data |
| **Bayesian Methods** | 10–11 | Bayesian inference, applied hierarchical models |
| **Supervised Learning** | 12–16 | Trees and ensembles, neural networks, model evaluation, explainability |
| **Applied Supervised Learning: Clinical Prediction Models** | 17–19 | Development, validation and calibration, reporting to TRIPOD+AI |
| **Unsupervised Learning** | 20–21 | PCA, t-SNE, UMAP, clustering |
| **Causal Inference and Evidence Synthesis** | 22–24 | Causal inference, mediation analysis, meta-analysis |

## Who it is for

Researchers who know basic statistics (means, t-tests, maybe some regression)
and want to learn the methods they see in current medical journals: splines,
penalised regression, prediction models, Bayesian analysis, gradient boosting,
and deep learning.

Everything uses clinical data. Every exercise works in **R** and **Python**.
Pick one or try both.

## Key features

- Bilingual R/Python code throughout
- Clinical examples in every chapter
- Exercises with starter code
- Current references (2024--2026) from BMJ, JAMA, Lancet, Nature Medicine
- TRIPOD+AI reporting guidance
- Journal-ready analysis capstone

## Citation

If you use this course in your teaching or research, please cite:

```bibtex
@online{statML2026,
  author = {Khurana, Mark},
  title = {Advanced Statistics and Machine Learning for Health Research},
  year = {2026},
  url = {https://mlhealthcourse.github.io/},
  note = {Online course, CC BY 4.0}
}
```

## Running locally

```bash
# Requires Quarto (>= 1.7), R, and the packages listed in scripts/common.R
quarto render
```

## Key sources

- Smits, van Kuijk & Wynants, _Improving Health Care with Clinical Prediction
  Models_ (2026)
- Van Calster et al., _Lancet Digital Health_ (2025) -- performance measures
- Lopez-Ayala et al., _BMJ_ (2025) -- continuous variables and splines
- Collins et al., _BMJ_ (2024) -- TRIPOD+AI reporting guidelines
- Harrell, _Regression Modeling Strategies_ (2015)
- McElreath, _Statistical Rethinking_ (2020)

## Licence

[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Use it, adapt it, share it -- just give credit.
