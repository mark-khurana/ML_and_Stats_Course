# Contributing

## Repository layout

| Path | Contents |
|------|----------|
| `.github/workflows/` | CI/CD: environment checks + GitHub Pages deployment |
| `_book/` | Rendered HTML output (gitignored; built in CI) |
| `_freeze/` | Quarto execution cache, committed so CI and other contributors skip re-execution |
| `_includes/` | HTML partials injected into the rendered site (analytics) |
| `advanced/` | Advanced toolkit chapters (causal inference, g-methods, mediation, meta-analysis) |
| `appendices/` | Appendices (dataset codebook, math notation, further reading, exercise solutions) |
| `assets/` | Logo and social preview images |
| `chapters/` | Main course chapters (`.qmd` files, numbered 00--16) |
| `data/` | Clinical datasets (CSV) used throughout the book |
| `images/` | SVG diagrams referenced by chapters |
| `scripts/` | Shared setup (`common.R`/`common.py`), package install/check scripts, `post_install.sh` |
| `solutions/` | Exercise solutions in R & Python |

Key root files:

| File | Purpose |
|------|---------|
| `_quarto.yml` | Book structure, format, and rendering options |
| `custom.scss` | Site theme overrides |
| `pixi.toml` | Dependency and environment declarations (conda + PyPI) |
| `references.bib` | Bibliography |
| `render.sh` | Local rendering helper (sets paths, runs `quarto render`) |

## Development setup

Install [Pixi](https://pixi.sh), then:

```bash
pixi install -e dev && pixi run -e dev post_install
```

This gives you R, Python, Quarto, knitr, reticulate, and every package used in the book. The `post_install` step compiles a handful of CRAN packages (brms, rstanarm, bayesplot, keras3, dcurves, EValue, CMAverse) that are not available on conda-forge.

## Previewing and rendering

```bash
pixi run -e dev preview   # live-reload dev server
pixi run -e dev render    # full render to _book/
```

The book uses `freeze: true`, so Quarto reuses cached outputs from `_freeze/` instead of re-executing every code chunk. To force a chapter to re-execute:

```bash
rm -rf _freeze/chapters/<chapter_name>
pixi run -e dev render
```

## Adding or editing a chapter

1. Create a `.qmd` file in `chapters/` (or `advanced/` for the advanced toolkit).
2. Register it in `_quarto.yml` under the appropriate `part:`.
3. Use `scripts/common.R` or `scripts/common.py` at the top of your code blocks for shared themes, palettes, and data loaders.
4. Add exercise solutions to `solutions/`.
5. Preview locally before pushing.

## Modifying dependencies

- Edit `pixi.toml`. Use bounded version specifiers (e.g. `">=1.2,<2"`), not `"*"`.
- If the package is needed by standalone-R users (no Pixi), also add it to `scripts/install_packages.R`.
- Push your changes -- the `check-envs.yml` workflow will test all four environment arms automatically.

## Deployment

Pushing to `main` triggers the `publish.yml` workflow, which installs the lightweight `book` pixi environment, renders the book from `_freeze/`, and deploys `_book/` to GitHub Pages.

Workflow:

1. Commit `_freeze/` changes (if any chapters were re-executed)
2. Push to `main`

## CI

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `check-envs.yml` | Push/PR touching `pixi.toml`, `scripts/install_packages.R`, `scripts/post_install.sh` | Tests environment arms across Ubuntu, macOS, and Windows |
| `publish.yml` | Push to `main` | Renders the book via pixi `book` env and deploys to GitHub Pages |
