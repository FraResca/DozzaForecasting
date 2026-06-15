# Compile instructions

This bundle follows the Springer LNCS LaTeX2e template.

Recommended compilation:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The bundle includes `llncs.cls`, `splncs04.bst`, split section files, bibliography, and selected figures copied from the project outputs.

CSV versions of the manuscript tables are in `tables/`. They can be rebuilt from
the local project outputs with:

```bash
python build_tables.py
```
