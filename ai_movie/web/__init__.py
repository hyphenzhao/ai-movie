"""Web front end for the dubbing pipeline (FastAPI + a static single page).

The desktop GUI (``ai_movie/gui``) is Tk and can only be used at the
machine.  This package exposes the *same* headless runner
(``scripts/run_pipeline.py`` and the v2/deliver scripts) over HTTP so the
pipeline can be driven, watched (live log + progress), edited (translation,
speaker labels, glossary, face binding) and previewed from any browser on
the LAN — or remotely through the VPS reverse proxy.

Nothing here imports torch: pipeline work always runs in a subprocess, and
stage status is asked from ``run_pipeline.py --status-json``.
"""
