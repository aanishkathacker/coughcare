---
title: CoughCare
emoji: 🫁
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.67.0
app_file: app.py
pinned: false
license: mit
short_description: Normal vs atypical cough research prototype
---

# CoughCare

A cough-check research prototype for Delhi students. Record or upload one cough;
the app compares its acoustics with a small labelled subset of the public COUGHVID
dataset and returns **“Normal cough”** or **“Monitor this cough”**, with a model
confidence score and a scrolling waveform.

> **Not a medical device.** CoughCare does not diagnose, rule out, or screen for
> any disease. Its held-out ROC-AUC is **0.53** — close to chance. It is a demo of
> an end-to-end audio ML workflow, not a health tool.

## Try it

- Open the app and press **Normal sample** or **Atypical sample**.
- Or record a cough / upload a WAV, MP3, M4A, OGG, WebM or FLAC clip.
- See **About & stats** for the confusion matrix, ROC/PR curves, 95% CIs,
  baselines and limitations.

## What is inside

- Streamlit UI, custom off-white/teal theme, Newsreader + Inter fonts.
- 1,000-clip balanced COUGHVID subset, 500 normal / 500 atypical.
- 395 log-mel/MFCC/spectral features + class-balanced logistic regression.
- 5-fold grouped cross-validation, bootstrap 95% CIs, majority/random baselines.
- Model artefact and metrics are committed; the 951 MB dataset is not needed at
  runtime.

## Notes

- Self-reported COVID-19 rows were excluded; no COVID label is mapped to a
  disease. The only target is normal vs atypical cough.
- COUGHVID 4048312 has no participant ID, so recording UUID is used as the
  grouping key. This limitation is disclosed in the app.
- Full repository, reproduction steps and screenshots:
  <https://github.com/aanishkathacker/coughcare>.

Dataset: COUGHVID (Zenodo 4048312), CC BY 4.0. Code: MIT.
