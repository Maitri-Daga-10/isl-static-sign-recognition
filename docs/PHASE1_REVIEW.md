# Phase 1 review sheet

## Project

**Technology for Inclusive Communication: Developing an AI-Based Sign Language Recognition System**

## Phase 1 scope

This milestone recognizes a selected set of **static, isolated ISL hand signs**
from a public, appropriately licensed dataset. It uses hand-landmark features
and a Linear SVC baseline. It does not recognize continuous signing, translate
sentences, or replace a qualified interpreter.

## Phase 1 evidence (run on November 24, 2026)

- Dataset selected: Indian Sign Language (ISL) character-level dataset, Prathum Arikeri, Kaggle, version 1
- Dataset source: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
- License listed by Kaggle: CC BY-SA 4.0; preserve creator attribution and ShareAlike terms when distributing adaptations
- Provenance caveat: Kaggle description says the dataset combines the creator's own images with images from other Kaggle and GitHub datasets; inspect page acknowledgements/upstream credits
- Downloaded data: 42,745 images across 35 folders (A-Z and digits 1-9; no 0).
- Training sample: cap of 100 images per label; 3,500 images selected.
- Signer/session metadata: none supplied in the class-folder download.
- Feature pipeline: two-hand MediaPipe landmarks, wrist/scale normalization,
  132 features, `StandardScaler` + `LinearSVC`.
- MediaPipe hand detections: 3,415 of 3,500 selected images (97.6%); 85 skipped.
- Lowest detection rate in this sample: P, 63%; next lowest C, 87%.
- Evaluation: stratified image-level holdout, 20% of detected images.
- Held-out accuracy: 99.85%; macro F1: 99.86%.
- Confusing label pairs: see `artifacts/confusion_matrix.png`.
- Offline prediction: one A image outside the sampled training set predicted A.
  The image overlay is generated locally and is not included because it contains
  a dataset image.
- Webcam: inference code is implemented, but the webcam was not accessed during
  this run. Test it on the review laptop before promising a live demo.
- Next action: validate labels and the low P detection rate with an
  ISL-knowledgeable reviewer; run signer-held-out evaluation if signer metadata
  becomes available.

## Why this dataset fits Phase 1

The Kaggle listing describes a character-level image dataset, lists a download
size of about 294 MB, and states a CC BY-SA 4.0 license. This matches a static
landmark + SVC baseline better than a multi-gigabyte continuous-video corpus.
The class listing has digits 1-9 and letters A-Z; digit 0 is omitted according
to the creator. The project download extracted 42,745 images across 35 folders.
The dataset is a candidate, not independent proof that every image is correctly
labeled ISL; validate labels with an ISL-knowledgeable reviewer before
describing performance as ISL recognition.

The 99.85% result is an **image-level baseline**, not an unseen-person result.
There are no signer/session IDs, and images from the same source or signer can
appear on both sides of the split. Present the metric with that limitation.
