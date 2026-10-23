# NeuroLens AI: An Agentic Brain MRI Research Assistant

A system that analyses brain MRI scans, segments suspicious regions, estimates uncertainty, and generates a radiology-style evidence report.

> **Disclaimer:** This is a research prototype. It is not a medical device and must not be used for diagnosis or clinical decisions.

## Planned components

- **Self-supervised learning** (`src/ssl`): learn representations from unlabeled MRI scans
- **Segmentation + uncertainty** (`src/segmentation`): segment suspicious regions and estimate confidence
- **Vision-language model** (`src/vlm`): connect image findings with medical text to draft reports
- **Diffusion models** (`src/diffusion`): controlled denoising/reconstruction experiments
- **Agentic workflow** (`src/agent`): check segmentation quality, request a second analysis when confidence is low, flag cases for human review

## Status

Work in progress.