# Smart MCQ Solver Challenge

**Name:** Raghav Kapoor  
**Roll No:** 23f2002364  
**Course:** DL & GenAI Project, IIT Madras BS in Data Science  
**Term:** T2-2026

## Project Overview

This repository contains my work for the Deep Learning & Generative AI project on multi-choice question answering. The task is to predict the top 3 most probable answers for each MCQ and evaluate performance using MAP@3.

The notebooks cover the full progression of the project, including:

- data loading and exploratory analysis
- classical NLP baselines such as TF-IDF and cosine similarity
- semantic similarity with Word2Vec / sentence embeddings
- zero-shot classification experiments
- retrieval-augmented prompting ideas
- model training with PyTorch
- pretrained transformer experiments
- LoRA fine-tuning
- ensembling and submission generation

## Repository Structure

- `notebooks/` - milestone notebooks from setup through fine-tuning
- `src/` - reusable code for preprocessing, modeling, and evaluation
- `models/` - saved checkpoints and trained artifacts
- `reports/` - project report, notes, and final write-up
- `requirements.txt` - Python dependencies used for the project

## Current Status

- Registration setup completed in the notebook workflow
- Baseline and intermediate experiments are documented in the milestone notebooks
- W&B logging is included in the training and evaluation workflow
- Submission generation is implemented in the later milestone notebook

## Setup

Create a Python environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

If you are working in Kaggle, install the missing packages from the notebook as needed.

## Notebooks

- `notebooks/milestone-0.ipynb` - orientation and setup
- `notebooks/milestone-1.ipynb` - NLP foundations and baseline methods
- `notebooks/milestone-2.ipynb` - transformer experiments and training setup
- `notebooks/milestone-3.ipynb` - RAG-oriented exploration and continued experimentation
- `notebooks/milestone-4.ipynb` - LoRA fine-tuning, validation, and submission workflow

## Dependencies

The project currently uses:

- torch
- transformers
- datasets
- wandb
- sentence-transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Notes

- The codebase is still notebook-led, but the long-term goal is to move reusable logic into `src/`.
- Trained weights and final submission artifacts should be stored in `models/`.
- The final technical report should live in `reports/`.

## Next Steps

- Add reusable preprocessing and inference code to `src/`
- Save trained model artifacts in `models/`
- Draft the final report in `reports/`
- Keep the README updated as the project matures
