# Bangladesh Legal QA — VS Code version

This repository version is designed to replace the unstable Colab notebook.

## Why it is safer

It does not render hundreds of outputs in a browser. Instead it:

- prints compact progress to the VS Code terminal;
- writes the complete log to `artifacts/logs/pipeline.log`;
- saves candidates and reranking every few steps;
- resumes after interruption;
- writes plots directly to PNG files;
- clears GPU cache between expensive models;
- uses `bge-reranker-v2-m3`, not the much heavier Gemma reranker.

## Recommended migration from your current Colab project

Download from your Google Drive backup:

- `legal_qa_splits/`
- `legal_e5_hardneg/`
- optionally `legal_e5_finetuned/`
- `retrieval_checkpoint.pkl` as an extra backup

Put them somewhere on your computer. They do NOT need to be committed to GitHub.

Then create a virtual environment:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run using your existing split and trained model:

```powershell
python legal_qa_pipeline.py --stage all --split-dir "C:\PATH\legal_qa_splits" --hardneg-model "C:\PATH\legal_e5_hardneg"
```

The pipeline will not retrain E5.

## If your computer has an NVIDIA GPU

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, install the correct CUDA-enabled PyTorch build from the official PyTorch installer.

## If your GPU memory is small

Start conservatively:

```powershell
python legal_qa_pipeline.py --stage all --split-dir "C:\PATH\legal_qa_splits" --hardneg-model "C:\PATH\legal_e5_hardneg" --reranker-batch-size 2 --e5-batch-size 16
```

If that still runs out of GPU memory:

```powershell
python legal_qa_pipeline.py --stage all --split-dir "C:\PATH\legal_qa_splits" --hardneg-model "C:\PATH\legal_e5_hardneg" --reranker-batch-size 1 --e5-batch-size 8
```

Or use CPU:

```powershell
python legal_qa_pipeline.py --stage all --device cpu --split-dir "C:\PATH\legal_qa_splits" --hardneg-model "C:\PATH\legal_e5_hardneg"
```

CPU is slower but should be much less prone to GPU OOM.

## Resume after a crash

Run the same command again. By default `--resume` behavior is ON.

The reranker saves partial rankings every 10 questions, so it can continue instead of restarting everything.

## Outputs

```text
artifacts/
  checkpoints/
  data/
  figures/
  logs/
    pipeline.log
  models/
  results/
  splits/
```

Important report files:

```text
artifacts/results/validation_metrics.json
artifacts/results/m3_validation_details.csv
artifacts/results/m3_by_language.csv
artifacts/results/m3_by_question_type.csv
artifacts/results/failure_analysis.csv
artifacts/results/validation_rag_top5.jsonl
```

Report figures:

```text
artifacts/figures/candidate_retrieval.png
artifacts/figures/reranker_recall_at_k.png
artifacts/figures/pipeline_stages.png
artifacts/figures/language_comparison.png
artifacts/figures/question_type_comparison.png
artifacts/figures/failure_decomposition.png
```

## Final test

Do not use the test set repeatedly while improving models.

After the retrieval architecture is frozen:

```powershell
python legal_qa_pipeline.py --stage all --final-test --split-dir "C:\PATH\legal_qa_splits" --hardneg-model "C:\PATH\legal_e5_hardneg"
```

## GitHub

The provided `.gitignore` excludes models, checkpoints and caches.

```powershell
git init
git add .
git commit -m "Initial Bangladesh Legal QA pipeline"
```

Later we can add:
1. judgment retrieval;
2. Qwen answer generation;
3. citation/groundedness evaluation;
4. a Gradio legal QA chatbot.
