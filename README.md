# Lab: Model Testing with Weights & Biases and LLMs

In this lab, you will use **Weights & Biases (W&B)** for interactive model evaluation and an LLM to generate targeted test cases.
You will compare a candidate sentiment model with a baseline, slice predictions to uncover failure modes, log the results to W&B, and stress-test a weak slice with synthetic examples.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlip-cmu-online/lab-model-testing/blob/main/lab4.ipynb)

## Recommended Environment

Open the notebook with the Colab button above and select a fresh CPU runtime.
Sign in to GitHub and authorize Colab if it asks for access to the starter repository.
The tested environment is Python 3.12.13 with the exact package versions in `requirements.txt`.
Run the notebook installation and version-check cells before beginning the lab.
The two sentiment models and the optional local case generator download from Hugging Face when first used, so the first run requires an internet connection.

To work locally, create a Python 3.12 virtual environment and run `python -m pip install -r requirements.txt`.

## Deliverables

Your goal is to act like an ML engineer preparing a model for deployment by justifying slices, inspecting slice performance in W&B, and validating a weakness with synthetic data.

1. **Run Steps 1–4 and define at least five hypothesis-driven slices.**
   Each slice should capture a specific property of the tweets, such as hashtags, negation, emoji density, unusual length, or the presence of mentions.
   Record why each slice matters to model behavior in the notebook.
2. **Log the analysis to W&B.**
   Ensure `df_long`, `slice_metrics`, `regression_metrics`, and `df_eval` are logged.
   Build comparative visualizations of your choice for the slices.
   Save the W&B run link and answer in the notebook why accuracy can be misleading and what slicing revealed.
3. **Complete the targeted stress test in Step 7.**
   Record a hypothesis, ten generated tweets, and the expected sentiment label for each tweet in the notebook.
   Run the helper that scores both models against those labels.
   Record any repeated or new failures and explain whether they change your confidence in deploying the candidate model.

For every slice you log, keep a short note in `saved_slice_notes` so your takeaways remain reviewable without rerunning the notebook.

## Generate the Submission Report

After completing the lab, run the final **Save submission evidence** cell and save the executed notebook.
Export a W&B table or save a screenshot that shows the run evidence without requiring another viewer to access your W&B project.
Then, from the repository root, run:

```shell
python3 scripts/generate-model-testing-submission.py \
  --learner "Your name" \
  --notebook lab4.ipynb \
  --wandb-evidence path/to/wandb-export-or-screenshot.png
```

The W&B evidence may be CSV, HTML, JSON, PDF, PNG, JPEG, or WebP.
Open `submission/model-testing-report.html` and correct every item marked `missing` before uploading it to Canvas.
Keep `submission/model-testing-manifest.json` with your submission evidence, and answer the slice and deployment interpretation questions separately in Canvas.

The command reads only local files; it does not open W&B, download models, or rerun the notebook.
It checks that every code cell was executed, that the report contains labeled overall and five-or-more slice metrics, and that exactly ten labeled stress-test cases have baseline and candidate scores.
It also checks the saved W&B run identifier, required logged artifact names, slice notes, and obvious plaintext credential patterns.
The checker cannot detect a credential visible inside an image, so review the finished report before uploading it.

## W&B Login

Create a free W&B account and retrieve your key from the [W&B authorization page](https://wandb.ai/authorize).
Run the notebook login cell and paste the key only into its hidden prompt.
Do not type the key into a code cell, save it in the notebook, or commit it to the repository.

## No-Paid-LLM Route

Step 7 includes an optional local generator based on the public `HuggingFaceTB/SmolLM2-360M-Instruct` model.
It runs on the Colab CPU and does not require an API key or paid service.
You may instead use a free LLM service available to you, but never paste a service token into the notebook.

## Reference

See the [W&B panels guide](https://docs.wandb.ai/guides/app/features/panels/) for help building slice visualizations.
