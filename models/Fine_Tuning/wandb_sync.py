import os
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv

# ==========================================================
# Configuration
# ==========================================================

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if WANDB_API_KEY is None:
    raise ValueError("WANDB_API_KEY not found in .env")

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# ==========================================================
# WandB API
# ==========================================================

api = wandb.Api()

# Replace with your run
RUN_PATH = "sairuthwikreddy2-wipro/price-prediction_gemma3-qlora_20260713_v1/lqlu2l5n"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Helper
# ==========================================================

def safe_columns(df, columns):
    """Return only columns that exist."""
    return [c for c in columns if c in df.columns]

# ==========================================================
# Main
# ==========================================================

def wandb_sync():

    print("=" * 80)
    print("Loading WandB Run")
    print("=" * 80)

    run = api.run(RUN_PATH)

    print(f"Run Name : {run.name}")
    print(f"Project  : {run.project}")
    print(f"Entity   : {run.entity}")
    print(f"State    : {run.state}")
    print()

    history = run.history()

    print(f"History Shape : {history.shape}")

    # ======================================================
    # TRAINING SUMMARY
    # ======================================================

    training_columns = [
        "train/global_step",
        "train/epoch",
        "train/loss",
        "train/mean_token_accuracy",
        "train/learning_rate",
        "train/grad_norm",
        "train/train_runtime",
        "train/train_samples_per_second",
        "train/train_steps_per_second",
    ]

    training_columns = safe_columns(history, training_columns)

    training_df = history[training_columns].copy()

    if "train/loss" in training_df.columns:
        training_df = training_df.dropna(subset=["train/loss"])

    training_df = training_df.rename(
        columns={
            "train/global_step": "Global Step",
            "train/epoch": "Epoch",
            "train/loss": "Train Loss",
            "train/mean_token_accuracy": "Train Accuracy",
            "train/learning_rate": "Learning Rate",
            "train/grad_norm": "Gradient Norm",
            "train/train_runtime": "Runtime (s)",
            "train/train_samples_per_second": "Samples/sec",
            "train/train_steps_per_second": "Steps/sec",
        }
    )

    training_df = training_df.round(4)

    # ======================================================
    # EVALUATION SUMMARY
    # ======================================================

    evaluation_columns = [
        "train/global_step",
        "train/epoch",
        "eval/loss",
        "eval/mean_token_accuracy",
        "eval/runtime",
        "eval/samples_per_second",
        "eval/steps_per_second",
    ]

    evaluation_columns = safe_columns(history, evaluation_columns)

    evaluation_df = history[evaluation_columns].copy()

    if "eval/loss" in evaluation_df.columns:
        evaluation_df = evaluation_df.dropna(subset=["eval/loss"])

    evaluation_df = evaluation_df.rename(
        columns={
            "train/global_step": "Global Step",
            "train/epoch": "Epoch",
            "eval/loss": "Eval Loss",
            "eval/mean_token_accuracy": "Eval Accuracy",
            "eval/runtime": "Runtime (s)",
            "eval/samples_per_second": "Samples/sec",
            "eval/steps_per_second": "Steps/sec",
        }
    )

    evaluation_df = evaluation_df.round(4)

    # ======================================================
    # SAVE
    # ======================================================

    training_file = OUTPUT_DIR / "training_summary.csv"
    evaluation_file = OUTPUT_DIR / "evaluation_summary.csv"

    training_df.to_csv(training_file, index=False)
    evaluation_df.to_csv(evaluation_file, index=False)

    print()
    print("=" * 80)
    print("Reports Generated Successfully")
    print("=" * 80)

    print(training_file)
    print(evaluation_file)

    print()
    print("Training Preview")
    print(training_df.head())

    print()
    print("Evaluation Preview")
    print(evaluation_df.head())


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    wandb_sync()