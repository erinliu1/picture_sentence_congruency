from pathlib import Path

import json
import os
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

HUMAN_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv")

human_df = pd.read_csv(HUMAN_PATH)
human_df = (human_df.groupby(["item_index", "image_word", "condition"], as_index=False)["rating"].mean())

def get_alignment(model_id):
    model_dir = MODELS_DIR / model_id
    output_dir = model_dir / "alignment"
    output_dir.mkdir(exist_ok=True)

    classifier_results_df = pd.read_csv(model_dir / "decoding" / "results.csv")

    combined_df = classifier_results_df.merge(
        human_df,
        on=["item_index", "image_word", "condition"],
        how="left"
    ).drop(columns=["pass"])

    coefficients = []
    for layer_index in combined_df["layer_index"].unique():

        layer_df = combined_df[
            combined_df.layer_index == layer_index
        ].copy()

        layer_df["probability_z"] = (
            layer_df["probability_congruent"]
            - layer_df["probability_congruent"].mean()
        ) / layer_df["probability_congruent"].std()

        model = smf.ols(
            "rating ~ probability_z + C(item_index)",
            data=layer_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": layer_df["item_index"]}
        )

        coefficients.append({
            "layer_index": layer_index,
            "coefficient": model.params["probability_z"],
            "se": model.bse["probability_z"],
            "p_value": model.pvalues["probability_z"]
        })

    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df["p_value_fdr"] = multipletests(coefficients_df["p_value"], method="fdr_bh")[1]
    coefficients_df["significant_fdr"] = (coefficients_df["p_value_fdr"] < 0.05)
    coefficients_df.to_csv(output_dir / "coefficients.csv", index=False)
    print(f"Alignment coefficients for {model_id} saved.")