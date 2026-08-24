from pathlib import Path

import json
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

CV_DIR = "/Intern/Erin/picture_sentence_congruency/decoding/cv"

with open(f"{CV_DIR}/cv_lookup_pt.json", "r") as f:
    CV_LOOKUP = json.load(f)

N_FOLDS = len(CV_LOOKUP)
SEED = 42

def unpack_item(item):
    parts = item.split("_")
    item_index = int(parts[0])
    image_word = parts[1]
    condition = parts[2].split(".")[0]
    return item_index, image_word, condition

def get_label(condition, reverse=False):
    if reverse:
        if condition == 'congruent':
            return 0
        else:
            return 1
    else:
        if condition == 'congruent':
            return 1
        else:
            return 0

def get_item(hidden_states_dir, pt_list, reverse=False):
    X, y = [], []
    for item_name in pt_list:
        item_index, image_word, condition = unpack_item(item_name)
        hidden_state = torch.load(f"{hidden_states_dir}/{item_name}", map_location="cpu", weights_only=False)
        X.append(hidden_state)
        y.append(get_label(condition, reverse=reverse))
    return X, y

def compute_layerwise_accuracy(results_df):
    layerwise_accuracy = []
    for layer_index in sorted(results_df["layer_index"].unique()):
        layer_df = results_df[results_df["layer_index"] == layer_index]
        n_correct = (layer_df["pass"] == "✅").sum()
        n_total = len(layer_df)
        accuracy = n_correct / n_total
        congruent_df = layer_df[layer_df["condition"] == "congruent"]
        n_congruent_correct = (congruent_df["pass"] == "✅").sum()
        n_congruent_total = len(congruent_df)
        congruent_accuracy = n_congruent_correct / n_congruent_total
        incongruent_df = layer_df[layer_df["condition"] == "incongruent"]
        n_incongruent_correct = (incongruent_df["pass"] == "✅").sum()
        n_incongruent_total = len(incongruent_df)
        incongruent_accuracy = n_incongruent_correct / n_incongruent_total
        layerwise_accuracy.append({
            "layer_index": layer_index,
            "all_accuracy": accuracy,
            "congruent_accuracy": congruent_accuracy,
            "incongruent_accuracy": incongruent_accuracy,
        })
    return pd.DataFrame(layerwise_accuracy)

# no label reversal for the real (non-permuted) run
def lr_train(model_id, reverse_indices=[]):
    model_dir = MODELS_DIR / model_id
    hidden_states_dir = model_dir / "hidden_states"

    output_dir = model_dir / "decoding"
    output_dir.mkdir(exist_ok=True)

    if (output_dir / 'results.csv').exists() and (output_dir / 'layerwise_accuracy.csv').exists():
        print(f"Decoding results already exist for {model_id}, skipping training.")
        return

    title = models_lookup[model_id]["title"]
    results = []
    for fold in tqdm(range(1, N_FOLDS + 1), desc=f"Decoding for {title}"):
        train_items = CV_LOOKUP[str(fold)]["train"]
        test_items = CV_LOOKUP[str(fold)]["test"]

        X_train, y_train = [], []
        for item_index, pt_list in train_items.items():
            reverse = int(item_index) in reverse_indices
            X_item, y_item = get_item(hidden_states_dir, pt_list, reverse=reverse)
            X_train.extend(X_item)
            y_train.extend(y_item)

        X_train = torch.stack(X_train, dim=0)
        y_train = torch.tensor(y_train, dtype=torch.long)

        X = X_train.float().numpy() 
        y = y_train.numpy() 

        classifiers = {}

        N_LAYERS = X.shape[1]

        for layer_index in range(N_LAYERS):
            X_layer = X[:, layer_index, :] 

            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    solver="liblinear",
                    max_iter=10_000,
                    random_state=SEED,
                ),
            )

            classifier.fit(X_layer, y)
            classifiers[layer_index] = classifier

        for item_index, pt_list in test_items.items():
            reverse = int(item_index) in reverse_indices
            for item_name in pt_list:
                _, image_word, condition = unpack_item(item_name)
                X_test = torch.load(f"{hidden_states_dir}/{item_name}", map_location="cpu", weights_only=False).float().numpy() 
                label = get_label(condition, reverse=reverse)
                condition = 'congruent' if label == 1 else 'incongruent'

                for layer_index, classifier in classifiers.items():
                    X_test_layer = X_test[layer_index, :].reshape(1, -1) # 
                    predicted_label = int(classifier.predict(X_test_layer)[0])
                    probability_congruent = float(classifier.predict_proba(X_test_layer)[0][1])
                    results.append({
                        'layer_index': layer_index,
                        'item_index': item_index,
                        'image_word': image_word,
                        'condition': condition,
                        'pass': '✅' if predicted_label == label else '❌',
                        'probability_congruent': probability_congruent
                    })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['layer_index', 'item_index']).reset_index(drop=True)
    results_df.to_csv(output_dir / "results.csv", index=False)
    print(f"Decoding results for {model_id} saved.")

    accuracy_df = compute_layerwise_accuracy(results_df)
    accuracy_df.to_csv(output_dir / "layerwise_accuracy.csv", index=False)
    print(f"Layerwise accuracy for {model_id} saved.")