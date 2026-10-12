"""Save frozen CLIP ViT-L/14 similarities for all 320 stimulus combinations."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'CLIP' / '.cache' / 'huggingface'))

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def save_similarity():
    with (ROOT / 'stimuli/sentences.json').open() as handle:
        items = json.load(handle)
    rows = []
    for index, item in enumerate(items):
        for image_word in item['word_options']:
            for final_word in item['word_options']:
                rows.append({
                    'item_index': index,
                    'image_word': image_word,
                    'condition': 'congruent' if image_word == final_word else 'incongruent',
                    'sentence': f"{item['sentence_frame']} {final_word}.",
                })
    table = pd.DataFrame(rows)
    if len(table) != 320:
        raise ValueError(f'Expected 320 combinations, got {len(table)}')
    keys = ['item_index', 'image_word', 'condition']
    human = pd.read_csv(ROOT / 'survey/results/ratings.csv')
    human = human.groupby(keys, as_index=False)['rating'].mean()
    human = human.rename(columns={'rating': 'human_mean_rating'})
    table = table.merge(human, on=keys, how='left', validate='one_to_one')
    if table['human_mean_rating'].isna().any():
        raise ValueError('Missing human ratings for stimulus combinations')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = 'openai/clip-vit-large-patch14'
    processor = CLIPProcessor.from_pretrained(checkpoint)
    model = CLIPModel.from_pretrained(checkpoint).to(device).eval()
    model.requires_grad_(False)
    image_embeddings, text_embeddings = {}, {}
    batch_size = 16
    with torch.inference_mode():
        words = table['image_word'].unique().tolist()
        for start in range(0, len(words), batch_size):
            batch = words[start:start + batch_size]
            images = []
            for word in batch:
                with Image.open(ROOT / 'stimuli/pictures' / f'{word}.png') as image:
                    images.append(image.convert('RGB'))
            inputs = processor(images=images, return_tensors='pt').to(device)
            output = model.vision_model(pixel_values=inputs['pixel_values'])
            embeddings = F.normalize(model.visual_projection(output.pooler_output).float(), dim=-1)
            image_embeddings.update(zip(batch, embeddings.cpu()))
            print(f'Encoded images: {min(start + batch_size, len(words))}/{len(words)}', flush=True)
        sentences = table['sentence'].unique().tolist()
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            # Preserve the full sentence, including the critical word; no template or truncation.
            inputs = processor(text=batch, padding=True, return_tensors='pt').to(device)
            output = model.text_model(**inputs)
            embeddings = F.normalize(model.text_projection(output.pooler_output).float(), dim=-1)
            text_embeddings.update(zip(batch, embeddings.cpu()))
            print(f'Encoded sentences: {min(start + batch_size, len(sentences))}/{len(sentences)}', flush=True)
    table['clip_similarity'] = [
        torch.dot(image_embeddings[row.image_word], text_embeddings[row.sentence]).item()
        for row in table.itertuples(index=False)
    ]
    if not torch.isfinite(torch.tensor(table['clip_similarity'].to_numpy())).all():
        raise ValueError('CLIP produced nonfinite similarities')
    output_path = ROOT / 'CLIP/results/similarity.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table[keys + ['human_mean_rating', 'clip_similarity']].to_csv(output_path, index=False)
    print(f'Saved {len(table)} rows to {output_path}', flush=True)
    return output_path


if __name__ == '__main__':
    save_similarity()
