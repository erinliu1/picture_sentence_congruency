"""Exact PLM behavioral prompt and assistant-prefix token construction."""
import torch

SYSTEM_PROMPT = """
You will be shown one picture and one sentence. Your task is to judge how compatible the final word of the sentence is with the situation shown in the picture.

When making your judgment:
* Focus on the final word of the sentence. Treat the rest of the sentence as given.
* The picture does not need to prove that the sentence ending is the only possible interpretation. Simply, decide whether the final word is a natural and reasonable interpretation of the picture.

Use the following rating scale:
1 - Not compatible at all. The final word is clearly inconsistent with / unrelated to / contradicts the picture.
2 - Low compatibility. The final word is technically plausible but would not be commonly associated with the picture / it would be unusual or difficult to reconcile this sentence with the picture.
3 - Uncertain. The final word provides weak / ambiguous / mixed evidence for the picture.
4 - Compatible. The final word makes the sentence a reasonable interpretation of the picture.
5 - Very compatible. The final word makes the sentence very consistent with / a natural interpretation of the picture.

Respond with exactly one token: 1, 2, 3, 4, or 5. Do not output any additional text.
""".strip()

ASSISTANT_PREFIX = "Rating: "

def build_rating_inputs(processor, image, sentence):
    ASSISTANT_PREFIX_IDS = processor.tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    messages = [
        # PLM's chat_template.jinja special-cases the system message as a plain string, unlike every other role
        # (and all other models' system messages), which expect the structured content-list format.
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": sentence,
                },
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    prefix_ids = torch.tensor([ASSISTANT_PREFIX_IDS], dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(prefix_ids)], dim=1
    )
    return inputs
