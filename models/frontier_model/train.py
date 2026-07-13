import os
from litellm import completion

from pricer.items import Item
from pricer.evaluator import evaluate
############################################################
# Load Dataset
############################################################

LITE_MODE = True

username = "ed-donner"

dataset = (
    f"{username}/items_lite"
    if LITE_MODE
    else f"{username}/items_full"
)

train, val, test = Item.from_hub(dataset)

print(
    f"Loaded {len(train):,} training items, "
    f"{len(val):,} validation items, "
    f"{len(test):,} test items"
)

############################################################
# Prompt
############################################################

def messages_for(item):

    message = f"""
    You are an expert e-commerce pricing model.

    Estimate the market price of the following product.

    Rules:
    - Output ONLY the numeric price.
    - Do not include '$', commas, or explanations.
    - If uncertain, estimate the most likely selling price.

    Product Description:

    {item.summary}
    """
    return [
        {
            "role": "user",
            "content": message
        }
    ]

############################################################
# Frontier Model
############################################################

def gemma3(item):

    response = completion(
        model="ollama/gemma3:4b",
        api_base="http://localhost:11434",
        messages=messages_for(item),
        temperature=0
    )

    return response.choices[0].message.content.strip()

############################################################
# Test One Prediction
############################################################

prediction = gemma3(test[0])

print("=" * 60)
print("Prediction :", prediction)
print("Actual     :", test[0].price)
print("=" * 60)

############################################################
# Evaluate
############################################################

evaluate(gemma3, test)
