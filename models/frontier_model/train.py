import os
from dotenv import load_dotenv
from litellm import completion

from pricer.items import Item
from pricer.evaluator import evaluate

############################################################
# Load API Key
############################################################

load_dotenv(override=True)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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
Estimate the price of this product.

Respond ONLY with the numeric price.

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

def groq_llama(item):

    response = completion(

        model="groq/llama-3.3-70b-versatile",

        messages=messages_for(item),

        temperature=0

    )

    return response.choices[0].message.content.strip()

############################################################
# Test One Prediction
############################################################

prediction = groq_llama(test[0])

print("=" * 60)
print("Prediction :", prediction)
print("Actual     :", test[0].price)
print("=" * 60)

############################################################
# Evaluate
############################################################

evaluate(groq_llama, test[:29], workers=1)