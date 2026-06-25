# ============================================================
# Import the required libraries for LLM inference, dataset
# loading, JSON handling and batch processing.
# ============================================================

import json
import os
from groq import Groq
from dotenv import load_dotenv
from litellm import completion

from pricer.batch import Batch
from pricer.items import Item


# ============================================================
# Load environment variables from the .env file.
# ============================================================

load_dotenv(override=True)


# ============================================================
# Choose whether to use the Lite dataset or the Full dataset.
#
# True  -> 20,000 training samples (Fast & Free)
# False -> 800,000 training samples (Large Dataset)
# ============================================================

LITE_MODE = True


# ============================================================
# Load the curated dataset from Hugging Face.
# ============================================================

username = "ed-donner"

dataset = (
    f"{username}/items_raw_lite"
    if LITE_MODE
    else f"{username}/items_raw_full"
)

train, validation, test = Item.from_hub(dataset)

items = train + validation + test

print(f"Loaded {len(items):,} items")
print(items[0])


# ============================================================
# Assign a unique ID to every product.
#
# These IDs will later be used to match the batch responses
# with the original products.
# ============================================================

for index, item in enumerate(items):
    item.id = index


# ============================================================
# Create the system prompt that instructs the language model
# how to rewrite every product.
# ============================================================

SYSTEM_PROMPT = """
Create a concise description of a product.

Respond ONLY using the following format.

Do not include part numbers.

Title: Rewritten short precise title
Category: Electronics
Brand: Brand name
Description: One sentence description
Details: One sentence describing important features.
"""


# ============================================================
# Display one raw product description before preprocessing.
# ============================================================

print(items[0].full)


# ============================================================
# Test the prompt using the Groq hosted GPT-OSS model.
# ============================================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": items[0].full,
    },
]

response = completion(
    model="groq/openai/gpt-oss-20b",
    messages=messages,
    reasoning_effort="low",
)

print(response.choices[0].message.content)
print()

print(f"Input Tokens  : {response.usage.prompt_tokens}")
print(f"Output Tokens : {response.usage.completion_tokens}")
print(
    f"Cost          : {response._hidden_params['response_cost'] * 100:.3f} cents"
)


# ============================================================
# Test the same prompt using a locally running Ollama model.
# ============================================================

response = completion(
    model="ollama/llama3.2",
    api_base="http://localhost:11434",
    messages=messages,
)

print(response.choices[0].message.content)
print()

print(f"Input Tokens  : {response.usage.prompt_tokens}")
print(f"Output Tokens : {response.usage.completion_tokens}")
print(
    f"Cost          : {response._hidden_params['response_cost'] * 100:.3f} cents"
)


# ============================================================
# Select the model that will be used for batch inference.
# ============================================================

MODEL = "openai/gpt-oss-20b"
# ============================================================
# Convert a product into a JSONL request for batch inference.
# ============================================================

def make_jsonl(item):
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": item.full,
            },
        ],
        "reasoning_effort": "low",
    }

    request = {
        "custom_id": str(item.id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }

    return json.dumps(request)


# ============================================================
# Display an example JSONL request.
# ============================================================

print(items[0])
print(make_jsonl(items[0]))


# ============================================================
# Create a JSONL file containing a batch of products.
# ============================================================

def make_file(start, end, filename):

    with open(filename, "w", encoding="utf-8") as file:

        for index in range(start, end):
            file.write(make_jsonl(items[index]))
            file.write("\n")


# ============================================================
# Generate a sample batch file.
# ============================================================

make_file(0, 1000, "jsonl/0_1000.jsonl")


# ============================================================
# Create the Groq client.
# ============================================================

groq = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# ============================================================
# Upload the JSONL batch file to Groq.
# ============================================================

with open("jsonl/0_1000.jsonl", "rb") as file:

    response = groq.files.create(
        file=file,
        purpose="batch",
    )

print(response)


# ============================================================
# Store the uploaded file ID.
# ============================================================

file_id = response.id

print(file_id)


# ============================================================
# Create a batch inference job.
# ============================================================

response = groq.batches.create(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=file_id,
)

print(response)


# ============================================================
# Check the current status of the batch job.
# ============================================================

result = groq.batches.retrieve(response.id)

print(result)


# ============================================================
# Download the completed batch results.
# ============================================================

response = groq.files.content(result.output_file_id)

response.write_to_file("jsonl/batch_results.jsonl")


# ============================================================
# Attach every generated summary back to its product.
# ============================================================

with open(
    "jsonl/batch_results.jsonl",
    "r",
    encoding="utf-8",
) as file:

    for line in file:

        json_line = json.loads(line)

        item_id = int(json_line["custom_id"])

        summary = (
            json_line["response"]["body"]["choices"][0]
            ["message"]["content"]
        )

        items[item_id].summary = summary


# ============================================================
# Display the original product description.
# ============================================================

print(items[0].full)


# ============================================================
# Display the rewritten product description.
# ============================================================

print(items[1000].summary)
# ============================================================
# Use the Batch class to automate the preprocessing pipeline.
# ============================================================

Batch.create(items, LITE_MODE)


# ============================================================
# Submit all batches to the language model provider.
# ============================================================

Batch.run()


# ============================================================
# Download all completed batch results.
# ============================================================

Batch.fetch()


# ============================================================
# Verify that every product received a summary.
# ============================================================

for index, item in enumerate(items):
    if not item.summary:
        print(index)


# ============================================================
# Display one rewritten product description.
# ============================================================

print(items[10234].summary)


# ============================================================
# Remove unnecessary fields before uploading the dataset.
# ============================================================

for item in items:
    item.full = None
    item.id = None


# ============================================================
# Configure the Hugging Face dataset names.
# ============================================================

username = "ed-donner"

full_dataset = f"{username}/items_full"
lite_dataset = f"{username}/items_lite"


# ============================================================
# Split the processed dataset and upload it to Hugging Face.
# ============================================================

if LITE_MODE:

    train = items[:20_000]
    validation = items[20_000:21_000]
    test = items[21_000:]

    Item.push_to_hub(
        lite_dataset,
        train,
        validation,
        test,
    )

else:

    train = items[:800_000]
    validation = items[800_000:810_000]
    test = items[810_000:]

    Item.push_to_hub(
        full_dataset,
        train,
        validation,
        test,
    )

    train_lite = train[:20_000]
    validation_lite = validation[:1_000]
    test_lite = test[:1_000]

    Item.push_to_hub(
        lite_dataset,
        train_lite,
        validation_lite,
        test_lite,
    )


# ============================================================
# Display a completion message after uploading the datasets.
# ============================================================

print("=" * 60)
print("Data preprocessing completed successfully.")
print(f"Lite Mode : {LITE_MODE}")

if LITE_MODE:
    print(f"Dataset uploaded : {lite_dataset}")
else:
    print(f"Full Dataset uploaded : {full_dataset}")
    print(f"Lite Dataset uploaded : {lite_dataset}")

print("=" * 60)