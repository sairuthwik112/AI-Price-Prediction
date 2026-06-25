# ============================================================
# Import all required libraries for data loading and processing.
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

from pricer.items import Item
from pricer.parser import parse
from pricer.loaders import ItemLoader


# ============================================================
# Load environment variables and authenticate with Hugging Face.
# ============================================================

load_dotenv(override=True)

hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


# ============================================================
# Download the raw Amazon Appliances dataset from Hugging Face.
# ============================================================

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Appliances",
    split="full",
    trust_remote_code=True,
)

# print(f"Number of Appliances: {len(dataset):,}")
# print(dataset[6])


# ============================================================
# Find the most expensive product in the raw dataset.
# ============================================================

max_price = 0
max_item = None

for datapoint in tqdm(dataset, desc="Finding most expensive item"):
    try:
        price = float(datapoint["price"])

        if price > max_price:
            max_price = price
            max_item = datapoint

    except ValueError:
        pass

# print(f"The most expensive item is {max_item['title']} and costs ${max_price:,.2f}")


# ============================================================
# Convert raw records into clean Item objects.
# ============================================================

items = [
    parse(datapoint, "Appliances")
    for datapoint in tqdm(dataset, desc="Parsing Appliances")
]

items = [item for item in items if item is not None]

print(f"Valid Items : {len(items):,}")


# ============================================================
# Collect prices and text lengths for analysis.
# ============================================================

prices = [item.price for item in items]
lengths = [len(item.full) for item in items]


# ============================================================
# (Optional) Plot the distribution of text lengths.
# ============================================================

"""
plt.figure(figsize=(15,6))
plt.title(
    f"Lengths: Avg {sum(lengths)/len(lengths):,.0f} "
    f"and highest {max(lengths):,}"
)
plt.xlabel("Length (chars)")
plt.ylabel("Count")
plt.hist(
    lengths,
    bins=range(0,6000,100),
    color="lightblue",
    rwidth=0.7,
)
plt.show()

max_length = max(lengths)
max_length_item = items[lengths.index(max_length)]

print(max_length_item.full)
"""


# ============================================================
# (Optional) Plot the distribution of prices.
# ============================================================

"""
plt.figure(figsize=(15,6))
plt.title(
    f"Prices: Avg {sum(prices)/len(prices):,.2f} "
    f"and highest {max(prices):,}"
)
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.hist(
    prices,
    bins=range(0,1000,10),
    color="orange",
    rwidth=0.7,
)
plt.show()
"""
# ============================================================
# Load all previously curated datasets from every product category.
# ============================================================

dataset_names = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

items = []

for dataset_name in dataset_names:
    loader = ItemLoader(dataset_name)
    items.extend(loader.load())

print(f"A grand total of {len(items):,} items")


# ============================================================
# Shuffle the dataset before removing duplicates.
# ============================================================

random.seed(42)
random.shuffle(items)


# ============================================================
# Remove duplicate products based on their title.
# ============================================================

seen = set()

items = [
    item
    for item in tqdm(items, desc="Removing duplicate titles")
    if not (item.title in seen or seen.add(item.title))
]


# ============================================================
# Remove duplicate products based on their full description.
# ============================================================

seen = set()

items = [
    item
    for item in tqdm(items, desc="Removing duplicate descriptions")
    if not (item.full in seen or seen.add(item.full))
]

del seen

print(f"After deduplication, we have {len(items):,} items")


# ============================================================
# Count how many products belong to each category.
# ============================================================

category_counts = Counter(item.category for item in items)

categories = list(category_counts.keys())
counts = [category_counts[category] for category in categories]


# ============================================================
# Visualize the category distribution using a bar chart.
# ============================================================

plt.figure(figsize=(15, 6))

plt.bar(
    categories,
    counts,
    color="goldenrod"
)

plt.title("How many items are in each category")
plt.xlabel("Categories")
plt.ylabel("Count")

plt.xticks(rotation=30, ha="right")

for i, count in enumerate(counts):
    plt.text(
        i,
        count,
        f"{count:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# Create a weighted sample that favors higher-priced products.
# ============================================================

np.random.seed(42)

SIZE = 820_000

prices = np.array(
    [item.price for item in items],
    dtype=float
)

categories = np.array(
    [item.category for item in items]
)

normalized_prices = (
    prices - prices.min()
) / (
    prices.max() - prices.min() + 1e-9
)

weights = normalized_prices ** 2

weights[categories == "Tools_and_Home_Improvement"] *= 0.5
weights[categories == "Automotive"] *= 0.05

weights /= weights.sum()


sample_indices = np.random.choice(
    len(items),
    size=SIZE,
    replace=False,
    p=weights
)

sample = [items[index] for index in sample_indices]


# ============================================================
# Visualize the price distribution of the sampled dataset.
# ============================================================

prices = [item.price for item in sample]

plt.figure(figsize=(15, 6))

plt.title(
    f"Prices: Avg {sum(prices)/len(prices):,.1f} "
    f"Lowest {min(prices):,.2f} "
    f"Highest {max(prices):,.2f}"
)

plt.xlabel("Price ($)")
plt.ylabel("Count")

plt.hist(
    prices,
    bins=range(0, 1000, 10),
    color="blueviolet",
    rwidth=0.7,
)

plt.show()


# ============================================================
# Shuffle the sampled dataset before creating train/validation/test splits.
# ============================================================

random.seed(42)
random.shuffle(sample)
# ============================================================
# Analyze the sampled dataset by category.
# ============================================================

category_counts = Counter(item.category for item in sample)

categories = list(category_counts.keys())
counts = [category_counts[category] for category in categories]


# ============================================================
# Display the category distribution of the sampled dataset.
# ============================================================

plt.figure(figsize=(15, 6))

plt.bar(categories, counts, color="goldenrod")

plt.title("How many items are in each category")
plt.xlabel("Categories")
plt.ylabel("Count")

plt.xticks(rotation=30, ha="right")

for i, count in enumerate(counts):
    plt.text(i, count, f"{count:,}", ha="center", va="bottom")

plt.tight_layout()
plt.show()


# ============================================================
# Display the category distribution as a donut chart.
# ============================================================

plt.figure(figsize=(12, 10))

plt.pie(
    counts,
    labels=categories,
    autopct="%1.0f%%",
    startangle=90,
)

centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Category Distribution")
plt.axis("equal")

plt.show()


# ============================================================
# Visualize the relationship between text length and price.
# ============================================================

sizes = [len(item.full) for item in sample]
prices = [item.price for item in sample]

plt.figure(figsize=(15, 8))

plt.scatter(
    sizes,
    prices,
    s=0.2,
    color="red",
)

plt.xlabel("Description Length")
plt.ylabel("Price ($)")
plt.title("Price vs Description Length")

plt.show()


# ============================================================
# Visualize the relationship between product weight and price.
# ============================================================

weights = [item.weight for item in sample]
prices = [item.price for item in sample]

plt.figure(figsize=(15, 8))

plt.scatter(
    weights,
    prices,
    s=0.2,
    color="darkorange",
)

plt.xlabel("Weight (ounces)")
plt.ylabel("Price ($)")
plt.xlim(0, 400)

plt.title("Price vs Weight")

plt.show()


# ============================================================
# Create the Train, Validation and Test splits.
# ============================================================

username = "your_username"

full_dataset_name = f"{username}/items_raw_full"
lite_dataset_name = f"{username}/items_raw_lite"

train = sample[:800_000]
validation = sample[800_000:810_000]
test = sample[810_000:]


# ============================================================
# Upload the complete curated dataset to Hugging Face.
# ============================================================

Item.push_to_hub(
    full_dataset_name,
    train,
    validation,
    test,
)


# ============================================================
# Create a lightweight version of the dataset.
# ============================================================

train_lite = train[:20_000]
validation_lite = validation[:1_000]
test_lite = test[:1_000]


# ============================================================
# Upload the lightweight dataset to Hugging Face.
# ============================================================

Item.push_to_hub(
    lite_dataset_name,
    train_lite,
    validation_lite,
    test_lite,
)

print("\nDataset curation completed successfully!")
print(f"Full Dataset : {full_dataset_name}")
print(f"Lite Dataset : {lite_dataset_name}")
