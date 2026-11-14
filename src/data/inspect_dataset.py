
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_dataset(filename):

    filepath = RAW_DIR / filename

    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    articles = dataset["data"]

    total_questions = 0
    answerable = 0
    unanswerable = 0

    unique_passages = 0
    unique_articles = set()

    for article in articles:

        title = article["title"]
        unique_articles.add(title)

        for paragraph in article["paragraphs"]:

            unique_passages += 1

            for qa in paragraph["qas"]:

                total_questions += 1

                if qa["is_impossible"]:
                    unanswerable += 1
                else:
                    answerable += 1

    print("\n" + "=" * 50)
    print(f"DATASET: {filename}")
    print("=" * 50)

    print(f"Articles:             {len(unique_articles)}")
    print(f"Passages:             {unique_passages}")
    print(f"Total questions:      {total_questions}")
    print(f"Answerable:           {answerable}")
    print(f"Unanswerable:         {unanswerable}")

    print(f"Answerable ratio:     {answerable / total_questions:.2%}")
    print(f"Unanswerable ratio:   {unanswerable / total_questions:.2%}")

    print("\nExample question:")

    first_article = articles[0]
    first_paragraph = first_article["paragraphs"][0]
    first_qa = first_paragraph["qas"][0]

    print("Article:", first_article["title"])
    print("Question:", first_qa["question"])
    print("Context:", first_paragraph["context"][:300])

    print("Unanswerable:", first_qa["is_impossible"])

    if not first_qa["is_impossible"]:
        print("Answer:", first_qa["answers"][0]["text"])


def main():
    inspect_dataset("train-v2.0.json")
    inspect_dataset("dev-v2.0.json")


if __name__ == "__main__":
    main()
