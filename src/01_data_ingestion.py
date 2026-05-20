import os
import requests
import pandas as pd

DATASETS = {
    "declarations": {
        "version": "v2",
        "name": "DisasterDeclarationsSummaries"
    },
    "public_assistance": {
        "version": "v2",
        "name": "PublicAssistanceFundedProjectsDetails"
    },
    "disaster_summaries": {
        "version": "v1",
        "name": "FemaWebDisasterSummaries"
    }
}

os.makedirs("data/raw", exist_ok=True)


def fetch_dataset(dataset_info, batch_size=10000):

    version = dataset_info["version"]
    dataset_name = dataset_info["name"]

    all_records = []

    skip = 0

    while True:

        url = f"https://www.fema.gov/api/open/{version}/{dataset_name}"

        params = {
            "$top": batch_size,
            "$skip": skip,
            "$format": "json"
        }

        print(f"Fetching rows {skip} to {skip + batch_size}...")

        response = requests.get(url, params=params)

        response.raise_for_status()

        data = response.json()

        records = data.get(dataset_name, [])

        # Stop if no more records
        if not records:
            break

        all_records.extend(records)

        skip += batch_size

    return pd.DataFrame(all_records)


def main():
    for file_name, dataset_info in DATASETS.items():
        print(f"\nLoading {dataset_info['name']}...")

        df = fetch_dataset(dataset_info)

        output_path = f"data/raw/{file_name}.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()