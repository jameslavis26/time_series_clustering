import numpy as np
from tqdm import tqdm

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import csv

from time_series.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, generate_datasets
from time_series.data_generators.lorenz_generator import generate_lorenz_curve

app = typer.Typer()

@app.command()
def main():
    for dataset_name, dataset_kwargs in generate_datasets.items():
        if dataset_kwargs["generator"] == "lorenz":
            logger.info(f"Generating Lorenz Curve {dataset_name}: {**dataset_kwargs['params']}")
            time, data = generate_lorenz_curve(**dataset_kwargs["params"])

            with open(f"{RAW_DATA_DIR}/{dataset_name}.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(["t", "x", "y", "z"])
                for t, x in zip(time, data):
                    writer.writerow([t, *x])

if __name__ == "__main__":
    app()
