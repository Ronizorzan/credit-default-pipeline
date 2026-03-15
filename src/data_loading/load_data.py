import logging
import os

import numpy as np
import pandas as pd
import kagglehub


logger = logging.getLogger("src.data_loading.load_data")


def fetch_data() -> pd.DataFrame:
    """Fetch credit card default dataset and convert to DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the credit card default data with features and target
    """
    logger.info("Fetching data...")
    folder_path = kagglehub.dataset_download("d4rklucif3r/defaulter")
    full_path = os.path.join(folder_path, "credit_card_defaulter.csv")
    
    # Reads the dataframe from cache with pandas
    dataset = pd.read_csv(full_path)

    # Renames Target column 
    dataset["target"] = dataset["default"]
    
    # Drop irrelevant and duplicated columns 
    dataset.drop(columns=["Unnamed: 0", "default"], inplace=True, errors="ignore")                
        
    return dataset


def save_data(data: pd.DataFrame) -> None:
    """Save the raw data to disk.

    Args:
        data (pd.DataFrame): Credit Card Default dataset to save
    """
    output_path = "data/raw/raw.csv"
    logger.info(f"Saving raw data to {output_path}")
    data.to_csv(output_path, index=False)


def main() -> None:
    """Main function to orchestrate the data loading process."""
    raw_data = fetch_data()
    save_data(raw_data)
    logger.info("Data loading completed")


if __name__ == "__main__":
    main()
