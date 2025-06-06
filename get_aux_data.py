import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re
from typing import Optional, List, Union


class AuxiliaryDatasetCreator:
    """
    Utility class for creating auxiliary datasets from various data sources
    for use in probability ratio reconstruction methods.
    """

    def __init__(self, base_path: str = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/"):
        """
        Initialize with base path to data directories.

        Args:
            base_path: Base path containing the data directories
        """
        self.base_path = Path(base_path)
        self.practice_problem_path = self.base_path / "25_PracticeProblem"
        self.red_team_path = self.base_path / "NIST_Red-Team_Problems1-24_v2"

    def auxiliary_creation_1(self) -> pd.DataFrame:
        """
        Auxiliary creation method 1: Simply use the original practice problem data.

        Returns:
            pd.DataFrame: The original 25_Demo_25f_OriginalData.csv dataset
        """
        original_file = self.practice_problem_path / "25_Demo_25f_OriginalData.csv"

        if not original_file.exists():
            raise FileNotFoundError(f"Original data file not found: {original_file}")

        return pd.read_csv(original_file)

    def auxiliary_creation_2(self, problem_range: range = range(1, 25)) -> pd.DataFrame:
        """
        Auxiliary creation method 2: Combine all deidentified datasets from problems 1-24.

        Args:
            problem_range: Range of problem numbers to include (default: 1-24)

        Returns:
            pd.DataFrame: Combined dataset from all specified problems
        """
        combined_datasets = []

        for problem_num in problem_range:
            # Find all deidentified files for this problem number
            pattern = f"{problem_num}_*_Deid.csv"
            deid_files = list(self.red_team_path.glob(pattern))

            if not deid_files:
                print(f"Warning: No deidentified files found for problem {problem_num}")
                continue

            for deid_file in deid_files:
                try:
                    df = pd.read_csv(deid_file)
                    # Add metadata columns to track source
                    df['source_problem'] = problem_num
                    df['source_file'] = deid_file.name
                    combined_datasets.append(df)
                    print(f"Added {len(df)} rows from {deid_file.name}")
                except Exception as e:
                    print(f"Error reading {deid_file}: {e}")

        if not combined_datasets:
            raise ValueError("No datasets were successfully loaded")

        # Combine all datasets
        result = pd.concat(combined_datasets, ignore_index=True, sort=False)
        print(f"Total combined dataset size: {len(result)} rows")

        return result

    def auxiliary_creation_3(self, exclude_problem_num: int,
                             problem_range: range = range(1, 25)) -> pd.DataFrame:
        """
        Auxiliary creation method 3: Same as method 2, but exclude specific problem number.

        Args:
            exclude_problem_num: Problem number to exclude from the combination
            problem_range: Range of problem numbers to consider (default: 1-24)

        Returns:
            pd.DataFrame: Combined dataset excluding the specified problem
        """
        # Filter out the excluded problem number
        filtered_range = [i for i in problem_range if i != exclude_problem_num]

        print(f"Creating auxiliary dataset excluding problem {exclude_problem_num}")
        print(f"Including problems: {filtered_range}")

        return self.auxiliary_creation_2(range(min(filtered_range), max(filtered_range) + 1))


    def auxiliary_creation_4(self, current_deid_file: str) -> pd.DataFrame:
        """
        Auxiliary creation method 4: Use complementary QID dataset.
        If using QID1 dataset, return QID2 dataset and vice versa.

        Args:
            current_deid_file: Current deidentified file being used
                             (e.g., "5_Method_QID1_Deid.csv")

        Returns:
            pd.DataFrame: Complementary QID dataset
        """
        # Parse the current filename to extract components
        filename = Path(current_deid_file).name if '/' in current_deid_file else current_deid_file

        # Extract problem number, method, and QID from filename
        # Expected format: "i_Method_QIDx_Deid.csv"
        match = re.match(r'^(\d+)_(.+)_QID([12])_Deid\.csv$', filename)
        if not match:
            raise ValueError(f"Cannot parse QID from filename: {filename}. "
                             f"Expected format: 'N_Method_QIDx_Deid.csv'")

        problem_num, method_name, current_qid = match.groups()

        # Determine complementary QID
        complementary_qid = '2' if current_qid == '1' else '1'

        # Construct complementary filename
        complementary_filename = f"{problem_num}_{method_name}_QID{complementary_qid}_Deid.csv"
        complementary_path = self.red_team_path / complementary_filename

        if not complementary_path.exists():
            raise FileNotFoundError(f"Complementary QID file not found: {complementary_path}")

        print(f"Using QID{current_qid} dataset: {filename}")
        print(f"Loading complementary QID{complementary_qid} dataset: {complementary_filename}")

        try:
            df = pd.read_csv(complementary_path)
            print(f"Loaded complementary dataset: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            raise IOError(f"Error reading complementary file {complementary_path}: {e}")


    def auxiliary_creation_5(self, problem_range: range = range(1, 25)) -> pd.DataFrame:
        """
        Auxiliary creation method 5: Combine only RANKSWAP deidentified datasets.

        Args:
            problem_range: Range of problem numbers to include (default: 1-24)

        Returns:
            pd.DataFrame: Combined dataset from all RANKSWAP files in specified problems
        """
        combined_datasets = []

        for problem_num in problem_range:
            # Find all deidentified files for this problem number that contain RANKSWAP
            pattern = f"{problem_num}_*_Deid.csv"
            all_deid_files = list(self.red_team_path.glob(pattern))

            # Filter for files containing RANKSWAP in the name
            rankswap_files = [f for f in all_deid_files if "RANKSWAP" in f.name]

            if not rankswap_files:
                print(f"Warning: No RANKSWAP files found for problem {problem_num}")
                continue

            for rankswap_file in rankswap_files:
                try:
                    df = pd.read_csv(rankswap_file)
                    # Add metadata columns to track source
                    df['source_problem'] = problem_num
                    df['source_file'] = rankswap_file.name
                    combined_datasets.append(df)
                    print(f"Added {len(df)} rows from {rankswap_file.name}")
                except Exception as e:
                    print(f"Error reading {rankswap_file}: {e}")

        if not combined_datasets:
            raise ValueError("No RANKSWAP datasets were successfully loaded")

        # Combine all datasets
        result = pd.concat(combined_datasets, ignore_index=True, sort=False)
        print(f"Total combined RANKSWAP dataset size: {len(result)} rows")

        return result

    def get_available_problems(self) -> List[int]:
        """
        Get list of available problem numbers in the red team directory.

        Returns:
            List[int]: Sorted list of available problem numbers
        """
        pattern = "*_*_Deid.csv"
        deid_files = list(self.red_team_path.glob(pattern))

        problem_numbers = set()
        for file in deid_files:
            # Extract problem number from filename (assumes format: "N_..._Deid.csv")
            match = re.match(r'^(\d+)_', file.name)
            if match:
                problem_numbers.add(int(match.group(1)))

        return sorted(list(problem_numbers))

    def create_stratified_auxiliary(self, method: int = 1,
                                    sample_fraction: float = 0.2,
                                    random_state: int = 42) -> pd.DataFrame:
        """
        Create a stratified sample auxiliary dataset.

        Args:
            method: Which auxiliary creation method to use as base (1, 2, or 3)
            sample_fraction: Fraction of data to sample
            random_state: Random seed for reproducibility

        Returns:
            pd.DataFrame: Stratified sample of the auxiliary dataset
        """
        if method == 1:
            base_data = self.auxiliary_creation_1()
        elif method == 2:
            base_data = self.auxiliary_creation_2()
        else:
            raise ValueError("method must be 1 or 2 for stratified sampling")

        # If there's a column that can be used for stratification, use it
        # Otherwise, just random sample
        if 'source_problem' in base_data.columns:
            # Stratify by source problem
            return base_data.groupby('source_problem', group_keys=False).apply(
                lambda x: x.sample(frac=sample_fraction, random_state=random_state)
            ).reset_index(drop=True)
        else:
            return base_data.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)


    def validate_auxiliary_dataset(self, aux_data: pd.DataFrame,
                                   target_columns: List[str]) -> bool:
        """
        Validate that auxiliary dataset has required columns and reasonable data.

        Args:
            aux_data: Auxiliary dataset to validate
            target_columns: Required columns that should be present

        Returns:
            bool: True if dataset is valid, False otherwise
        """
        # Check if required columns exist
        missing_cols = set(target_columns) - set(aux_data.columns)
        if missing_cols:
            print(f"Warning: Missing columns in auxiliary data: {missing_cols}")
            return False

        # Check for empty dataset
        if len(aux_data) == 0:
            print("Warning: Auxiliary dataset is empty")
            return False

        # Check for excessive missing values
        missing_pct = aux_data[target_columns].isnull().sum() / len(aux_data)
        high_missing = missing_pct[missing_pct > 0.5]
        if not high_missing.empty:
            print(f"Warning: High missing values in columns: {high_missing.to_dict()}")

        print(f"Auxiliary dataset validation passed: {len(aux_data)} rows, {len(aux_data.columns)} columns")
        return True

def create_auxiliary_complementary_qid(current_deid_file: str) -> pd.DataFrame:
    """Quick function to create auxiliary dataset using complementary QID."""
    creator = AuxiliaryDatasetCreator()
    return creator.auxiliary_creation_4(current_deid_file)


# Convenience functions for easy import and use
def create_auxiliary_original() -> pd.DataFrame:
    """Quick function to create auxiliary dataset using original data."""
    creator = AuxiliaryDatasetCreator()
    return creator.auxiliary_creation_1()


def create_auxiliary_combined(exclude_problem: Optional[int] = None) -> pd.DataFrame:
    """Quick function to create auxiliary dataset using combined deidentified data."""
    creator = AuxiliaryDatasetCreator()
    if exclude_problem is not None:
        return creator.auxiliary_creation_3(exclude_problem)
    else:
        return creator.auxiliary_creation_2()


def get_auxiliary_for_problem(current_problem_file: str) -> pd.DataFrame:
    """
    Get appropriate auxiliary dataset for a given problem file.
    Automatically determines which problem to exclude based on filename.

    Args:
        current_problem_file: Filename of current problem being processed

    Returns:
        pd.DataFrame: Auxiliary dataset excluding the current problem
    """
    creator = AuxiliaryDatasetCreator()

    # Extract problem number from filename
    match = re.match(r'^(\d+)_', current_problem_file)
    if match:
        problem_num = int(match.group(1))
        return creator.auxiliary_creation_3(problem_num)
    else:
        # If we can't parse problem number, use method 2 (all combined)
        print(f"Warning: Could not parse problem number from {current_problem_file}, using all combined data")
        return creator.auxiliary_creation_2()


def get_rankswap_data_as_auxiliary() -> pd.DataFrame:
    """Quick function to create auxiliary dataset using only RANKSWAP files."""
    creator = AuxiliaryDatasetCreator()
    return creator.auxiliary_creation_5()