import pandas as pd

def csv_to_pkl(csv_filepath, pkl_filepath):
    """
    Converts a CSV file to a pickle file.

    Args:
        csv_filepath (str): Path to the input CSV file.
        pkl_filepath (str): Path to the output pickle file.
    """
    try:
        df = pd.read_csv(csv_filepath)
        df.to_pickle(pkl_filepath)
        print(f"CSV file '{csv_filepath}' successfully converted to pickle file '{pkl_filepath}'")
    except FileNotFoundError:
         print(f"Error: CSV file '{csv_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# Example usage:
csv_file = 'random.csv'
pkl_file = 'my_data.pkl'
csv_to_pkl(csv_file, pkl_file)