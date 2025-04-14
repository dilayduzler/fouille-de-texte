import pandas as pd
import argparse

def find_duplicates(reviews):
    """
    Find duplicate reviews and return their line numbers.
    """
    duplicates = []
    seen = set()
    for i, review in enumerate(reviews, 1):  
        if review in seen:
            duplicates.append(i)
        seen.add(review)
    return duplicates

def main():
    parser = argparse.ArgumentParser(description="Analyze CSV for category counts and duplicate reviews.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file (with ; delimiter)")
    parser.add_argument("--review_column", type=str, default="Review", help="Name of the column containing review texts")
    parser.add_argument("--category_column", type=str, default="Category", help="Name of the column containing categories")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path, delimiter=";")

    print("Category Counts:")
    print(df[args.category_column].value_counts())

    duplicate_lines = find_duplicates(df[args.review_column])
    if duplicate_lines:
        print("\nDuplicate review line numbers:")
        print(duplicate_lines)
    else:
        print("\nNo duplicate reviews found.")

if __name__ == "__main__":
    main()