import pandas as pd
import os
import argparse

def create_text_files(input_file, output_dir=None):
    """
    Convert CSV reviews to text files organized by category.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str, optional): Directory to save the text files. If not specified,
                                  creates a 'reviews_by_category' directory in the current folder.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file, sep=';')
        
        # If no output directory is specified, create a default one
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Corpus')
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each review
        for index, row in df.iterrows():
            category = str(row['Category']).strip()
            review = str(row['Review']).strip()
            
            # Create category directory if it doesn't exist
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Create the text file
            file_name = f"review_{index + 1}.txt"
            file_path = os.path.join(category_dir, file_name)
            
            # Write the review to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(review)
        
        print(f"\nProcessing completed successfully.")
        print(f"Text files have been created in: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''
        Convert CSV reviews to text files organized by category.
        
        This script will:
        1. Read a CSV file containing reviews and categories
        2. Create a directory structure based on categories
        3. Convert each review to a separate text file
        4. Save the text files in their respective category folders
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='''
        Directory to save the text files (optional).
        If not specified, creates a 'reviews_by_category' directory in the current folder.
        '''
    )
    
    args = parser.parse_args()
    
    success = create_text_files(args.input_file, args.output_dir)
    if not success:
        print("\nProcessing failed.") 