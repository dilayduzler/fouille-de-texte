import pandas as pd
import re
import emoji
import os
from nltk.corpus import stopwords
import nltk
import argparse

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Clean the text by  removing emojis, punctuation (except hyphens, question and excalamation marks), stopwords, and normalizing whitespace.
    """
    if pd.isna(text):
        return ""
        
    # Remove emojis
    text = emoji.replace_emoji(text, '')
    
    # Remove the "♥" character
    text = text.replace('♥', '')
    
    text = re.sub(r'(\w)([!?])', r'\1 \2', text)

    text = re.sub(r'([!?]){2,}', lambda m: ' '.join(m.group(0)), text)

    # Remove punctuation except hyphens, spaces, exclamation marks, and question marks
    text = re.sub(r"[^\w\s!?-]", " ", text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def process_dataset(input_file, output_dir=None):
    """
    Process the dataset: read, find duplicates, clean reviews, and save results.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file, sep=';')
        
        # Clean the reviews and replace the original column
        df['Review'] = df['Review'].apply(lambda x: clean_text(x))
        
        # If no output directory is specified, use the current directory
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"\nNo output directory specified. Using current directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the cleaned dataset
        output_file = os.path.join(output_dir, 'cleaned_dataset.csv')
        df.to_csv(output_file, sep=';', index=False)
        print(f"\nCleaning completed. Results saved to '{output_file}'")
        
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean and process review dataset')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output-dir', '-o', help='Directory to save the cleaned dataset (optional)')
    
    args = parser.parse_args()
    
    success = process_dataset(args.input_file, args.output_dir)
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed.") 