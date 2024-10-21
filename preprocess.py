import os
import re

# Directory containing scraped text files
scraped_data_dir = 'scraped_data'
combined_file = 'scraped_data/combined_content.txt'
preprocessed_file = 'scraped_data/preprocessed_content.txt'

# Function to clean and preprocess text
def clean_text(text):
    # Convert to lowercase (optional based on your needs)
    text = text.lower()

    # Remove unnecessary whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    
    # Strip leading/trailing spaces
    return text.strip()

# Function to preprocess combined content file
def preprocess_combined_file():
    try:
        # Read the combined content
        with open(combined_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Clean the text
        cleaned_content = clean_text(content)

        # Save the preprocessed content to a new file
        with open(preprocessed_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        print(f"Preprocessed data saved to {preprocessed_file}")

    except Exception as e:
        print(f"Failed to preprocess data: {e}")

if __name__ == "__main__":
    preprocess_combined_file()
