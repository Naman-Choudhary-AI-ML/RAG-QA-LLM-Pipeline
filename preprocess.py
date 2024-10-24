import os
import re
import nltk

# Download necessary data for NLTK's sentence tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Directory containing scraped text files
scraped_data_dir = 'scraped_data'
combined_file = 'scraped_data/combined_content.txt'
preprocessed_file = 'scraped_data/preprocessed_content.txt'

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
    return text.strip()

# Function to chunk the text into smaller pieces based on sentences
def chunk_text(text, max_chunk_size=2):
    # Split the text into sentences using NLTK's sentence tokenizer
    sentences = sent_tokenize(text)

    # Group sentences into chunks (e.g., group every 'max_chunk_size' sentences into one chunk)
    chunks = [' '.join(sentences[i:i + max_chunk_size]) for i in range(0, len(sentences), max_chunk_size)]

    return chunks

# Function to preprocess and chunk the combined content file
def preprocess_and_chunk():
    try:
        # Read the combined content
        with open(combined_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Clean the text
        cleaned_content = clean_text(content)

        # Chunk the text (group sentences into chunks of 5 sentences per chunk by default)
        chunks = chunk_text(cleaned_content, max_chunk_size=5)

        # Save the preprocessed and chunked content
        with open(preprocessed_file, 'w', encoding='utf-8') as file:
            for chunk in chunks:
                file.write(chunk + '\n')
        print(f"Preprocessed and chunked data saved to {preprocessed_file}")

    except Exception as e:
        print(f"Failed to preprocess and chunk data: {e}")

if __name__ == "__main__":
    preprocess_and_chunk()
