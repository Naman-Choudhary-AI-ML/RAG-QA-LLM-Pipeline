from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from tqdm import tqdm
import gc  # Import garbage collector

# Directory to store embeddings and FAISS index
output_dir = 'embeddings/'
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the model from Hugging Face (pretrained embedding model)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can change the model if needed

# List of preprocessed data files
preprocessed_files = [
    'scraped_data/preprocessed_data_10.txt',
    'scraped_data/preprocessed_data_5.txt',
    'scraped_data/preprocessed_data_2.txt',
    'scraped_data/preprocessed_data_1g.txt'
]

# Function to embed the preprocessed data and store it in FAISS
def embed_and_store(preprocessed_file, batch_size=100):  # Added batch_size parameter
    try:
        # Read the preprocessed data
        with open(preprocessed_file, 'r', encoding='utf-8') as file:
            text_data = file.readlines()

        # Initialize list to store all the embeddings
        all_embeddings = []

        # Generate embeddings in batches
        print(f"Generating embeddings for {preprocessed_file} in batches of {batch_size}...")

        # Loop through the text data in batches
        for i in tqdm(range(0, len(text_data), batch_size), desc="Embedding texts", unit="batch"):
            batch_texts = text_data[i:i + batch_size]  # Get the current batch
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)  # Embed the batch
            all_embeddings.append(batch_embeddings)  # Append the batch embeddings to the list

            # Free memory by clearing the batch variables
            del batch_embeddings, batch_texts
            gc.collect()  # Explicitly run garbage collector

        # Convert all batches to a single numpy array after processing all batches
        all_embeddings = np.vstack(all_embeddings)

        # Save embeddings to a file in the embeddings/ folder
        embeddings_file = os.path.join(output_dir, f'{os.path.basename(preprocessed_file).split(".")[0]}_embeddings.npy')
        np.save(embeddings_file, all_embeddings)
        print(f"Embeddings saved to {embeddings_file}")

        # Store embeddings in FAISS for retrieval
        dimension = all_embeddings.shape[1]  # Dimension of the embeddings
        faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance metric
        faiss_index.add(all_embeddings)

        # Save the FAISS index in the embeddings/ folder
        faiss_index_file = os.path.join(output_dir, f'{os.path.basename(preprocessed_file).split(".")[0]}_faiss_index.idx')
        faiss.write_index(faiss_index, faiss_index_file)
        print(f"FAISS index saved to {faiss_index_file}")

        # Free memory after processing the entire file
        del all_embeddings, faiss_index, text_data  # Delete large variables
        gc.collect()  # Run garbage collector to free memory

    except Exception as e:
        print(f"Error during embedding process for {preprocessed_file}: {e}")

if __name__ == "__main__":
    for preprocessed_file in preprocessed_files:
        embed_and_store(preprocessed_file, batch_size=100)  # You can adjust the batch size based on memory
