from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load preprocessed content
preprocessed_file = 'scraped_data/preprocessed_content.txt'

# Load the model from Hugging Face (pretrained embedding model)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can change the model if needed

# Function to embed the preprocessed data and store it in FAISS
def embed_and_store():
    try:
        # Read the preprocessed data
        with open(preprocessed_file, 'r', encoding='utf-8') as file:
            text_data = file.readlines()

        # Generate embeddings using the pretrained model
        print("Generating embeddings...")
        embeddings = model.encode(text_data, show_progress_bar=True)

        # Save embeddings to a file (for backup or further use)
        np.save('embeddings.npy', embeddings)
        print(f"Embeddings saved to embeddings.npy")

        # Store the embeddings in FAISS for retrieval
        dimension = embeddings.shape[1]  # Dimension of the embeddings
        faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance metric
        faiss_index.add(embeddings)

        # Save the FAISS index
        faiss.write_index(faiss_index, 'faiss_index.idx')
        print("FAISS index saved to faiss_index.idx")

    except Exception as e:
        print(f"Error during embedding process: {e}")

if __name__ == "__main__":
    embed_and_store()
