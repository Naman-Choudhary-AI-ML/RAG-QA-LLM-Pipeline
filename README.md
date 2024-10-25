# RAG-QA-LLM-Pipeline

The scrapped data and preprocessed data are not present in this github due to storage issues. They can be created using the scrapper.py and preprocess.py code. Following which, the rag model pipeline is implemented in the following steps.

Steps to run code:
1. Install libraries from the requirements.txt file
2. Run the scraper.py code to scrape the data
3. Run the preprocess.py code to preprocess the data
4. Run the embedding.py to create embeddings, which would produce the embeddings.npy and the faiss_index.idx file.
5. To implement the rag pipeline, run the rag.py code, which involves the paths to the data embeddings and question and answers text data (would require GPU to run the model).
6. The answers would be output in terminal, indicating average F1 and exact match score after all Q&A pairs are covered.


Contributions:
1. Scraping and processing of data: Amulya
2. Dataset creation (different chunks): Amulya
3. Questions-answers pairs (Q&A) creation: Amulya
4. Q&A pair annotation 1: Amulya
5. Q&A pair annotation 2: Naman
6. Dataset embeddings and rag model pipeline creation: Naman
7. Dataset abalations and model abalations with evaluation metrics: Naman
8. Test data checking: Naman
