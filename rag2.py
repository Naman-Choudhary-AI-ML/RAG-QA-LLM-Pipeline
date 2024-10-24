import os
import numpy as np
import faiss
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm  # For progress bar
import re

# Load preprocessed data and embeddings
preprocessed_file = 'scraped_data/preprocessed_content.txt'
questions_file = 'data/test/questions.txt'
reference_answers_file = 'data/test/reference_answers.txt'
faiss_index_path = 'embeddings/faiss_index.idx'
embeddings_path = 'embeddings/embeddings.npy'

# Load the preprocessed document texts (one document per line in preprocessed_file)
with open(preprocessed_file, 'r', encoding='utf-8') as f:
    document_texts = [line.strip() for line in f.readlines()]

# Step 1: Load FAISS index and document embeddings
index = faiss.read_index(faiss_index_path)
document_embeddings = np.load(embeddings_path)

# Create Document objects for the docstore
documents = [Document(page_content=text) for text in document_texts]

# Step 2: Use an InMemoryDocstore to map FAISS indices to documents
docstore = InMemoryDocstore(dict(enumerate(documents)))

# Create index_to_docstore_id mapping (index to document IDs)
index_to_docstore_id = {i: i for i in range(len(documents))}

# Step 3: Embedder - Load the pre-trained sentence transformer model for query embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create FAISS vector store
vectorstore = FAISS(embedding_model, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Step 5: Set up a Hugging Face model (Flan-T5 used for this example)
model_name = "google/flan-t5-large"  # You can adjust to any other model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Helper function to normalize text for evaluation (removes articles, punctuation, extra spaces, and lowercases)
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  # Remove articles
    text = re.sub(r'[^a-z0-9]', ' ', text)  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Function to calculate Exact Match (EM)
def exact_match_score(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

# Function to calculate F1 score
def f1_score_metric(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Function to retrieve the context and generate an answer
def retrieve_and_generate_answer(query):
    # Perform similarity search using FAISS to get the relevant documents
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)  # Perform the similarity search

    # Combine all retrieved documents into a single context string
    context = "\n".join([doc.page_content for doc in docs])

    # Prepare the prompt for the LLM (without an answer in the prompt)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate the answer using the LLM
    answer = qa_pipeline(prompt)[0]['generated_text']

    # Log the prompt and answer for debugging
    print(f"Prompt to LLM:\n{prompt}\nGenerated Answer: {answer}\n")

    return answer

# Evaluation function to process questions and calculate metrics
def evaluate_rag_system():
    # Load questions and reference answers
    with open(questions_file, 'r', encoding='utf-8') as q_file, \
         open(reference_answers_file, 'r', encoding='utf-8') as a_file:
        
        questions = q_file.readlines()
        reference_answers = a_file.readlines()

    # Initialize evaluation tracking
    total_em = 0
    total_f1 = 0
    total_queries = len(questions)

    # Iterate over all questions and evaluate the generated answers
    for idx, (question, reference_answer) in tqdm(enumerate(zip(questions, reference_answers)), total=total_queries, desc="Evaluating QA Pairs"):
        question = question.strip()
        reference_answer = reference_answer.strip()

        # Step 1: Retrieve context and generate answer
        generated_answer = retrieve_and_generate_answer(question)

        # Step 2: Calculate Exact Match and F1 score
        em = exact_match_score(generated_answer, reference_answer)
        f1 = f1_score_metric(generated_answer, reference_answer)

        # Accumulate the scores
        total_em += em
        total_f1 += f1

        # Print individual results for tracking (optional)
        print(f"Question {idx+1}: {question}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Reference Answer: {reference_answer}")
        print(f"Exact Match: {em}, F1 Score: {f1}\n")

    # Calculate and print the average scores
    avg_em = (total_em / total_queries) * 100  # Percentage
    avg_f1 = total_f1 / total_queries

    print(f"Average Exact Match (EM): {avg_em:.2f}%")
    print(f"Average F1 Score: {avg_f1:.2f}")

if __name__ == "__main__":
    evaluate_rag_system()
