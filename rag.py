import os
import numpy as np
import faiss
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from tqdm import tqdm 
import re
import torch
import gc
import csv

# Load preprocessed data and embeddings
preprocessed_file = '/content/drive/MyDrive/Data Ablations ANLP/scraped_data/preprocessed_data_5.txt'
questions_file = '/content/drive/MyDrive/Data Ablations ANLP/data/test/test_questions.csv'
reference_answers_file = '/content/drive/MyDrive/Data Ablations ANLP/data/test/reference_answers.txt'
faiss_index_path = '/content/drive/MyDrive/Data Ablations ANLP/embeddings/baai_preprocessed_data_5_faiss_index.idx'
embeddings_path = '/content/drive/MyDrive/Data Ablations ANLP/embeddings/baai_preprocessed_data_5_embeddings.npy'

DEVICE = "cuda" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the preprocessed document texts
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
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': DEVICE})


# Step 4: Create FAISS vector store
vectorstore = FAISS(embedding_model, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

#model_name = "google/flan-t5-large"  
# model_name = "bigscience/bloom-7b1"  
model_name = "google/flan-t5-base"  


tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map = "auto")

#model.to(DEVICE)
# qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
#qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)

def load_questions_from_csv(file_path):
    questions = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            questions.append(row[0]) 
    return questions

# normalize text for evaluation
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  
    text = re.sub(r'[^a-z0-9]', ' ', text) 
    text = ' '.join(text.split()) 
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
    prompt = f"Context: {context}\n Based on the above prompt, give an answer for the following question. Keep your answer concise.\nQuestion: {query}\nAnswer:"

    # Generate the answer using the LLM
    answer = qa_pipeline(prompt)[0]['generated_text']

    # Log the prompt and answer for debugging
    print(f"Prompt to LLM:\n{prompt}\nGenerated Answer: {answer}\n")

    return answer

def evaluate_rag_system(batch_size=10):
    # Load questions and reference answers
    with open(questions_file, 'r', encoding='utf-8') as q_file, \
         open(reference_answers_file, 'r', encoding='utf-8') as a_file:
        
        questions = q_file.readlines()[:100]
        reference_answers = a_file.readlines()[:100]
    questions = load_questions_from_csv(questions_file)
    # total_queries = len(questions)
    # Initialize evaluation tracking
    total_em = 0
    total_f1 = 0
    total_queries = len(questions)
    output_file_path='/content/drive/MyDrive/Data Ablations ANLP/data/test/generated_answers.txt'
    # Open the file to write generated answers
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Process in batches to reduce memory usage
      for i in range(0, total_queries, batch_size):
          batch_questions = questions[i:i + batch_size]
          batch_reference_answers = reference_answers[i:i + batch_size]

          # Iterate over the batch and evaluate the generated answers
          for idx, (question, reference_answer) in enumerate(batch_questions, batch_reference_answers):
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
              output_file.write(f"{generated_answer.strip()}\n")

              # Print individual results for tracking (optional)
              print(f"Batch {i//batch_size + 1} - Question {i + idx + 1}: {question}")
              print(f"Generated Answer: {generated_answer}")
              print(f"Reference Answer: {reference_answer}")
              print(f"Exact Match: {em}, F1 Score: {f1}\n")

          if (i // batch_size) % 10 == 0:
              print(f"Processed {i + len(batch_questions)} questions.")

          if torch.cuda.is_available():
              torch.cuda.empty_cache()
          elif torch.backends.mps.is_available():
              torch.mps.empty_cache()

          del batch_questions
          gc.collect()  

    print("GG ENDDDDDDDDDDDDD")


if __name__ == "__main__":
    evaluate_rag_system()