from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import faiss
import numpy as np

# Step 1: Embedder - Load the pre-trained sentence transformer model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Load the precomputed FAISS index and document embeddings
faiss_index_path = "embeddings/faiss_index.idx"
embeddings_path = "embeddings/embeddings.npy"
index = faiss.read_index(faiss_index_path)
document_embeddings = np.load(embeddings_path)

# Create a FAISS vector store using the loaded index and document embeddings
vectorstore = FAISS(embedding_model.embed_query, index, document_embeddings)

# Step 3: Document Reader - Set up the Llama2 model as the LLM
llama_model_name = "meta-llama/Llama-2-7b-chat-hf"  # You can use Llama-2-7b or any other version
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
llama_model = LlamaForCausalLM.from_pretrained(llama_model_name)

# Create a Hugging Face pipeline for Llama2
qa_pipeline = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_length=512)

# Use Langchain's HuggingFacePipeline for Llama2
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Step 4: Define a custom chain where we explicitly retrieve relevant chunks
def retrieve_and_generate_answer(query):
    # Step 4.1: Retrieve the most relevant chunks from the FAISS index
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)  # Perform the similarity search

    # Combine all retrieved document chunks into a single context string
    context = "\n".join([doc.page_content for doc in docs])

    # Step 4.2: Combine the query and the context, then pass to LLM for answer generation
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate the answer using the LLM
    answer = qa_pipeline(input_text)[0]['generated_text']

    return answer

# Step 5: Process a query and run the RAG pipeline
def rag_pipeline(query):
    # Run the RAG pipeline to get the answer
    answer = retrieve_and_generate_answer(query)
    return answer

if __name__ == "__main__":
    # Example query to test the system
    query = "What is the name of the annual pickle festival held in Pittsburgh?"
    result = rag_pipeline(query)
    print(f"Answer: {result}")