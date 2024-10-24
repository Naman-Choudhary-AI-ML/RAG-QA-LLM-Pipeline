from transformers import pipeline
import os
from tqdm import tqdm  # Progress bar

# Use Flan-T5-large model for generating questions and answers
qa_generator = pipeline('text2text-generation', model='google/flan-t5-large')

# Paths for preprocessed data and output directory
preprocessed_file = os.path.join('scraped_data', 'preprocessed_content.txt')
output_dir = os.path.join('data', 'test')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to generate a question from a text chunk
def generate_question(chunk):
    prompt = (
        f"Context: {chunk}\n"
        "Generate a factual question based on the text. Format the output like this:\n"
        "Question: (The question goes here)"
    )
    question_result = qa_generator(prompt, max_length=200, num_return_sequences=1)
    return question_result[0]['generated_text']

# Function to generate the answer based on the question and context, with emphasis on conciseness
def generate_answer(question, chunk):
    prompt = (
        f"Context: {chunk}\n"
        f"Based on the above context, generate a short and concise factual answer to the following question:\n"
        f"{question}\nAnswer: (The answer goes here, keep it concise and to the point)"
    )
    answer_result = qa_generator(prompt, max_length=100, num_return_sequences=1)
    return answer_result[0]['generated_text']


# Function to split question and answer based on the explicit format
def split_question_answer(qa_pair):
    if "Question:" in qa_pair and "Answer:" in qa_pair:
        question, answer = qa_pair.split("Answer:", 1)
        question = question.replace("Question:", "").strip()
        answer = answer.strip()
        return question, answer
    else:
        return None, None

# Process chunks, generate questions first, then answers
def process_chunks_and_generate_qa():
    seen_questions = set()  # Track unique questions

    # Read the preprocessed file
    with open(preprocessed_file, 'r', encoding='utf-8') as file:
        chunks = file.readlines()

    # Limit to the first 10 chunks for now (for testing)
    chunks = chunks[:10]

    # Open output files
    with open(os.path.join(output_dir, 'questions.txt'), 'w', encoding='utf-8') as q_file, \
         open(os.path.join(output_dir, 'reference_answers.txt'), 'w', encoding='utf-8') as a_file, \
         open(os.path.join(output_dir, 'raw_llm_output.txt'), 'w', encoding='utf-8') as raw_file:

        # Progress bar for tracking
        for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
            # Step 1: Generate the question
            question_result = generate_question(chunk)
            raw_file.write(f"Raw Question LLM Output: {question_result}\n")
            print(f"Raw Question LLM Output: {question_result}")

            # Step 2: Generate the answer based on the generated question
            question = question_result.replace("Question:", "").strip()
            if question and question not in seen_questions:
                seen_questions.add(question)
                answer_result = generate_answer(question, chunk)
                raw_file.write(f"Raw Answer LLM Output: {answer_result}\n")
                print(f"Raw Answer LLM Output: {answer_result}")

                # Split the question and answer from the LLM output
                question, answer = split_question_answer(f"Question: {question}\nAnswer: {answer_result}")
                
                if question and answer:
                    q_file.write(f"{question}\n")
                    a_file.write(f"{answer}\n")

                    # Flush the files after each write
                    q_file.flush()
                    a_file.flush()

    print(f"Questions saved to {os.path.join(output_dir, 'questions.txt')}")
    print(f"Answers saved to {os.path.join(output_dir, 'reference_answers.txt')}")
    print(f"Raw LLM output saved to {os.path.join(output_dir, 'raw_llm_output.txt')}")

# Run the QA generation process
if __name__ == "__main__":
    process_chunks_and_generate_qa()
