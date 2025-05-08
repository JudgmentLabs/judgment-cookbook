import os
import PyPDF2
import re # Added for regex operations
from dotenv import load_dotenv
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

# Load environment variables from .env file
load_dotenv()

def _is_single_letter_alpha(token: str) -> bool:
    """Checks if a token is a single alphabetical character."""
    return len(token) == 1 and token.isalpha()

def _attempt_despace_text(text: str) -> str:
    """Heuristically attempts to fix 'S p a c e d o u t t e x t'."""
    tokens = text.split(' ')
    if not tokens:
        return text

    processed_tokens = []
    current_word_chars = []

    for token in tokens:
        if _is_single_letter_alpha(token):
            current_word_chars.append(token)
        else:
            if current_word_chars:  # We were accumulating a spaced-out word
                processed_tokens.append("".join(current_word_chars))
                current_word_chars = []
            processed_tokens.append(token) # Append the current multi-letter or non-alpha token
    
    if current_word_chars:  # Append any remaining accumulated chars at the end
        processed_tokens.append("".join(current_word_chars))
        
    return " ".join(filter(None, processed_tokens)) # filter(None,...) removes empty strings if any resulted

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts text content from a given PDF file with enhanced cleaning."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if not reader.pages:
                print(f"Warning: No pages found in {pdf_path}.")
                return None
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path} (it might be image-based, empty, or only contain non-textual elements).")
            return None

        # --- Text Cleaning Step 1: Unicode REPLACEMENT CHARACTER U+FFFD () ---
        text = text.replace('\uFFFD', '') 

        # --- Text Cleaning Step 2: Normalize general whitespace ---
        text = re.sub(r'\s+', ' ', text).strip()
        
        # --- Text Cleaning Step 3: Attempt to fix "S p a c e d o u t t e x t" ---
        text = _attempt_despace_text(text)

        if not text.strip():
            print(f"Warning: All text content was removed during cleaning for {pdf_path}. Original might have only been artifacts.")
            return None
            
        print(f"Successfully extracted and cleaned text from: {pdf_path}")
        return text
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {pdf_path}: {e}")
        return None

def main():
    # --- Configuration ---
    # Define the paths to your PDF files
    # These files should be in the same directory as this script,
    # or you should provide absolute paths.
    output_doc_path = "parrot_demo/summary_record.pdf"  # E.g., a summary or answer document
    source_doc_path = "parrot_demo/source_record.pdf"  # E.g., the context or source material

    # Initialize JudgmentClient
    # Ensure JUDGMENT_API_KEY (and JUDGMENT_ORG_ID if needed by your setup) is in your .env file
    try:
        client = JudgmentClient()
    except Exception as e:
        print(f"Failed to initialize JudgmentClient: {e}")
        print("Please ensure JUDGMENT_API_KEY is correctly set in your .env file or environment variables.")
        return

    # --- PDF Text Extraction ---
    print("Attempting to import PDF files...")
    output_text = extract_text_from_pdf(output_doc_path)
    source_text = extract_text_from_pdf(source_doc_path)

    print(f"Output text: {output_text[:200]}")
    print(f"Source text: {source_text[:200]}")

    if output_text is None or source_text is None:
        print("Failed to extract text from one or both PDF documents. Exiting.")
        return

    # --- Setup Evaluation Example ---
    # The 'input' frames the task for the scorers.
    # 'actual_output' is the content we are evaluating.
    # 'retrieval_context' is the ground truth or source material.
    example_input = "Is this an accurate summary of the source document?"
    
    example = Example(
        input=example_input,
        actual_output=output_text,
        retrieval_context=[source_text]  # retrieval_context should be a list of strings
    )

    # --- Initialize Scorers ---
    # Thresholds are from 0.0 to 1.0. Adjust as needed.
    faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
    answer_relevancy_scorer = AnswerRelevancyScorer(threshold=0.7)

    print("\nRunning evaluation...")
    # --- Run Evaluation ---
    try:
        results = client.run_evaluation(
            project_name="parrot_demo_pdf_comparison", # You can change this project name
            examples=[example],
            eval_run_name="parrot_demo_pdf_comparison_run_11",
            scorers=[faithfulness_scorer, answer_relevancy_scorer],
            model="gpt-4.1",  # Ensure this model is appropriate for your needs and subscription
        )
        
        print("\n--- Evaluation Results ---")
        # The results object structure might vary; typically, it's a list of result objects.
        # For a single example, you might inspect results[0].
        # Refer to Judgeval documentation for detailed result structure.
        if results and hasattr(results, 'get') and results.get('examples_results'):
             for eval_result in results.get('examples_results', []):
                print(f"Input: {eval_result.example.input}")
                # print(f"Actual Output: {eval_result.example.actual_output[:200]}...") # Truncate for brevity
                print(f"Scores: {eval_result.scores}")
                print("----")
        else:
            print(f"Raw results: {results}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()
