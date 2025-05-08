import os
import fitz  # PyMuPDF
import re # Added for regex operations
import docx # Added for DOCX processing
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

def _clean_extracted_text(text: str) -> str | None:
    """Applies common cleaning steps to extracted text."""
    if not text or not text.strip():
        return None
    # 1. Replace Unicode REPLACEMENT CHARACTER U+FFFD ()
    text = text.replace('\uFFFD', '')
    # 2. Normalize general whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Attempt to fix "S p a c e d o u t t e x t"
    text = _attempt_despace_text(text)
    if not text.strip():
        return None
    return text

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts and cleans text content from a PDF file using PyMuPDF."""
    raw_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text += page.get_text("text") # Extract plain text
        doc.close()
        
        cleaned_text = _clean_extracted_text(raw_text)
        if cleaned_text is None:
            print(f"Warning: No usable text extracted/cleaned from PDF: {pdf_path} using PyMuPDF")
            return None
        print(f"Successfully extracted and cleaned text from PDF: {pdf_path} using PyMuPDF")
        return cleaned_text
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading PDF {pdf_path} with PyMuPDF: {e}")
        return None

def extract_text_from_docx(docx_path: str) -> str | None:
    """Extracts and cleans text content from a DOCX file."""
    raw_text = ""
    try:
        document = docx.Document(docx_path)
        for para in document.paragraphs:
            raw_text += para.text + "\n" # Add newline to separate paragraphs
        
        cleaned_text = _clean_extracted_text(raw_text)
        if cleaned_text is None:
            print(f"Warning: No usable text extracted/cleaned from DOCX: {docx_path}")
            return None
        print(f"Successfully extracted and cleaned text from DOCX: {docx_path}")
        return cleaned_text
    except FileNotFoundError:
        print(f"Error: The file {docx_path} was not found.")
        return None
    except Exception as e:
        # Catch specific docx errors if known, e.g., docx.opc.exceptions.PackageNotFoundError
        print(f"An error occurred while reading DOCX {docx_path}: {e}")
        return None

def main():
    # --- Configuration ---
    output_doc_path = "parrot_demo/summary_discharge.docx"  # Changed to .docx
    source_doc_path = "parrot_demo/source_discharge.pdf"    # Stays as .pdf

    try:
        client = JudgmentClient()
        print("Successfully initialized JudgmentClient!") # Moved success message here
    except Exception as e:
        print(f"Failed to initialize JudgmentClient: {e}")
        print("Please ensure JUDGMENT_API_KEY is correctly set in your .env file or environment variables.")
        return

    # --- Text Extraction ---
    print("\nAttempting to import documents...")
    output_text = None
    if output_doc_path.lower().endswith('.docx'):
        output_text = extract_text_from_docx(output_doc_path)
    elif output_doc_path.lower().endswith('.pdf'): # Keep PDF option for flexibility
        output_text = extract_text_from_pdf(output_doc_path)
    else:
        print(f"Error: Unsupported file type for output document: {output_doc_path}")
        return

    source_text = extract_text_from_pdf(source_doc_path) # Source is still PDF

    # Print extracted text (or part of it) for verification
    print(f"\nOutput text (first 200 chars): {output_text[:200] if output_text else 'None'}")
    print(f"Source text (first 200 chars): {source_text[:200] if source_text else 'None'}")

    if output_text is None or source_text is None:
        print("\nFailed to extract text from one or both documents. Exiting.")
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
            project_name="parrot_demo_pdf_test", # You can change this project name
            examples=[example],
            eval_run_name="parrot_demo_pdf_comparison_run_discharge",
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