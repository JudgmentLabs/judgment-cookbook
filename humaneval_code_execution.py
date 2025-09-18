"""
HumanEval Code Execution Evaluation Cookbook

This cookbook demonstrates how to evaluate code generation using the HumanEval dataset
with functional correctness (code execution + test verification) using judgeval.

The evaluation works by:
1. Loading HumanEval problems from Hugging Face
2. Generating code using GPT for each problem
3. Running the generated code against test cases in a sandbox
4. Scoring based on whether all tests pass (1.0) or fail (0.0)
"""

from judgeval import JudgmentClient
from judgeval.dataset import Dataset
from judgeval.scorers.example_scorer import ExampleScorer
from judgeval.data import Example
from datasets import load_dataset
from human_eval.execution import check_correctness
from openai import OpenAI
from typing import Dict, Any

# Initialize clients
judgment = JudgmentClient()
client = OpenAI()


def generate_code_with_gpt(problem: Dict[str, Any], model_client) -> str:
    """
    Generate code using GPT for a given HumanEval problem.
    
    Args:
        problem: HumanEval problem dict with 'prompt' key
        model_client: OpenAI client
    
    Returns:
        Generated code string
    """
    prompt = problem["prompt"]
    
    response = model_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert Python programmer. Write a function that solves the given problem."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=512
    )
    
    generated_code = response.choices[0].message.content
    
    # Extract just the function code (remove markdown if present)
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0].strip()
    
    return generated_code



class HumanEvalCodeExecutionScorer(ExampleScorer):
    """
    A scorer for evaluating code generation using functional correctness.
    
    This scorer uses the human_eval.execution.check_correctness function
    to run generated code against test cases and determine if it passes.
    
    Attributes:
        name (str): The name of the scorer
    """
    name: str = "HumanEval Code Execution Scorer"

    async def a_score_example(self, example: Example) -> None:
        """
        Score an example by running the generated code against test cases.
        
        This method uses check_correctness to execute the generated code
        in a sandboxed environment and check if it passes all test cases.
        
        Args:
            example (HumanEvalExample): The example containing the problem and generated code
            
        Returns:
            float: The score (1.0 if all tests pass, 0.0 otherwise)
        """
        # Create problem dict in the format expected by check_correctness
        problem = {
            "task_id": example.task_id,
            "prompt": example.prompt,
            "test": example.test,
            "entry_point": example.entry_point
        }
        
        # Use check_correctness to evaluate the generated code
        result = check_correctness(
            problem=problem,
            completion=example.generated_code,
            timeout=3.0
        )
        
        # Set score based on whether tests passed
        if result["passed"]:
            self.score = 1.0
            self.reason = "All test cases passed"
        else:
            self.score = 0.0
            self.reason = f"Test failed: {result['result']}"
        
        return self.score


dataset = load_dataset("openai/openai_humaneval")

examples = []
for i, problem in enumerate(dataset["test"].select(range(5))):
    print(f"   Problem {i+1}/5: {problem['task_id']}")
    
    # Generate code
    generated_code = generate_code_with_gpt(problem, client)
    
    # Create example
    example = Example(
        task_id=problem["task_id"],
        prompt=problem["prompt"],
        canonical_solution=problem["canonical_solution"],
        test=problem["test"],
        entry_point=problem["entry_point"],
        generated_code=generated_code
    )
    examples.append(example)

# Step 3: Create judgeval project and dataset
print("\n📝 Creating judgeval project...")
Dataset.create(
    name="humaneval-dataset", 
    project_name="humaneval-project", 
    examples=examples, 
    overwrite=True
)

# Step 4: Run evaluation
print("\n⚡ Running evaluation...")
judgment.run_evaluation(
    examples=examples,
    scorers=[HumanEvalCodeExecutionScorer()],
    project_name="humaneval-project",
    
)

print("\n✅ Evaluation complete! Check your judgeval dashboard for results.")




   



