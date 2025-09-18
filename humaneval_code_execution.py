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
from openai import AsyncOpenAI
from typing import Dict, Any
import asyncio

# Initialize clients
judgment = JudgmentClient()
client = AsyncOpenAI()


async def generate_code(problem: Dict[str, Any]) -> str:
    """Generate code using LLM for a given HumanEval problem."""
    prompt = problem["prompt"]
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert Python programmer. Write ONLY the Python function code that solves the given problem. Do not include any markdown formatting, explanations, or code blocks. Return only the raw Python code."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512
    )
    
    generated_code = response.choices[0].message.content
    
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


async def main():
    print("🚀 Starting HumanEval Code Execution Evaluation")
    
    # Step 1: Load dataset
    print("\n📊 Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval")
    print(f"   Found {len(dataset['test'])} problems")
    
    # Step 2: Generate code for first 5 problems (for demo)
    print("\n🤖 Generating code...")
    problems = list(dataset["test"].select(range(164)))
    
    # Generate all code in parallel
    generated_codes = await asyncio.gather(*[
        generate_code(problem) 
        for problem in problems
    ])
    
    # Create examples
    examples = []
    for i, (problem, generated_code) in enumerate(zip(problems, generated_codes)):
        print(f"   Problem {i+1}/5: {problem['task_id']}")
        
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
    dataset = Dataset.create(
        name="humaneval-dataset", 
        project_name="humaneval-project", 
        examples=examples,
        overwrite=True
    )

    dataset = Dataset.get(
        name="humaneval-dataset",
        project_name="humaneval-project"
    )

    # Step 4: Run evaluation
    print("\n⚡ Running evaluation...")
    judgment.run_evaluation(
        examples=dataset.examples,
        scorers=[HumanEvalCodeExecutionScorer()],
        project_name="humaneval-project"
    )
    
    print("\n✅ Evaluation complete! Check your judgeval dashboard for results.")


if __name__ == "__main__":
    asyncio.run(main())




   



