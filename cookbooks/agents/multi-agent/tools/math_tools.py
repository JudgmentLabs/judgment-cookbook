from simpleeval import simple_eval
from .common import judgment, client

# Example pairs for formalize
FORMALIZE_EXAMPLES = """
Example 1:
Word problem: If you have 3 apples and get 2 more, how many apples do you have?
Equation: 3 + 2

Example 2:
Word problem: Sarah has 5 pencils. She gives 2 to Tom. How many pencils does Sarah have left?
Equation: 5 - 2

Example 3:
Word problem: A box contains 4 red balls and 6 blue balls. How many balls are there in total?
Equation: 4 + 6

Example 4:
Word problem: You invest $1000 at 5% interest. How much will you have in 3 years?
Equation: 1000 * (1 + 0.05) ** 3

Example 5:
Word problem: You invest $1000 at 5% interest every year for 5 years. How much will you have in 5 years?
Equation: 1000 * (1 + 0.05) ** 5 + 1000 * (1 + 0.05) ** 4 + 1000 * (1 + 0.05) ** 3 + 1000 * (1 + 0.05) ** 2 + 1000 * (1 + 0.05) ** 1 + 1000 * (1 + 0.05) ** 0

"""

# Example pairs for format
FORMAT_EXAMPLES = """
Example 1:
Equation: 3 + 2
Code: 3 + 2

Example 2:
Equation: 5 - 2
Code: 5 - 2

Example 3:
Equation: x = 4, y=3, (x + 6) * y
Code: (4 + 6) * 3

Example 4:
Equation: 1000 * (1 + 0.05) ** 5 + 1000 * (1 + 0.05) ** 4 + ... + 1000 * (1 + 0.05) ** 1 + 1000 * (1 + 0.05) ** 0
Code: 1000 * (1 + 0.05) ** 5 + 1000 * (1 + 0.05) ** 4 + 1000 * (1 + 0.05) ** 3 + 1000 * (1 + 0.05) ** 2 + 1000 * (1 + 0.05) ** 1 + 1000 * (1 + 0.05) ** 0

Example 5:
Equation: 1000 * (1 + 0.05) ** 5 + 1000 * (1 + 0.05) ** 4 + ... + 1000 * (1 + 0.05) ** 0
Code: 1000 * (1 + 0.05) ** 5 + 1000 * (1 + 0.05) ** 4 + 1000 * (1 + 0.05) ** 3 + 1000 * (1 + 0.05) ** 2 + 1000 * (1 + 0.05) ** 1 + 1000 * (1 + 0.05) ** 0

"""

class MathToolsMixin:
    """Mixin providing mathematical problem-solving tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'operations'):
            self.operations = 0
    
    @judgment.observe(span_type="tool")
    def formalize(self, problem: str) -> str:
        """
        Tool to convert a math word problem into a math equation.
        
        Args:
            problem: The word problem text to convert
            
        Returns:
            The math equation as a string
        """
        prompt = f"""
    You are a math expert. Read the following word problem and write a single math equation (in plain text, no explanation) that solves it.

    {FORMALIZE_EXAMPLES}Word problem: {problem}
    Equation:
    """
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": "You are a helpful math expert."},
                    {"role": "user", "content": prompt}]
        )
        equation = response.choices[0].message.content.strip()
        return equation

    @judgment.observe(span_type="tool")
    def format(self, equation: str) -> str:
        """
        Tool to convert a math equation into simpleeval-runnable code.
        
        Args:
            equation: The math equation to convert to code
            
        Returns:
            The executable code as a string
        """
        prompt = f"""
    Convert the following math equation into Python code that can be run by the simpleeval library. Only output the code, no explanation.

    {FORMAT_EXAMPLES}Equation: {equation}
    Code:
    """
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": "You are a helpful tool calling agent that can call tools to solve math problems. Make sure to break down the problem and call multiple tools to solve it."},
                    {"role": "user", "content": prompt}]
        )
        code = response.choices[0].message.content.strip()
        return code

    @judgment.observe(span_type="tool")
    def calculate(self, code: str) -> str:
        """
        Tool to evaluate simpleeval code and return the answer.
        
        Args:
            code: The code expression to evaluate
            
        Returns:
            The computed answer as a string
        """
        try:
            answer = simple_eval(code)
            return str(answer)
        except Exception as e:
            return f"Error: {e}" 