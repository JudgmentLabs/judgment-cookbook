"""
The code in this file contains an implementation of a custom scorer checking for code style compliance.

The scorer checks for the following:
- Function spacing
- Docstring presence
- Log format

The score is calculated as the ratio of passed style rules to the total number of rules.
For example, if 2 out of 3 rules pass, the score would be 0.67 (67%).
The scorer is considered successful if the score meets or exceeds the threshold (default: 0.9 or 90%).
"""

from judgeval import JudgmentClient
from judgeval.scorers import JudgevalScorer
from judgeval.data import Example


class CodeStyleScorer(JudgevalScorer):
    def __init__(
        self,
        threshold=0.9,  # Require 90% of style rules to be followed
        score_type="Code Style Compliance",
        include_reason=True,
        async_mode=True
    ):
        super().__init__(score_type=score_type, threshold=threshold)
        self.include_reason = include_reason
        self.async_mode = async_mode
        
        # Define style rules to check
        self.style_rules = {
            "function_spacing": 2,  # Number of blank lines between functions
            "require_docstrings": True,  # Whether functions must have docstrings
            "log_format": "structured"  # Log format type
        }

    def score_example(self, example: Example):
        try:
            code = example.actual_output
            violations = []
            passed_rules = 0
            
            # Check each style rule
            if self._check_function_spacing(code):
                passed_rules += 1
            else:
                violations.append("Function spacing does not match requirements")
            
            if self._check_docstrings(code):
                passed_rules += 1
            else:
                violations.append("Missing required docstrings")
            
            if self._check_log_format(code):
                passed_rules += 1
            else:
                violations.append("Log formatting does not match requirements")
            
            # Calculate score as ratio of passed rules to total rules
            self.score = passed_rules / len(self.style_rules)
            
            # Generate reason for the score
            if self.include_reason:
                if not violations:
                    self.reason = "Code follows all style guidelines."
                else:
                    self.reason = f"Style violations found: {', '.join(violations)}"
            
            self.success = self.score >= self.threshold
            
        except Exception as e:
            self.error = str(e)
            self.success = False

    async def a_score_example(self, example: Example):
        try:
            code = example.actual_output
            violations = []
            passed_rules = 0
            
            # Check each style rule
            if self._check_function_spacing(code):
                passed_rules += 1
            else:
                violations.append("Function spacing does not match requirements")
            
            if self._check_docstrings(code):
                passed_rules += 1
            else:
                violations.append("Missing required docstrings")
            
            if self._check_log_format(code):
                passed_rules += 1
            else:
                violations.append("Log formatting does not match requirements")
            
            # Calculate score as ratio of passed rules to total rules
            self.score = passed_rules / len(self.style_rules)
            
            # Generate reason for the score
            if self.include_reason:
                if not violations:
                    self.reason = "Code follows all style guidelines."
                else:
                    self.reason = f"Style violations found: {', '.join(violations)}"
            
            self.success = self.score >= self.threshold
            
        except Exception as e:
            self.error = str(e)
            self.success = False

    def _check_function_spacing(self, code):
        """Check if there are exactly 2 blank lines between functions."""
        lines = code.split('\n')
        function_lines = [i for i, line in enumerate(lines) if line.strip().startswith('def ')]
        
        # Check spacing between consecutive functions
        for i in range(len(function_lines) - 1):
            spacing = function_lines[i + 1] - function_lines[i] - 1
            if spacing != self.style_rules["function_spacing"]:
                return False
        return True

    def _check_docstrings(self, code):
        """Check if each function has a docstring."""
        if not self.style_rules["require_docstrings"]:
            return True
            
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                # Look for docstring in next few lines
                docstring_found = False
                for j in range(i + 1, min(i + 4, len(lines))):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        docstring_found = True
                        break
                if not docstring_found:
                    return False
        return True

    def _check_log_format(self, code):
        """Check if logs follow the structured format."""
        if self.style_rules["log_format"] == "structured":
            # Check if all log lines use structured format (contain {})
            return all('{' in line and '}' in line 
                      for line in code.split('\n') 
                      if 'log.' in line or 'logger.' in line)
        return True

    def _success_check(self):
        if self.error is not None:
            return False
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Code Style Compliance Scorer"
    

if __name__ == "__main__":
    code_style_scorer = CodeStyleScorer()

    example = Example(  # Missing docstring, incorrect log format, missing spacing between functions
    actual_output="""
    def process_data(data):
        result = []
        for item in data:
            log.info(f"Processing item: {item}")
            result.append(item * 2)
        
        def helper(x):
            return x + 1
        
        return result
    """
)
    
    client = JudgmentClient()
    results = client.run_evaluation(
        examples=[example],
        scorers=[code_style_scorer],
        model="gpt-4o",
        project_name="code_style_scorer",
        eval_run_name="code_style_scorer_test",
    ) 
    print(results)