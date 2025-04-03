from judgeval.scorers import JudgevalScorer
from judgeval import Example, JudgmentClient

# Example input
example = Example(
    target_info={
        "name": "Andy",
        "company": "Datadog",
        "role": "Senior Software Engineer",
        "experience": ["Python", "AWS", "Kubernetes"],
        "achievements": ["Led team of 5 engineers", "Reduced latency by 50%"]
    },
    actual_output="""
    Hey Andy!
    
    I noticed your impressive work at Datadog as a Senior Software Engineer. 
    Your experience with Python and AWS is exactly what we're looking for.
    I'd love to connect and discuss an opportunity that aligns with your expertise.
    
    Best,
    Sarah
    """
)

class ColdEmailInfoScorer(JudgevalScorer):
    def __init__(
        self,
        threshold=0.8,  # Require 80% of info points to be used
        score_type="Cold Email Information Usage",
        include_reason=True,
        async_mode=True
    ):
        super().__init__(score_type=score_type, threshold=threshold)
        self.include_reason = include_reason
        self.async_mode = async_mode

    def score_example(self, example):
        try:
            # Get target info and generated email from example
            target_info = example.target_info
            email = example.actual_output.lower()  # Convert to lowercase for case-insensitive matching
            
            # Define what information points to look for
            info_points = {
                "name": target_info.get("name", ""),
                "company": target_info.get("company", ""),
                "role": target_info.get("role", ""),
                "experience": target_info.get("experience", []),
                "achievements": target_info.get("achievements", [])
            }
            
            # Count how many information points are used
            used_points = []
            for key, value in info_points.items():
                if isinstance(value, list):
                    # For lists (experience, achievements), check if any item is mentioned
                    if any(item.lower() in email for item in value):
                        used_points.append(key)
                else:
                    # For strings, check if the value is mentioned
                    if value.lower() in email:
                        used_points.append(key)
            
            # Calculate score as ratio of used points to total points
            self.score = len(used_points) / len(info_points)
            
            # Generate reason for the score
            if self.include_reason:
                missing = set(info_points.keys()) - set(used_points)
                if not missing:
                    self.reason = "All key information points were utilized in the email."
                else:
                    self.reason = f"Missing information points: {', '.join(missing)}"
            
            self.success = self.score >= self.threshold
            
        except Exception as e:
            self.error = str(e)
            self.success = False

    def _success_check(self):
        if self.error is not None:
            return False
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Cold Email Information Usage Scorer"

# Run the scorer
if __name__ == "__main__":
    # Initialize the scorer
    cold_email_scorer = ColdEmailInfoScorer()
    
    # Score the example
    cold_email_scorer.score_example(example)
    
    # Print results
    print(f"Score: {cold_email_scorer.score}")
    print(f"Reason: {cold_email_scorer.reason}")
    print(f"Success: {cold_email_scorer.success}")
    
    # Optional: Run with Judgment platform
    # client = JudgmentClient()
    # results = client.run_evaluation(
    #     examples=[example],
    #     scorers=[cold_email_scorer],
    #     model="gpt-4"
    # ) 
