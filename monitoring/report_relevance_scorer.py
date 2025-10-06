from judgeval.scorers.example_scorer import ExampleScorer
from judgeval.data import Example
from openai import OpenAI

class Report(Example):
    query: str
    report: str

class ReportRelevanceScorer(ExampleScorer):
    name: str = "Report Relevance Scorer"
    server_hosted: bool = True # Enable server hosting

    async def a_score_example(self, example: Report):
        client = OpenAI()
        # Use LLM to evaluate if research report is relevant to the query
        evaluation_prompt = f"""
        Evaluate if this research report is relevant to the query.
        
        Query: {example.query}
        Report: {example.report}
        
        Is the report relevant and does it answer the query? Answer only "YES" or "NO".
        """
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        evaluation = completion.choices[0].message.content.strip().upper()

        if evaluation == "YES":
            self.reason = "LLM evaluation: Report is relevant to the query"
            return 1.0
        else:
            self.reason = "LLM evaluation: Report is not relevant to the query"
            return 0.0