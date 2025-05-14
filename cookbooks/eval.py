from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import AnswerRelevancyScorer, AnswerCorrectnessScorer

client = JudgmentClient()

example = Example(
    input="What is the capital of France?",
    actual_output="Paris",
    expected_output="Paris",
    retrieval_context=["Paris is the capital of France.", "Paris is a city in France."],
)
example2 = Example(
    input="What is the capital of Kansas?",
    actual_output="Topeka",
    expected_output="Topeka",
    retrieval_context=["Topeka is the capital of Kansas.", "Topeka is a city in Kansas."],
)
example3 = Example(
    input="What is the capital of Texas?",
    actual_output="Austin",
    expected_output="Austin",
    retrieval_context=["Austin is the capital of Texas.", "Austin is a city in Texas."],
)

# results = client.run_evaluation(
#     eval_run_name="hard-demo2",
#     project_name="hard-demo",
#     examples=[example, example2, example3],
#     scorers=[AnswerRelevancyScorer(threshold=0.5), AnswerCorrectnessScorer(threshold=0.5)],
#     model="gpt-4o",
#     override=True,
# )
