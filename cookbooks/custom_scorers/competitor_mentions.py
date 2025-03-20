"""
** WARNING **: This script relies on creating a ClassifierScorer on the Judgment platform.
If you haven't created one yet, you can do so by following the instructions in this YouTube guide: https://www.youtube.com/watch?v=LNEX-RkeQQI

This script implements a custom scorer to evaluate customer support responses.

It checks if a support response mentions competitors (like Adidas, Reebok, etc.) in a positive way.
The scorer penalizes responses that promote competitor products, helping maintain brand focus in
customer interactions. This would be useful to score customer support responses for Nike, for example.
"""


from judgeval import JudgmentClient
from judgeval.data import Example


if __name__ == "__main__":
    client = JudgmentClient()

    positive_example = Example(
        input="What are the best shoes for running priced under $130?",
        actual_output="You'd want to check out the newest Nike Vaporfly, it's only $120 and built for performance."
    )

    negative_example = Example(
        input="What are the best shoes for running priced under $130?",
        actual_output="The Nike Vaporfly is a great shoe built for performance. Other great options include the Adidas Ultraboost and the Reebok Nano X which are affordable and speedy."
    )

    competitor_mentions_scorer = client.fetch_classifier_scorer("<YOUR_SLUG_HERE>")  # replace with slug, see video guide above

    client.run_evaluation(
        examples=[positive_example, negative_example],
        scorers=[competitor_mentions_scorer],
        model="gpt-4o-mini",
        project_name="competitor_mentions",
        eval_run_name="competitor_brand_demo",
    )



