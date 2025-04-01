"""
This script is a cookbook of how to create a custom scorer using a ClassifierScorer.

Simply use a natural language prompt and guide the LLM to output a score based on the input by 
choosing from a set of options.
"""

from judgeval import JudgmentClient
from judgeval.scorers import ClassifierScorer
from judgeval.data import Example

brand_mention_scorer = ClassifierScorer(
    "Brand Mention Checker",
    slug="brand-mention-100",  # you can make this your own slug
    threshold=1.0,
    conversation=[{
        "role": "system",
        "content": """You will be given a text that may contain mentions of athletic brands.

** TASK INSTRUCTIONS **
Your task is to determine whether the provided text contains any positive mentions of Nike's competitors. A positive mention is any reference that portrays a competitor's brand, products, or services in a favorable light.

** NIKE'S COMPETITORS TO LOOK FOR **
- Adidas
- Reebok
- Under Armour
- Puma
- New Balance
- Asics
- Converse
- Vans
- Fila
- On
- Skechers
- Lululemon
- Columbia
- The North Face
- Salomon
- Brooks
- Hoka
- Any other athletic apparel, footwear, or equipment brands that compete with Nike

** WHAT CONSTITUTES A POSITIVE MENTION **
- Praising a competitor's product quality, design, or performance
- Recommending a competitor's products or services
- Highlighting innovative features of a competitor's offerings
- Describing positive experiences with a competitor's products
- Comparing a competitor favorably to other brands (including Nike)
- Mentioning a competitor's superior value, comfort, durability, etc.
- Expressing preference for a competitor's brand or products

** FORMATTING YOUR ANSWER **
First, analyze the text carefully and identify any mentions of Nike's competitors. For each mention, determine whether it is positive or neutral/negative.
Then, choose option "NO_POSITIVE_COMPETITOR_MENTIONS" if the text does not contain any positive mentions of Nike's competitors, or "CONTAINS_POSITIVE_COMPETITOR_MENTIONS" if it does contain positive mentions of any competitors.

** EXAMPLES **

Example 1:
Text: "I've been running in Nike Air Zoom Pegasus for years and they provide excellent support for my marathon training."
Analysis: This text mentions only Nike positively and no competitors.
Answer: NO_POSITIVE_COMPETITOR_MENTIONS

Example 2:
Text: "While Nike makes good basketball shoes, I find that Adidas Ultraboost provides better energy return and comfort for my running needs."
Analysis: This text contains a positive mention of Adidas (a Nike competitor) by stating their product provides "better energy return and comfort."
Answer: CONTAINS_POSITIVE_COMPETITOR_MENTIONS

Example 3:
Text: "I compared several brands including Nike, Adidas, and Under Armour, but ultimately Nike's Dri-FIT technology was the best for my workout routine."
Analysis: While competitors are mentioned, Nike is portrayed as superior, so there are no positive mentions of competitors.
Answer: NO_POSITIVE_COMPETITOR_MENTIONS

Example 4:
Text: "The athletic footwear market is dominated by major players like Nike, Adidas, and Puma, each with their own market share."
Analysis: This text mentions competitors but in a neutral, factual way without positive attributes.
Answer: NO_POSITIVE_COMPETITOR_MENTIONS

** YOUR TURN **
Text:
{{actual_output}}
        """
    }],
    options={
        "NO_POSITIVE_COMPETITOR_MENTIONS": 1.0, 
        "CONTAINS_POSITIVE_COMPETITOR_MENTIONS": 0.0
    }
)


if __name__ == "__main__":
    client = JudgmentClient()

    # Examples with positive competitor mentions
    positive_adidas_example = Example(
        input="Review athletic shoes",
        actual_output="I've tried many running shoes, but Adidas Ultraboost offers the best cushioning and responsiveness I've ever experienced. The Continental rubber outsole also provides exceptional grip in wet conditions."
    )
    
    positive_under_armour_example = Example(
        input="Review athletic apparel",
        actual_output="Under Armour's HeatGear compression shirts are revolutionary for intense workouts. They wick sweat better than any other brand and keep you cool even during the hottest summer training sessions."
    )
    
    positive_multiple_brands_example = Example(
        input="Compare athletic brands",
        actual_output="While Nike has good marketing, Adidas offers superior comfort in their running shoes, and New Balance provides the best width options for people with wider feet. Hoka's maximalist cushioning is also unmatched for long-distance runners."
    )
    
    positive_subtle_example = Example(
        input="Discuss workout gear",
        actual_output="For my workout routine, I prefer moisture-wicking fabrics. Lululemon's fabrics feel luxurious against the skin while maintaining excellent breathability during intense training sessions."
    )
    
    # Examples without positive competitor mentions
    nike_positive_example = Example(
        input="Review athletic shoes",
        actual_output="Nike Air Zoom Pegasus are the perfect all-around running shoes. They provide excellent support, cushioning, and durability for daily training runs."
    )
    
    neutral_mention_example = Example(
        input="Discuss athletic market",
        actual_output="The athletic apparel market includes major players like Nike, Adidas, Under Armour, and Puma. Each brand has different price points and target demographics."
    )
    
    negative_competitor_example = Example(
        input="Compare athletic shoes",
        actual_output="I tried Adidas running shoes but found them uncomfortable compared to my Nike React Infinity. The Nike shoes provided better stability and cushioning for my running style."
    )
    
    no_brand_mention_example = Example(
        input="Discuss workout routine",
        actual_output="My morning workout routine includes 30 minutes of cardio followed by strength training. I focus on different muscle groups each day to ensure proper recovery."
    )

    client.run_evaluation(
        examples=[
            positive_adidas_example,
            positive_under_armour_example,
            positive_multiple_brands_example,
            positive_subtle_example,
            nike_positive_example,
            neutral_mention_example,
            negative_competitor_example,
            no_brand_mention_example
        ],
        scorers=[brand_mention_scorer],
        model="gpt-4o-mini",
        project_name="brand_mention_checker",
        eval_run_name="competitor_mention_test",
        override=True
    )
