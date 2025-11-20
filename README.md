# judgment-cookbook

This repo contains cookbooks demonstrating evaluations of AI Agents using the `judgeval` package implemented by [Judgment Labs](https://judgmentlabs.ai/).

## Prerequisites

Before running these examples, make sure you have:

1. Installed the latest version of the Judgeval package:

   ```bash
   pip install judgeval
   ```
2. Set up your Judgeval API key and organization ID as environment variables:

   ```bash
   export JUDGMENT_API_KEY="your_api_key"
   export JUDGMENT_ORG_ID="your_org_id"
   ```

To get your API key and Organization ID, make an account on the [Judgment Labs platform](https://app.judgmentlabs.ai/login).

## 📚 Cookbooks

| Try Out | Notebook | Description |
|:---------|:-----|:------------|
| RL | [Wikipedia Racer](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/rl/WikiRacingAgent_RL.ipynb) | Train agents with reinforcement learning |
| Online Monitoring | [Research Agent](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/monitoring/Research_Agent_Online_Monitoring.ipynb) | Monitor agent behavior in production |
| Custom Scorers | [HumanEval](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/custom_scorers/HumanEval_Custom_Scorer.ipynb) | Build custom evaluators for your agents |
| Offline Testing | [Get Started For Free] | Compare how different prompts, models, or agent configs affect performance across ANY metric |
| Prompt Comparisons | [Customer Support Agent](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/JudgmentCustomScorer.ipynb) | Test and score customer support prompts to see which style performs best |

You can find a list of [video tutorials for Judgeval use cases](https://www.youtube.com/@Alexshander-JL).
