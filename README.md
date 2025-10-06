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
| Custom Scorers | [HumanEval](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/custom_scorers/HumanEval_Custom_Scorer.ipynb) | Build custom evaluators for your agents |
| Online Monitoring | [Research Agent](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/monitoring/Report_Agent_Online_Monitoring.ipynb) | Monitor agent behavior in production |
| RL | [Wikipedia Racer](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/rl/WikiRacingAgent_RL.ipynb) | Train agents with reinforcement learning |
