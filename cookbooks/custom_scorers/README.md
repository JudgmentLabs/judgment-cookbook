# Judgeval Custom Scorer Examples

This directory contains examples of custom scorers for the Judgeval platform.

## Prerequisites

Before running these examples, make sure you have:

1. Installed the Judgeval package:
   ```bash
   pip install judgeval
   ```

2. Set up your Judgeval API key and organization ID as environment variables:
   ```bash
   export JUDGMENT_API_KEY="your_api_key"
   export JUDGMENT_ORG_ID="your_org_id"
   ```

## Examples

### 1. Cold Email Generator Scorer (`cold_email_scorer.py`)

This scorer evaluates if a cold email generator properly utilizes all relevant information about the target recipient.

```bash
python cold_email_scorer.py
```

Expected output:
```
Score: 0.6
Reason: Missing information points: achievements
Success: False
```

### 2. Code Review Style Scorer (`code_style_scorer.py`)

This scorer evaluates if code suggestions align with team style preferences.

```bash
python code_style_scorer.py
```

Expected output:
```
Score: 0.33
Reason: Style violations found: Missing required docstrings, Function spacing does not match requirements, Log formatting does not match requirements
Success: False
```

## Customizing the Examples

You can customize these examples by:

1. Modifying the example input data in each file
2. Adjusting the scoring thresholds
3. Adding or removing scoring criteria
4. Changing the scoring logic

## Running with Judgeval Platform

To run these scorers with the Judgeval platform:

1. Uncomment the platform integration code at the bottom of each file:
   ```python
   client = JudgmentClient()
   results = client.run_evaluation(
       examples=[example],
       scorers=[scorer],
       model="gpt-4"
   )
   ```

2. Make sure you have your API key and organization ID set up correctly.

## Troubleshooting

If you encounter any issues:

1. Make sure your Judgeval API key and organization ID are correctly set
2. Check that you have the latest version of the Judgeval package
3. Verify that your Python environment has all required dependencies

## File Structure

```
examples/
├── README.md
├── cold_email_scorer.py
└── code_style_scorer.py
``` 
