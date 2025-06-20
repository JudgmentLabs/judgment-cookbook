# Music Recommendation Agent (LangGraph)

A basic LangGraph agent that generates music recommendations based on user taste.

Uses the TavilySearch API to perform web searching over artist music.

## Running the Music Agent
1. Obtain a `TAVILY_API_KEY` and remaining environmental variables as in `.env.example`
2. Run `music_agent.py`.


If you want to run the agent without the evaluation, uncomment the section in the `if __name__ == "__main__":` section.

## Integration Points
| Integration Point | Where/How Used |
|--------------------------|--------------------------------------------------------------------------------|
| JudgmentClient | For running and logging evaluations/assertions |
| Tracer | For tracing and monitoring the workflow |
| JudgevalCallbackHandler| For integrating tracing with LangGraph execution |
| async_evaluate | For online evaluation of model outputs during workflow |
| Scorers | For specifying evaluation criteria (relevancy, tool order, etc.) |
| Example/Assertions | For defining and running test cases with expected tool usage |

