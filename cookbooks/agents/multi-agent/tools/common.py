from openai import OpenAI
from judgeval.common.tracer import Tracer, wrap

client = OpenAI()

judgment = Tracer(project_name="multi-agent-system", deep_tracing=False)