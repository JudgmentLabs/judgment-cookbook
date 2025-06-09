from openai import OpenAI
from openai import (
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
    APIStatusError,
)
import requests
from judgeval.common.tracer import Tracer, wrap
from requests.models import Response, Request

# Set up our own client and tracer for testing
test_client = wrap(OpenAI())
test_judgment = Tracer(project_name="test-error-handling", deep_tracing=False)

class MockRequest:
    def __init__(self, method="POST", url="https://api.openai.com/v1/chat/completions"):
        self.method = method
        self.url = url
        self.headers = {}
        self.body = None

class MockResponse:
    def __init__(self, status_code=429):
        self.status_code = status_code
        self.headers = {}
        self.text = ""
        self.request = MockRequest()

class TestErrorHandling:
    """
    Test suite for error handling in the judgment.observe infrastructure.
    Mixin providing tools that deliberately raise different types of errors.
    """
    
    def __init__(self):
        self.function_map = {
            "raise_rate_limit": self.raise_rate_limit,
            "raise_timeout": self.raise_timeout,
            "raise_connection": self.raise_connection,
            "raise_auth": self.raise_auth,
            "raise_http_400": lambda: self.raise_http(400),
            "raise_http_401": lambda: self.raise_http(401),
            "raise_http_403": lambda: self.raise_http(403),
            "raise_http_404": lambda: self.raise_http(404),
            "raise_http_429": lambda: self.raise_http(429),
            "raise_http_500": lambda: self.raise_http(500),
            "raise_http_503": lambda: self.raise_http(503),
            "raise_network": self.raise_network,
        }

    @test_judgment.observe(span_type="function")
    def test_all_errors(self):
        """Test all error-raising functions."""
        for tool_name in self.function_map:
            try:
                self.function_map[tool_name]()
                print(f"Unexpected: {tool_name} did not raise an error")
            except Exception as e:
                print(f"Successfully caught error from {tool_name}: {type(e).__name__}: {str(e)}")

    @test_judgment.observe(span_type="tool")
    def raise_rate_limit(self): 
        response = MockResponse(status_code=429)
        raise RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
        )

    @test_judgment.observe(span_type="tool")
    def raise_timeout(self):
        raise requests.exceptions.Timeout("Request timed out")

    @test_judgment.observe(span_type="tool")
    def raise_connection(self): 
        raise APIConnectionError(
            message="Connection failed",
            response=MockResponse(),
            body={"error": {"message": "Connection failed", "type": "connection_error"}}
        )

    @test_judgment.observe(span_type="tool")
    def raise_auth(self): 
        raise AuthenticationError(
            message="Invalid API key",
            response=MockResponse(status_code=401),
            body={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )

    @test_judgment.observe(span_type="tool")
    def raise_http(self, status_code: int): 
        raise requests.exceptions.HTTPError(f"HTTP {status_code} error")

    @test_judgment.observe(span_type="tool")
    def raise_network(self): 
        raise requests.exceptions.ConnectionError("Network connection failed")


if __name__ == '__main__':
    TestErrorHandling().test_all_errors()