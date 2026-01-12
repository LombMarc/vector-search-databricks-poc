from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

def _get_endpoint_task_type(endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    w = WorkspaceClient()
    ep = w.serving_endpoints.get(endpoint_name)
    return ep.task

def is_endpoint_supported(endpoint_name: str) -> bool:
    """Check if the endpoint has a supported task type."""
    task_type = _get_endpoint_task_type(endpoint_name)
    supported_task_types = ["agent/v1/chat", "agent/v2/chat", "llm/v1/chat", "agent/v1/responses"]
    return task_type in supported_task_types

def _validate_endpoint_task_type(endpoint_name: str) -> None:
    """Validate that the endpoint has a supported task type."""
    if not is_endpoint_supported(endpoint_name):
        raise Exception(
            f"Detected unsupported endpoint type for this basic chatbot template. "
            f"This chatbot template only supports chat completions-compatible endpoints. "
            f"For a richer chatbot template with support for all conversational endpoints on Databricks, "
            f"see https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app"
        )
def _query_endpoint(endpoint_name: str, messages: list[dict[str, str]], max_tokens) -> list[dict[str, str]]:
    """Calls a model serving endpoint."""
    _validate_endpoint_task_type(endpoint_name)
    print("pre-call input formatting")
    input_dict = {'input': messages}
    print(messages)
    res = get_deploy_client('databricks').predict(
        endpoint=endpoint_name,
        inputs=input_dict,
    )
    res = dict(res.output[-1])
    print(res)
    
    #{'type': 'message', 'id': 'chatcmpl_9a17ffe0-a810-42d7-a43e-a34a9f02f428', 'content': [{'text': 'Delta Lake is an open‑source storage layer for data lakes that adds **ACID transactions** and **schema enforcement** to the otherwise raw, unstructured data stored in a lake. This ensures reliable, consistent data operations and helps maintain data quality within large, distributed data environments.', 'type': 'output_text'}], 'role': 'assistant'}
    return {'content' : res['content'][0]['text']}


def query_endpoint(endpoint_name, messages, max_tokens):
    return _query_endpoint(endpoint_name, messages, max_tokens)
