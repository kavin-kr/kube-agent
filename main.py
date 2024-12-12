import json
import logging
from typing import Literal
from flask import Flask, request, jsonify
import kubernetes
from pydantic import BaseModel, Field, field_validator
from kubernetes import config
import openai
from dotenv import load_dotenv
import jq

# Load environment variables
load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s - %(message)s",
#     filename="agent.log",
#     filemode="w",
# )

import logging.handlers

logger = logging.getLogger()
logger.level = logging.INFO

file_handler = logging.FileHandler("agent.log")
logger.addHandler(file_handler)

http_handler = logging.handlers.HTTPHandler(
    "hot-polliwog-natural.ngrok-free.app",
    "/log",
    method="POST",
    secure=True,
)
logger.addHandler(http_handler)

# Configure Kubernetes client
config.load_kube_config()


def call_kubernetes_api(api_class: str, method: str, params: dict, jq_filter: str):
    logger.info(f"Kubernetes API call: {api_class}.{method}({params} with filter {jq_filter})")
    try:
        api_instance = getattr(kubernetes.client, api_class)()
        api_method = getattr(api_instance, method)
        result = api_method(**params, _preload_content=False)
        result = json.loads(result.data)
        logger.info(f"Kubernetes API result: {result}")

        result = jq.compile(jq_filter).input(result).all()
        logger.info(f"Kubernetes API filtered result: {result}")

        return result

    except Exception as e:
        logger.error(f"Kubernetes API error: {e}")
        return json.dumps({"error": str(e)})


class CallKubernetesAPI(BaseModel):
    """
    Dynamically call a Kubernetes API using the Kubernetes Python Client library
    by specifying the API class, method, and parameters and return the result.
    """

    api_class: str = Field(
        ...,
        description="""
        The name of the Kubernetes Python client API class to use for the API call.
        This must be a valid class as documented in the Kubernetes Python client library.
        (e.g., `CoreV1Api`, `AppsV1Api`)
        """,
    )
    method: str = Field(
        ...,
        description="""
        The specific method to call on the provided `api_class`.
        This must be a valid method of the specified Kubernetes API class as documented in the Kubernetes Python client library.
        (e.g., `list_namespaced_pod`, `read_namespaced_service`)
        """,
    )
    params: str | dict = Field(
        ...,
        description="""
        A JSON object as a string containing the parameters required for the specified `method` of the `api_class`.
        These parameters must be valid arguments of the specified `method` of the `api_class` as documented in the Kubernetes Python client library.
        If no parameters are needed, provide an empty JSON object `{}`.
        If no namespace is explicitly specified for namespaced APIs, default to `'{"namespace": "default"}'`.
        (e.g., `{"name": "example-pod", "namespace": "default"}`, `{"label_selector": "component=etcd"}`).
        """,
    )

    jq_filter: str = Field(
        ...,
        description="""
        A jq filter to apply to the API response to extract only the needed relevant information.
        This filter should be a valid jq filter string that can be applied to the API response.
        Use filter expressions whenever possible to extract only the relevant information from the API response.
        For example, to extract the pod names from a list of pods, you can use the filter '.items[].metadata.name'.
        If no filtering is needed, provide `'.'` this as default.
        """,
    )

    @field_validator("params")
    @classmethod
    def validate_and_convert_params(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided for 'params'.")
        elif isinstance(value, dict):
            return value
        else:
            raise TypeError(
                "'params' must be either a dictionary or a valid JSON string."
            )


class FinalAnswer(BaseModel):
    """
    Represents the final answer to a user query, that can be sent back to the user.
    """

    answer: str

class Error(BaseModel):
    """
    Represents a response where the query could not be resolved or error occurred.
    """

    message: str


class OpenAIResponse(BaseModel):
    """
    Represents the response from the OpenAI API, which can include the final answer, a count list, or logs.
    """

    type: Literal["final_answer", "error"]
    data: FinalAnswer | Error


# Fine-tuned system prompt for GPT
SYSTEM_PROMPT = """
You are a helpful Kubernetes assistant designed to interact with a Kubernetes cluster to answer user queries about its deployed applications. Use the `call_kubernetes_api` function to dynamically fetch details from the cluster and provide concise, accurate answers based on results.

#### Function Details
The `call_kubernetes_api` function dynamically calls Kubernetes APIs using the Kubernetes Python Client library and returns the API call result. It takes three parameters:
  1. `api_class (string)`: The name of the Kubernetes Python client API class to use for the API call. This must be a valid class as documented in the Python client library. (e.g., `CoreV1Api`, `AppsV1Api`)
  2. `method (string)`: The specific method to call on the provided `api_class`. This must be a valid method of the specified Kubernetes API class as documented in the Kubernetes Python library. (e.g., `list_namespaced_pod`, `read_namespaced_service`)
  3. `params (JSON object as string)`: A JSON object as a string containing the parameters required for the specified `method` of the `api_class`. These parameters must be valid arguments of the specified `method` of the `api_class` as documented in the Kubernetes Python client library. If no parameters are needed, provide an empty JSON object `{}`. If no namespace is explicitly specified for namespaced APIs, default to `'{"namespace": "default"}'`. (e.g., `{"name": "example-pod", "namespace": "default"}`, `{"label_selector": "component=etcd"}`). Refer to the Kubernetes Python client documentation for required and optional parameters for each method.
  4. `jq_filter (string)`: A jq filter to apply to the API response to extract only the needed relevant information. This filter should be a valid jq filter string that can be applied to the API response. Use filter expressions whenever possible to extract only the relevant information from the API response. For example, to extract the pod names from a list of pods, you can use the filter '.items[].metadata.name'. If no filtering is needed, provide `'.'` this as default.

#### Steps to solve the user query
1. Understand the user query
    - Understand the context of the user query and the information requested by the user about the Kubernetes cluster.
    - Determine the logical steps required to find the needed data.
    - If the query is ambiguous or unclear, use the best judgment to interpret the user's intent.
    - Understand the kubernetes key words and their relatioships to get the required data.
2. Identify the appropriate `api_class`, `method`, and `params` to call the Kubernetes API and `jq_filter` to filter the API response to extract only the relevant information.
    1. Based on the user query and logical steps, determine the `api_class`, `method`, and `params` to get details from the cluster.
    2. Ensure that the `api_class` and `method` are valid classes and methods from the Kubernetes Python client library.
    3. Provide the necessary parameters in the `params` field to fetch the required information.
        - If no parameters are needed, provide an empty JSON object `{}`.
        - For the namespaced APIs, if the namespace is not explicitly specified, default to `'{"namespace": "default"}'` or use the list based api to find the namespace.
        - Use `field_selector` or `label_selector` like `metadata.name`, `metadata.namespace` wherever possible in the params to reduce fetching unnecessary data in API responses.
        - As an expert use the best guess for the params to get the required data, such that it should be inclusive.
        - Combine selectors whenever applicable for optimal filtering.
    4. Provide a valid `jq_filter` to filter the API response to extract only the relevant information. Use filter expressions whenever possible to extract only the relevant information from the API response. For example, to extract the pod names from a list of pods, you can use the filter '.items[].metadata.name'. If no filtering is needed, provide `'.'` this as default. The `jq_filter` should be valid based on the API response structure. Refer kubernetes API documentation to understand the response structure.
    5. Make smart combination of `params` and appropriate `jq_filter` to get only the required data.
3. Fetch the needed details from the cluster using the `call_kubernetes_api` function with the determined `api_class`, `method`, `params` and `jq_filter`.
    - Execute no more than 10 Kubernetes API calls per query.
    - If the query takes more than 10 calls, respond with `Not able to process the query due to API call limits`.
    - Minimize the number of API calls by batching requests or reusing data where possible.
4. Process the API response to generate a concise answers without additional explanations or context and output the return type as `final_answer`. Follow the below guidelines to generate the answer:
    - Provide single-word, numeric, or concise answers with just enough information to answer the question, without forming a complete sentence whenever possible (e.g., 'Running', '3').
    - For lists of items, return a comma-separated list (e.g., 'item1, item2, item3').
    - Instead of using Kubernetes-generated suffixes in pod names or other identifiers, check if resources have meaningful labels like `app`, `component`, or similar, and use those as names.
    - If no meaningful label exists, truncate Kubernetes-generated suffixes (e.g., return 'mongodb' instead of 'mongodb-56c598c8fc').
    - When multiple answers are possible, return only the most relevant one.
5. If the query could not be processed or any error retry with the best guess within the 10 API. If still the error occurs return `Not able to process the query`.

#### Example Queries and Responses
- Question: "Which pod is spawned by my-deployment?"
  call_kubernetes_api function call params: `{
    "api_class": "AppsV1Api",
    "method": "list_namespaced_deployment",
    "params": "{\"name\": \"my-deployment\", \"namespace\": \"default\"}",
    "jq_filter": "[.spec.template.spec.containers[].name]"
  }`
  Response: `{
    "type": "final_answer",
    "data": {
      "answer": "my-pod"
    }
  }`

- Question: "What is the status of the pod named example-pod?"
  call_kubernetes_api function call params: `{
    "api_class": "CoreV1Api",
    "method": "read_namespaced_pod",
    "params": "{\"name\": \"example-pod\", \"namespace\": \"default\"}",
    "jq_filter": "[.status.containerStatuses[].state]"
  }`
  Response: `{
    "type": "final_answer",
    "data": {
      "answer": "running"
    }
  }`

- Question: "How many pods are there in the cluster?"
  (don't need to fetch the exact number of nodes, just return the call_kubernetes_api details)
  Response: `{
    "type": "count_list",
    "data": {
      "count_items_from_kubernetes_apis": [{
        "api_class": "CoreV1Api",
        "method": "list_pod_for_all_namespaces",
        "params": {},
        "jq_filter": ".items | length"
      }]
    }
  }`
"""

# Flask app initialization
app = Flask(__name__)


@app.route("/query", methods=["POST"])
def create_query():
    class QueryRequest(BaseModel):
        query: str

    class QueryResponse(BaseModel):
        query: str
        answer: str

    try:
        query = QueryRequest.model_validate(request.json).query
        logger.info(f"Received user query: {query}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        max_attempts = 11
        for _ in range(max_attempts):
            response = openai.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                tools=[
                    openai.pydantic_function_tool(
                        name=call_kubernetes_api.__name__, model=CallKubernetesAPI
                    )
                ],
                tool_choice="auto",
                temperature=0.1,
                response_format=OpenAIResponse,
            ).choices[0]

            logger.info(f"OpenAI response: {response}")

            if response.finish_reason == "stop":
                openai_response = response.message.parsed
                logger.info(f"OpenAI response: {openai_response}")

                if openai_response.type == "final_answer" and isinstance(
                    openai_response.data, FinalAnswer
                ):
                    final_answer = openai_response.data.answer
                    response_model = QueryResponse(query=query, answer=final_answer)
                    return jsonify(response_model.model_dump())

                elif openai_response.type == "error" and isinstance(
                    openai_response.data, Error
                ):
                    error_message = openai_response.data.message
                    logger.error(f"Error from GPT: {error_message}")
                    return jsonify({"query": query, "result": "error"})

            elif response.finish_reason == "tool_calls":
                messages.append(response.message)

                for tool_call in response.message.tool_calls:
                    args = tool_call.function.parsed_arguments
                    if isinstance(args, CallKubernetesAPI):
                        kube_result = call_kubernetes_api(
                            args.api_class, args.method, args.params, args.jq_filter
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(kube_result, default=str),
                            }
                        )
                    else:
                        raise ValueError(
                            f"Unexpected function call in GPT response: {tool_call}"
                        )

            else:
                raise ValueError("Unexpected finish_reason in GPT response.")

        # If max attempts exceeded, return an error message
        error_message = f"Query could not be resolved within {max_attempts} attempts."
        logger.error(error_message)
        return jsonify({"query": query, "result": "unknown"})

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"query": query, "result": "unknown"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
