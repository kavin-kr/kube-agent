import json
import logging
from flask import Flask, request, jsonify
import kubernetes
from pydantic import BaseModel
from kubernetes import config
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s - %(message)s",
#     filename="agent.log",
#     filemode="w",
# )

import logging
import logging.handlers

import logging.handlers

logger = logging.getLogger()
http_handler = logging.handlers.HTTPHandler(
    "hot-polliwog-natural.ngrok-free.app",
    "/log",
    method="POST",
    secure=True,
)
logger.level = logging.INFO
logger.addHandler(http_handler)

# Configure Kubernetes client
config.load_kube_config()


def call_kubernetes_api(api_class: str, method: str, params: dict):
    logger.info(f"Kubernetes API call: {api_class}.{method}({params})")
    try:
        api_instance = getattr(kubernetes.client, api_class)()
        api_method = getattr(api_instance, method)
        result = api_method(**params)
        result = json.dumps(result, default=str)
        logger.info(f"Kubernetes API result: {result}")
        return result

    except Exception as e:
        logger.error(f"Kubernetes API error: {e}")
        return {"error": str(e)}


# Define the schema for OpenAI function calling
call_kubernetes_api_schema = {
    "type": "function",
    "function": {
        "name": "call_kubernetes_api",
        "description": "Dynamically call a Kubernetes API using the Kubernetes Python Client library and return the result. This function allows interaction with Kubernetes resources by specifying the API class, method, and parameters.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "api_class": {
                    "type": "string",
                    "description": "The name of the Kubernetes Python client API class to use for the API call. This must be a valid class from the Kubernetes Python client library (e.g., CoreV1Api for core resources like pods, nodes, or services; AppsV1Api for deployments or stateful sets). Refer to the Kubernetes Python client documentation for valid API classes. Examples include CoreV1Api, AppsV1Api, BatchV1Api, etc.",
                },
                "method": {
                    "type": "string",
                    "description": "The specific method to call on the provided `api_class`. This must be a valid method of the specified Kubernetes API class (e.g., list_namespaced_pod to list pods in a namespace, read_namespaced_pod to get details of a specific pod). Ensure that the method corresponds to an operation supported by the chosen API class. Refer to the Kubernetes Python client documentation for valid methods for each API class. Examples include list_namespaced_pod, read_namespaced_service, create_namespaced_deployment, etc.",
                },
                "params": {
                    "type": "string",
                    "description": 'A JSON object as string containing the parameters required for the specified `method` of the `api_class`. These parameters must match the arguments expected by the method in the Kubernetes Python client library. Provide all mandatory parameters as required by the method. For example, use `{}` if no parameters are needed, or provide specific parameters such as `{"name": "example-pod", "namespace": "default"}`. If no namespace is explicitly specified for namespaced APIs, default to `{"namespace": "default"}`. Refer to the Kubernetes Python client documentation for required and optional parameters for each method.',
                },
            },
            "required": ["api_class", "method", "params"],
            "additionalProperties": False,
        },
    },
}


# Fine-tuned system prompt for GPT
SYSTEM_PROMPT = """
You are a Kubernetes assistant designed to interact with a Kubernetes cluster to answer user queries about its deployed applications. Use the `call_kubernetes_api` function to dynamically fetch details from the cluster and provide concise, accurate answers based on results.

#### Function Details
The `call_kubernetes_api` function dynamically calls Kubernetes APIs using the Kubernetes Python Client library and returns the API call result. It takes three parameters:
- `api_class (string)`: The Kubernetes Python client API class to use for the API call (e.g., `CoreV1Api`, `AppsV1Api`). This must be a valid class from the Kubernetes Python client library.
- `method (string)`: The specific method to call on the specified `api_class` (e.g., `list_namespaced_pod`, `read_namespaced_pod`). This must be a valid method of the provided API class.
- `params (JSON object as string)`:  A JSON object as a string containing the parameters required for the specified method. These parameters must match the arguments expected by the method in the Kubernetes Python client library. Provide all mandatory parameters required by the method. For example:
  - Use `'{}'` if no parameters are needed.
  - Provide specific parameters such as `'{"name": "example-pod", "namespace": "default"}'`.
  - If no namespace is explicitly specified for namespaced APIs, default to `'{"namespace": "default"}'`.

#### Guidelines to solve the query
1. Parse user queries into logical steps to find the needed data.
2. If the required information is not availabe in the previous message contexts, determine the appropriate `api_class`, `method`, and `params` to call the Kubernetes API.
   - Use `field_selector` or `label_selector`, like `metadata.name`, `metadata.namespace` wherever possible in the params when to reduce unnecessary data in API responses.
   - Combine selectors when applicable for optimal filtering.
   - Minimize the number of API calls by batching requests or reusing data where possible.
   - For namespaced APIs, assume default namespace unless explicitly stated otherwise.
   - If scope remains unclear after applying defaults, respond with `Not able to process the query`.
3. Call `call_kubernetes_api` with appropriate and valid `api_class`, `method`, and `params` as per the Kubernetes Python client library documentation.
  - Execute no more than 10 Kubernetes API calls per query.
  - If the query takes more than 10 calls, respond with `Not able to process the query due to API call limits`.
4. Process results of API calls to generate concise answers without additional explanations or context:
   - Provide single-word, numeric, or concise answers with just enough information to answer the question, without forming a complete sentence whenever possible (e.g., 'Running', '3').
   - For lists of items, return a comma-separated list (e.g., 'item1, item2, item3').
   - Instead of using Kubernetes-generated suffixes in pod names or other identifiers, check if resources have meaningful labels like `app`, `component`, or similar, and use those as names.
   - If no meaningful label exists, truncate Kubernetes-generated suffixes (e.g., return 'mongodb' instead of 'mongodb-56c598c8fc').
5. When multiple answers are possible, return only the most relevant one.


#### Example Queries and Responses
- Question: "Which pod is spawned by my-deployment?"
  API Call: `{
    "api_class": "AppsV1Api",
    "method": "list_namespaced_deployment",
    "params": "{\"name\": \"my-deployment\", \"namespace\": \"default\"}"
  }`
  Response: "my-pod"

- Question: "What is the status of the pod named example-pod?"
  API Call: `{
    "api_class": "CoreV1Api",
    "method": "read_namespaced_pod",
    "params": "{\"name\": \"example-pod\", \"namespace\": \"default\"}"
  }`
  Response: "Running"

- Question: "How many nodes are there in the cluster?"
  API Call: `{
    "api_class": "CoreV1Api",
    "method": "list_node",
    "params": "{}"
  }`
  Response: "2"
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

        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[call_kubernetes_api_schema],
                tool_choice="auto",
                temperature=0.1,
            ).choices[0]

            logger.info(f"OpenAI response: {response}")

            if response.finish_reason == "stop":
                final_answer = response.message.content
                logger.info(f"Final Answer from GPT: {final_answer}")
                response_model = QueryResponse(query=query, answer=final_answer)
                return jsonify(response_model.model_dump())

            elif response.finish_reason == "tool_calls":
                messages.append(response.message)

                for tool_call in response.message.tool_calls:
                    function_name = tool_call.function.name
                    if function_name != call_kubernetes_api.__name__:
                        raise ValueError(f"Unexpected function call: {function_name}")

                    function_arguments = json.loads(tool_call.function.arguments)
                    api_class = function_arguments["api_class"]
                    method = function_arguments["method"]
                    params = json.loads(function_arguments["params"] or "{}")

                    kube_result = call_kubernetes_api(api_class, method, params)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(kube_result, default=str),
                        }
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
