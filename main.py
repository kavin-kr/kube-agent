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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    filename="agent.log",
    filemode="w",
)

# Configure Kubernetes client
config.load_kube_config()


def call_kubernetes_api(api_class: str, method: str, params: dict):
    logging.info(f"Kubernetes API call: {api_class}.{method}({params})")
    try:
        api_instance = getattr(kubernetes.client, api_class)()
        api_method = getattr(api_instance, method)
        result = api_method(**params)
        result = json.dumps(result, default=str)
        logging.info(f"Kubernetes API result: {result}")
        return result

    except Exception as e:
        logging.error(f"Kubernetes API error: {e}")
        return {"error": str(e)}


# Define the schema for OpenAI function calling
call_kubernetes_api_schema = {
    "type": "function",
    "function": {
        "name": "call_kubernetes_api",
        "description": "Call a Kubernetes API dynamically using the Kubernetes Python Client library and return its result.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "api_class": {
                    "type": "string",
                    "description": "The Kubernetes Python client API class to use for the API call. This must be a valid class from the Kubernetes Python client library (e.g., CoreV1Api, AppsV1Api).",
                },
                "params": {
                    "type": "string",
                    "description": 'The parameters required for the specified `method` of the `api_class`. These parameters must match the arguments expected by the method in the Kubernetes Python client library. If no parameters are needed for the method, provide an empty object `{}`. Use default namepace if no namespace is explicitly specified for namespaced APIs. (e.g., {"name": "example-pod", "namespace": "default"}).',
                },
                "method": {
                    "type": "string",
                    "description": "The specific method to call on the specified `api_class`. This must be a valid method of the provided Kubernetes API class. (e.g., list_namespaced_pod, read_namespaced_pod).",
                },
            },
            "required": ["api_class", "method", "params"],
            "additionalProperties": False,
        },
    },
}


# Fine-tuned system prompt for GPT
SYSTEM_PROMPT = """
You are a Kubernetes assistant that interacts with a Kubernetes cluster to answer questions about its deployed applications.

Your responsibilities:
1. Parse natural language queries into efficient Kubernetes API calls.
2. Minimize the number of API calls by batching requests or reusing data where possible.
3. Determine which Kubernetes API calls are required to answer the query.
4. Execute those API calls and process the results.
5. Ensure no more than 5 Kubernetes API calls are made per query.
6. Provide concise, accurate answers without additional explanations or context.
7. Avoid identifiers in answers (e.g., return 'mongodb' instead of 'mongodb-56c598c8fc').
8. If you cannot answer, respond with 'I don't know.'
9. If there are list of items, return a comma separated list (e.g., 'item1, item2, item3').
10. If there are multiple answers, return the most relevant one.

Example Queries and Responses:
Q: 'Which pod is spawned by my-deployment?' A: 'my-pod'
Q: 'What is the status of the pod named example-pod?' A: 'Running'
Q: 'How many nodes are there in the cluster?' A: '2'
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
        logging.info(f"Received user query: {query}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[call_kubernetes_api_schema],
                tool_choice="auto",
            ).choices[0]

            logging.info(f"OpenAI response: {response}")

            if response.finish_reason == "stop":
                final_answer = response.message.content
                logging.info(f"Final Answer from GPT: {final_answer}")
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
                    params = json.loads(function_arguments["params"])

                    kube_result = call_kubernetes_api(api_class, method, params)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(kube_result, default=str),
                        }
                    )

            else:
                raise ValueError("Unexpected finish_reason in GPT response.")

        # If max attempts exceeded, return an error message
        error_message = f"Query could not be resolved within {max_attempts} attempts."
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
