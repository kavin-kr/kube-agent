import logging
from flask import Flask, request, jsonify
import kubernetes
from pydantic import BaseModel, Field, ValidationError
from kubernetes import client, config
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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
        result = api_method(**params).to_dict()
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
        "description": "Execute a Kubernetes API call using Kubernetes Python Clien library with the specified api_class, method, and parameters and return the result.", # todo
        "parameters": {
            "type": "object",
            "properties": {
                "api_class": {
                    "type": "string",
                    "description": "The Kubernetes Python client class to use (e.g., CoreV1Api, AppsV1Api).",
                },
                "method": {
                    "type": "string",
                    "description": "The method to call on the specified api_class (e.g., list_namespaced_pod, read_namespaced_pod).",
                },
                "params": {
                    "type": "object",
                    "description": "The function parameters that needs to be passed to this api_class's method. If no parameters are needed, provide an empty object ({}). (e.g., {'namespace': 'default'}).",
                },
            },
            "required": ["api_class", "method", "params"],
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
        # Step 1: Extract user query
        query = QueryRequest.model_validate(request.json).query
        logging.info(f"Received user query: {query}")

        # Step 2: Define messages with refined system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        max_attempts = 2
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Step 3: Send query to GPT with function calling enabled
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[call_kubernetes_api_schema],
                tool_choice="auto",
            )

            logging.info(f"OpenAI response: {response}")

            # Step 4: Handle GPT response
            if response.choices[0].finish_reason == "stop":
                # Final answer from GPT
                final_answer = response.choices[0].message.content
                logging.info(f"Final Answer from GPT: {final_answer}")
                response_model = QueryResponse(query=query, answer=final_answer)
                return jsonify(response_model.model_dump())

            elif response.choices[0].finish_reason == "tool_calls":
                # Extract function call details
                function = response.choices[0].message.tool_calls[0].function
                function_name = function.name
                arguments = eval(function.arguments)

                logging.info(
                    f"Function call requested: {function_name} with arguments {arguments}"
                )

                if function_name == "call_kubernetes_api":
                    # Validate arguments
                    api_class = arguments.get("api_class")
                    method = arguments.get("method")
                    params = arguments.get("params", {})

                    if not api_class or not method:
                        raise ValueError(
                            "Missing required keys 'api_class' or 'method'."
                        )

                    # Execute Kubernetes API call
                    raw_result = call_kubernetes_api(api_class, method, params)
                    logging.info(f"Raw Kubernetes API result: {raw_result}")

                    # Append raw result for further processing by GPT
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": str(raw_result),
                        }
                    )

            else:
                raise ValueError("Unexpected finish_reason in GPT response.")

        # If max attempts exceeded, return an error message
        error_message = f"Query could not be resolved within {max_attempts} attempts."
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
