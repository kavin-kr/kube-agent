import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
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

# Define a function for Kubernetes API calls
def kubernetes_api_call(api_class: str, method: str, params: dict):
    logging.info(f"Kubernetes API call: {api_class}.{method}({params})")
    try:
        api_instance = getattr(client, api_class)()
        api_method = getattr(api_instance, method)
        result = api_method(**params).to_dict()
        logging.info(f"Kubernetes API result: {result}")
        return result

    except Exception as e:
        logging.error(f"Kubernetes API error: {e}")
        return {"error": str(e)}

# Define the schema for OpenAI function calling
kubernetes_api_call_schema = [
    {
        "name": "kubernetes_api_call",
        "description": "Execute a Kubernetes API call.",
        "parameters": {
            "type": "object",
            "properties": {
                "api_class": {
                    "type": "string",
                    "description": "The Kubernetes Python client class to use (e.g., CoreV1Api, AppsV1Api).",
                },
                "method": {
                    "type": "string",
                    "description": "The method to call on the specified class.",
                },
                "params": {
                    "type": "object",
                    "description": "The parameters required for the method.",
                },
            },
            "required": ["api_class", "method", "params"],
        },
    }
]

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
        logging.info(f"Received query: {query}")

        # Step 2: Define messages with refined system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        while True:
            # Step 3: Send query to GPT with function calling enabled
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=kubernetes_api_call_schema,
                function_call="auto",
            )

            # Step 4: Handle GPT response
            if response.choices[0].finish_reason == "stop":
                # Final answer from GPT
                final_answer = response.choices[0].message.content
                logging.info(f"Final Answer from GPT: {final_answer}")
                response_model = QueryResponse(query=query, answer=final_answer)
                return jsonify(response_model.model_dump())

            elif response.choices[0].finish_reason == "function_call":
                # Extract function call details
                function_call = response.choices[0].message.function_call
                function_name = function_call.name
                arguments = eval(function_call.arguments)

                logging.info(f"Function call requested: {function_name} with arguments {arguments}")

                if function_name == "kubernetes_api_call":
                    # Validate arguments
                    api_class = arguments.get("api_class")
                    method = arguments.get("method")
                    params = arguments.get("params", {})

                    if not api_class or not method:
                        raise ValueError("Missing required keys 'api_class' or 'method'.")

                    # Execute Kubernetes API call
                    raw_result = kubernetes_api_call(api_class, method, params)
                    logging.info(f"Raw Kubernetes API result: {raw_result}")

                    # Append raw result for further processing by GPT
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": str(raw_result),
                    })

            else:
                raise ValueError("Unexpected finish_reason in GPT response.")

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)