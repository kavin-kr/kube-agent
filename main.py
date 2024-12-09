import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    filename="agent.log",
    filemode="w",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


openai = OpenAI()

config.load_kube_config()


def kubernetes_api_call(api_class: str, method: str, params: dict):
    logger.info(f"Kubernetes API call: {api_class}.{method}({params})")
    try:
        api_instance = getattr(client, api_class)()
        api_method = getattr(api_instance, method)
        result = api_method(**params).to_dict()
        logger.info(f"Kubernetes API result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during Kubernetes API call: {e}")
        return {"error": str(e)}

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
        logger.info(f"Received query: {query}")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that translates natural language queries into Kubernetes API calls.",
            },
            {"role": "user", "content": query},
        ]

        while True:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=kubernetes_api_call_schema,
                function_call="auto",
            )

            if response.choices[0].finish_reason == "stop":
                final_answer = response.choices[0].message.content
                logger.info(f"Final Answer from GPT: {final_answer}")
                response_model = QueryResponse(query=query, answer=final_answer)
                return jsonify(response_model.model_dump())

            elif response.choices[0].finish_reason == "function_call":
                function_call = response.choices[0].message.function_call
                function_name = function_call.name
                arguments = eval(function_call.arguments)

                logger.info(
                    f"Function call requested: {function_name} with arguments {arguments}"
                )

                if function_name == "kubernetes_api_call":
                    raw_result = kubernetes_api_call(
                        api_class=arguments["api_class"],
                        method=arguments["method"],
                        params=arguments.get("params", {}),
                    )

                    logger.info(f"Raw Kubernetes API result: {raw_result}")

                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": str(raw_result),
                        }
                    )
                else:
                    raise ValueError(f"Unexpected function call in GPT response - {function_name}")

            else:
                raise ValueError("Unexpected finish_reason in GPT response.")

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
