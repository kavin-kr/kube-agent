import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
from openai import OpenAI
from dotenv import load_dotenv

# load .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s - %(message)s",
    filename="agent.log",
    filemode="a",
)

# configure openai client
openai = OpenAI()

# configure kube client
config.load_kube_config()


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
        logging.info(f"Received query: {query}")

        prompt = f"""
        You are an assistant that translates natural language queries into Kubernetes Python client API calls.
        For the query: "{query}", provide:
        1. The Kubernetes Python client class to use (e.g., CoreV1Api, AppsV1Api).
        2. The method to call on that class.
        3. The parameters required for the method.

        Example output format:
        {{
            "api_class": "CoreV1Api",
            "method": "list_namespaced_pod",
            "params": {{"namespace": "default"}}
        }}
        """
        gpt_response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0,
        )
        
        gpt_instructions = eval(gpt_response.choices[0].message.content)  # Convert string to dict

        # Extract details from GPT response
        api_class_name = gpt_instructions["api_class"]
        method_name = gpt_instructions["method"]
        params = gpt_instructions["params"]

        logging.info(f"GPT Instructions: {gpt_instructions}")

        # Step 3: Dynamically get the Kubernetes API class and execute the method
        api_class = getattr(client, api_class_name)()  # Dynamically get API class (e.g., AppsV1Api)
        method = getattr(api_class, method_name)  # Get method dynamically from Kubernetes client class
        raw_result = method(**params)  # Execute method with provided parameters

        # Step 4: Use GPT for post-processing of the result
        post_process_prompt = f"""
        Based on the original query: "{query}" and the following Kubernetes API response:
        
        Response: {raw_result}

        Provide a clear and concise answer to the query.
        """
        
        post_process_response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": post_process_prompt}
            ],
            max_tokens=200,
            temperature=0,
        )
        
        answer = post_process_response.choices[0].message.content

        # Log the answer
        logging.info(f"Generated answer: {answer}")

        # Create the response model
        response = QueryResponse(query=query, answer=str(answer))

        print(response)

        return jsonify(response.dict())

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
