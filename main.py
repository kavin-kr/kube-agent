from dotenv import load_dotenv
from flask import Flask, request, jsonify
from kubernetes import config
from pydantic import BaseModel, Field, field_validator
import jq
import json
import kubernetes
import logging
import logging.handlers
import openai

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger()
logger.level = logging.INFO

file_handler = logging.FileHandler("agent.log")
logger.addHandler(file_handler)

# Configure Kubernetes client
config.load_kube_config()


def call_kubernetes_api(api_class: str, method: str, params: dict, jq_filter: str):
    logger.info(
        f"Kubernetes API call: {api_class}.{method}({params}) with filter {jq_filter}"
    )
    try:
        api_instance = getattr(kubernetes.client, api_class)()
        api_method = getattr(api_instance, method)
        result = api_method(**params, _preload_content=False)
        result = json.loads(result.data)
        logger.info(f"Kubernetes API result: {json.dumps(result, default=str)}")

        result = jq.compile(jq_filter).input(result).all()
        logger.info(
            f"Kubernetes API filtered result: {json.dumps(result, default=str)}"
        )

        return result

    except Exception as e:
        logger.error(f"Kubernetes API error: {e}")
        return json.dumps({"error": str(e)})


class CallKubernetesAPI(BaseModel):
    """
    Dynamically call a Kubernetes API using the Kubernetes Python Client library by specifying the API class, method, and parameters and return the result.
    """

    api_class: str = Field(
        ...,
        description="""
        The name of the Kubernetes Python client API class to use for the API call, which must be a valid class as documented in the Kubernetes Python client library.
        (e.g., `CoreV1Api`, `AppsV1Api`)
        """,
    )
    method: str = Field(
        ...,
        description="""
        The specific method to call on the provided `api_class`, which must be a valid method of the specified Kubernetes API class as documented in the Kubernetes Python client library
        (e.g., `list_namespaced_pod`, `read_namespaced_service`)
        """,
    )
    params: str | dict = Field(
        ...,
        description="""
        A JSON object as a string containing the parameters required for the specified `method` of the `api_class`, which must be valid arguments as documented in the Kubernetes Python client library.
        If no parameters are needed, provide an empty JSON object `{}`.
        (e.g., `{"name": "my-pod", "namespace": "my-namespace"}`).
        """,
    )
    jq_filter: str = Field(
        ...,
        description="""
        A jq filter to extract relevant information and metadata from the API response. This filter should be a valid jq filter string that can be applied without errors. Use filter expressions to extract the necessary information and relevant metadata to answer the user query.
        If no filtering is needed, provide `'.'` this as default.
        (e.g., '[.items[] | select(.metadata.name | test("my-deployment")) | pick(.metadata.name, .metadata.namespace, .metadata.labels, .spec.template.metadata, .spec.template.spec.containers)]').
        """,
    )

    @field_validator("params")
    @classmethod
    def validate_and_convert_params(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string for 'params' - {value}")
        if isinstance(value, dict):
            return value
        raise TypeError("'params' must be a dictionary or a valid JSON string.")


class ReadNamespacedPodLogParams(BaseModel):
    """
    Represents the parameters to fetch logs for a specific pod or a container within a pod using `read_namespaced_pod_log` method from Kubernetes Python Client library.
    The `optional_params` field can be used to provide additional parameters like container name, tail lines, etc.
    """

    pod_name: str = Field(..., description="The name of the pod to fetch logs for.")
    namespace: str = Field(..., description="The namespace of the pod.")
    optional_params: dict | None = Field(
        ...,
        description="The optional parameters, such as container name and tail lines, to fetch logs based on the user query.",
    )


class FetchLogs(BaseModel):
    """
    Represents the list of logs that need to be fetched for pods or containers within a pod.
    """

    list: list[ReadNamespacedPodLogParams]


class FinalAnswer(BaseModel):
    """
    Represents the final answer to a user query that can be sent back directly.
    """

    answer: str


class Error(BaseModel):
    """
    Represents a response where the query could not be resolved or error occurred.
    """

    message: str


class OpenAIResponse(BaseModel):
    """
    Represents the final response from the OpenAI API, which can be a final direct answer to the user, list of logs to fetch or an error.
    """

    data: FinalAnswer | FetchLogs | Error


SYSTEM_PROMPT = """
You are a smart, intelligent, self-learning and helpful Kubernetes assistant designed to interact with a Kubernetes cluster to answer user queries about its deployed applications. Use the `call_kubernetes_api` function to dynamically fetch details from the cluster, filter and remove unwanted data from the API response which are not related to user query, process the data and provide concise, accurate answers.


#### Function Details

The `call_kubernetes_api` function dynamically calls Kubernetes APIs using the Kubernetes Python Client library and returns the API call result. It takes four parameters:

1. `api_class (string)`: The name of the Kubernetes Python client API class to use for the API call. This must be a valid class as documented in the Python client library (e.g., `CoreV1Api`, `AppsV1Api`).

2. `method (string)`: The specific method to call on the provided `api_class`. This must be a valid method of the specified Kubernetes API class as documented in the Kubernetes Python library (e.g., `list_namespaced_pod`, `read_namespaced_service`).

3. `params (JSON object as string)`: A JSON object as a string containing the parameters required for the specified `method` of the `api_class`. These parameters must be valid arguments of the specified `method` of the `api_class` as documented in the Kubernetes Python client library. If no parameters are needed, provide an empty JSON object `{}`. If no namespace is explicitly specified use list resource for all namespace and filter the relavent resource. (e.g., `{"name": "example-service"}`, `{"name": "example-pod", "namespace": "example-namespace"}`).

4. `jq_filter (string)`: A jq filter to apply to the API response to remove the unwanted data from the API response which are not related to the user query. This filter should be a valid jq filter string that can be applied to the API response received using the specified `api_class`, `method` and `params`. For example:
   - To extract pod names from a list of pods: `[.items[] | .metadata.name]`.
   - If no filtering is needed, provide `'.'` as default.


#### Guidelines for Solving User Queries

1. Understand the user query:
   - Break down the query, understand the context and information requested about the Kubernetes cluster.
   - Based on the Kubernetes API documentation, understand the relationships between resources such as pods, deployments, services, secrets, volumes, and statefulsets and their labels, selectors, metadata fields and their relationships.
   - Think and determine the logical steps required to gather the needed data from the Kubernetes cluster to answer the query based on the general understanding of Kubernetes resources and their relationships.
   - If the query is ambiguous or unclear, use your best judgment to interpret the user's intent.
   - The user query may not contain the exact resource names, so always use the list resource for all namespace based APIs, remove the unwanted fields from the API response and find the closest match based on the query from the filtered data.
   - If the namespace is not explicitly mentioned, use the list resource for all namespace based APIs and filter the specific resource.
   - Unless explicitly mentioned, assume that the query is related to applications deployed on the Kubernetes cluster by the user and not about kube-system or resources created by Kubernetes itself.
   - If the query is to output logs of pods or containers, return the list of params that will be used to fetch the required logs using `read_namespaced_pod_log` method in the expected structured output schema.

2. Gather the relavent information from the Kubernetes cluster:

   1. Determine the API call (`api_class`, `method`, and `params`):
      - Based on the user query and logical steps, identify the appropriate Kubernetes API class (`api_class`), method (`method`), and parameters (`params`) required to fetch details from the cluster.
      - Ensure that `api_class`, `method`, and `params` are valid as per the Kubernetes Python client library documentation.
      - If no parameters are needed for `params`, provide an empty JSON object `{}`.
      - User query may contain partial or fuzzy resource names, so use the list resource for all namespace based APIs and filter names of the resources based on the query and find the closest match.
      - Make educated guesses for parameters when exact details are unavailable, ensuring inclusivity.

   2. Generate a valid jq filter:
      - Design the `jq` filter to produce a valid JSON object or array by removing unwanted data from the API response.
      - Extract all necessary fields and metadata (e.g., `name`, `namespace`, `labels`) to ensure the output is self-contained and ready for processing.
      - Always include essential metadata fields for each resource and its nested resources (e.g., `name`, `namespace`, `labels`) to maintain context and relationships between resources.
      - Utilize `jq` operators like `map` and `select` to filter data intelligently. Use functions like `contains`, `match`, or `test` for pattern-based filtering instead of exact matches.
      - Use functions like `pick` to extract only the required fields along with metadata, ensuring the output is concise yet complete.
      - Apply optional (`?`) expressions in `jq` to handle missing fields safely (e.g., `.status.phase?`).
      - Avoid overly complex filters; aim for simplicity while maintaining efficiency. Use straightforward expressions that are easy to understand and maintain.
      - Avoid hardcoding exact values; instead, use patterns or relationships (e.g., regex matching with `test()` or partial matches with `contains()`).
      - Keep metadata fields like labels in the output to preserve relationships between related resources during processing.
      - Ensure the filter extracts all relevant information from the API response and forms a valid JSON object or array that includes both essential details and metadata for further processing.

   3. Make the `call_kubernetes_api` function call along with the determined params and filters to fetch the required data from the Kubernetes cluster.

3. Process the data and if needed make additional API calls:
   - Process the data fetched from the cluster and based on the fields and values in the API resposne.
   - Based on the available data, proceed with the next logical step to answer the query.
   - If the Kubernetes API call fails or any filter output returns empty data, try extending the scope of jq filter (think about different combinatios of filter to get the needed data) to include more data or change the parameters of the `call_kubernetes_api` function to get the desired information.
   - Retry within 10 attempts to answer the query and use the best judgment to provide the most relevant answer.
   - From the data available, find the most relavent answer to the user query.

4. Output the answer as per the expected structured output schema.
   - If the query is to output logs of pods or containers, return the list of params that will be used to fetch the required logs using `read_namespaced_pod_log` method.
   - For all other queries, provide the most relevant final answer in a precise and concise format.
     - Provide single-word or numeric answers without forming complete sentences when possible (e.g., 'Running', '3').
     - For lists of items, return comma-separated items (e.g., 'item1, item2').
     - Make use of labels and annotations to provide meaningful information to the user about the pod names, deployment names, service names, etc.
     - Unless explicitly mentioned, remove Kubernetes-generated suffixes from the resource names and return the user assigned prefix. (e.g., return `mongodb` instead of `mongodb-56c598c8fc`)
     - Return the most relevant answer when multiple possibilities exist.

5. Error Handling
   - If unable to process the query after multiple attempts due to missing context or invalid inputs, respond an error message with concise reason in the format "Not able to process the query - {{{concise reason}}}".


#### Important Notes
- The user provided names may not be exact, so use the list resource for all namespace based APIs, extract the names and/or namespace of the resources, process based on the query to find the closest match and use those names to fetch the further required information from the cluster.
- Don't use hardcoded values from user queries, always use the list resource for all namespace based APIs and filter the names of the resources based on the query.
- For questions about resources in the cluster, include namespace information in the filter and unless explicitly mentioned don't consider kube-system resources.
- Refer the examples below to understand the plan of action for different types of queries and how to process them to provide the most relevant answer to the user query.
- Don't use select or test functions in jq based on judgement, always filter the resource along with the namespace and then process the data to get the initial relavent information.


#### Example Queries and Responses

- Question: "Which pod is spawned by my-deployment?"
  Plan of Action:
    1. The query asks for the pod spawned by a specific deployment named "my-deployment."
    2. Use the `list_deployment_for_all_namespaces` method from the `AppsV1Api` with `{}` as params to retrieve all deployments in the cluster.
    3. Use filter `.items[] | pick(.metadata.name, .metadata.namespace)` to extract all the deployment name and its namespace.
    4. From the filtered data, find the deployment with the name closest match to "my-deployment" and get its namespace.
    5. Since pods created by deployments will have the same name as the deployment with some suffix appended, return the deployment name as the answer. (Because the user wants the pod name without the suffix). [NOTE: If the user wants the pod name with the suffix, then from the `list_pod_for_all_namespaces` filter the pod names starting with "my-deployment" and then process and return the answer.]
  Response: "my-pod"

- Question: "What is the status of pod named example-pod?"
  Plan of Action:
    1. The query asks for the status of a specific pod named "example-pod."
    2. Use the `list_pod_for_all_namespaces` method from the `CoreV1Api` to retrieve all pods in the cluster.
    3. Filter the results using `[.items[] | pick(.metadata.name, .metadata.namespace)]` to get the pod name and namespace.
    4. From the filtered data, find the pod with the name closest match to "example-pod".
    5. Get the status of the pod using `read_namespaced_pod` method from the `CoreV1Api` with params containing name and namespace of the filtered pod and extract the status related information from the response using filter `.status`.
    6. Now process the status information and return the status of the pod.
  Response: "Running"

- Question: "How many nodes are there in the cluster?"
  Plan of Action:
    1. The query is asking for the number of nodes in the Kubernetes cluster.
    2. To find this information, we need to call the `list_node` method from the `CoreV1Api` class with params as `{}`.
    3. Filter the names from the api response using `[.items[] | pick(.metadata.name)]`.
    4. Process the filterd data and count the number of nodes.
  Response: "2"
"""

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

        max_attempts = (
            10  # Maximum number of attempts or chat conversations to resolve the query
        )
        for _ in range(max_attempts + 1):
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
                # recieved final answer from GPT
                openai_response = response.message.parsed

                if isinstance(openai_response.data, FinalAnswer):
                    final_answer = openai_response.data.answer
                    logger.info(f"OpenAI answer: {final_answer}")

                    response_model = QueryResponse(query=query, answer=final_answer)
                    return jsonify(response_model.model_dump())

                elif isinstance(openai_response.data, FetchLogs):
                    fetch_logs = openai_response.data.list
                    logger.info(f"OpenAI answer: {fetch_logs}")

                    logs = []
                    for log in fetch_logs:
                        logs.append(
                            kubernetes.client.CoreV1Api().read_namespaced_pod_log(
                                log.pod_name,
                                log.namespace,
                                **(log.optional_params or {}),
                            )
                        )
                    logger.info(f"Fetched logs: {logs}")

                    logs = "\n".join(logs)
                    response_model = QueryResponse(query=query, answer=logs)
                    return jsonify(response_model.model_dump())

                elif isinstance(openai_response.data, Error):
                    error_message = openai_response.data.message
                    logger.error(f"OpenAI answer: {error_message}")

                    response_model = QueryResponse(query=query, answer=error_message)
                    return jsonify(response_model.model_dump())

            elif response.finish_reason == "tool_calls":
                # openai requests cluster related information from the tool calls
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
        response_model = QueryResponse(query=query, answer=error_message)
        return jsonify(response_model.model_dump())

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        response_model = QueryResponse(query=query, answer=f"Error: {e}")
        return jsonify(response_model.model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
