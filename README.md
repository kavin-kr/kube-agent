<h1 align="center">kube-agent</h1>

kube-agent is an intelligent assistant designed to interact with your Kubernetes cluster and answer queries about its deployed applications. It dynamically calls Kubernetes APIs using the Kubernetes Python Client library to provide concise and accurate responses based on your input.

## Requirements
- Python 3.10
- A running Kubernetes cluster with its kubeconfig file located at `~/.kube/config`
- [OpenAI API Key](https://platform.openai.com/docs/quickstart)

## Setup and Usage
1. **Clone the repository:**
   ```sh
   git clone https://github.com/kavin-kr/kube-agent.git
   cd kube-agent
   ```
2. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure the OpenAI API Key:**
   - Option 1: Set Environment Variable via Terminal
     ```sh
     export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
     ```
   - Option 2: Use the .env File
     ```sh
     cp .env.template .env
     # Then, replace the OPENAI_API_KEY value in the .env file with your actual API key
     ```
4. **Ensure Your Kubernetes Cluster is Running:**

   The kubeconfig file should be located at `~/.kube/config`. If needed, you can start a local cluster using [Minikube](https://minikube.sigs.k8s.io/docs/start).
   ```sh
   minikube start
   ```
5. **Deploy Applications:**

   Make sure some applications are deployed on your Kubernetes cluster so that those resources can be queried by the agent.
6. **Run kube-agent:**
   ```sh
   python main.py
   ```

## API Specifications
kube-agent responds to user queries via an HTTP POST endpoint.

#### Request:
- URL: `http://localhost:8000/query`
- Method: POST
- Payload format:
  ```json
  {
      "query": "How many pods are in the default namespace?"
  }
  ```

#### Response:
```json
{
    "query": "How many pods are in the default namespace?",
    "answer": "2"
}
```

## Supported Queries
The agent can answer various queries related to the status, information, or logs of resources deployed on the Kubernetes cluster. Some examples include:

1. Q: "Which pod is spawned by my-deployment?"
   A: "my-pod"
2. Q: "What is the status of the pod named 'example-pod'?"
   A: "Running"
3. Q: "How many nodes are there in the cluster?"
   A: "2"

## How It Works

1. **Determining the API Call:**
   Based on the user query, kube-agent identifies the relevant Kubernetes API call to fetch the required data. It determines the appropriate `api_class`, `method`, and `parameters` needed for the Kubernetes Python client.

2. **OpenAI Function Calling with RAG:**
   kube-agent leverages a Retrieval-Augmented Generation (RAG) approach by combining real-time data retrieval with OpenAI's function calling capabilities. This process ensures that the agent fetches the most current information from your Kubernetes cluster by invoking the appropriate API call with the determined parameters.

3. **Filtering API Responses:**
   The response from the Kubernetes API may contain excessive information that might not fit within the limitations of OpenAI's API context window. To address this, the agent provides a `jq_filter` along with the `api_class`, `method`, and `params`. This filter extracts only the essential information required to answer your query from the Kubernetes API response, keeping the context window concise and within manageable limits.

4. **Processing the Filtered Output:**
   The filtered output is processed by OpenAI through function calling, which then provides the final answer to the user query.

5. **Robust Error Handling:**
   If an API call fails - due to issues such as an incorrect `api_class` or method - the agent is designed to recover by trying alternative methods or adjusting parameters. This ensures that even if one approach fails, kube-agent can still retrieve and process the necessary information.
