# Retrieval harness over the inference20 reasoning traces.
import json
import requests

# URL for your local FastAPI server
url = "http://127.0.0.1:8000/retrieve"
# file path for the inference outputs
inference_file_path = "./inference20.md"
# file path for the retrieval outputs (json)
output_file = "./retrieve_output.json"

# Example payload
# payload = {
#     "queries": ["What is the capital of France?", "Explain neural networks."] * 200,
#     "topk": 3,
#     "return_scores": True
# }

# # Send POST request
# response = requests.post(url, json=payload)
# 
# # Raise an exception if the request failed
# response.raise_for_status()
# 
# # Get the JSON response
# retrieved_data = response.json()
# 
# print("Response from server:")
# print(retrieved_data)


def read_searches():
    searches = []
    with open(inference_file_path, "r") as file:
        for line in file:
            line = line.strip() # remove leading/trailing whitespace
            if line.startswith("<search>"): # line is search query
                # remove search tags from line
                line = line.replace("<search>", '').replace("</search>", '')
                searches.append(line.strip())
    return searches


def run_experiment():
    searches = read_searches()
    print(len(searches))
    retrieved_data = {}

    for search in searches:
        payload = {
            "queries": [search],
            "topk": 3,
            "return_scores": True
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        retrieved_data[search] = response.json()

    with open(output_file, "w") as f:
        f.write(json.dumps(retrieved_data))


if __name__ == "__main__":
    run_experiment()
