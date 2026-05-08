# HotpotQA val-500 inference loop with file-based resume, optional CPU inference, and a tqdm progress bar.
import os
import json
from tqdm import tqdm
import transformers
import torch
import requests
from datasets import load_dataset

# if verbose is set to true, it will print out all the reasoning traces as they are generated
# if verbose is set to false, it will show a progress bar of the overall generation progress
verbose = False
output_file_path = "results/inference500_no_repeat_docs.txt"

# Global variable for the cache file path
cache_file_path = "results/retrieved_docs_cache.json"

# Ensure the cache file exists
if not os.path.exists(cache_file_path):
    with open(cache_file_path, "w") as f:
        json.dump({}, f)

# Load dataset
base_dir = os.path.join(os.environ.get("DATASET_DIR", "./data"), "nq_hotpotqa")
dataset = load_dataset("parquet", data_files=f"{base_dir}/val_split_500.parquet")["train"]
print(f"Loaded {len(dataset)} data points sampled from validataion split.")

questions = []
answers = []

for example in dataset.select(range(500)):
    question = example['question'].strip()
    if question[-1] != '?':
        question += '?'
    questions.append(question)
    answers.append("[" + " | ".join(example['golden_answers']) + "]")

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")    # CPU inference

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={"":0})
# model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")   # CPU inference

# Define stopping criteria
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None

# Modify the search function to cache retrieved documents and their scores
def search(query: str, retrieved_docs: set):
    top_k = 3
    payload = {"queries": [query], "topk": top_k, "return_scores": True}
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    new_docs = []
    cached_results = []

    while len(new_docs) < 3:
        for doc_item in results[0]:
            doc_id = doc_item['document']['id']  # Assuming each document has a unique ID
            if doc_id not in retrieved_docs:
                retrieved_docs.add(doc_id)
                new_docs.append(doc_item)
                cached_results.append({
                    "id": doc_id,
                    "contents": doc_item['document']['contents'],
                    "score": doc_item['score']
                })
                if len(new_docs) == 3:
                    break
        if len(new_docs) < 3:
            top_k += 3  # Increase top_k to fetch more documents
            payload["topk"] = top_k
            results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']

    # Cache the retrieved documents and their scores
    with open(cache_file_path, "r+") as f:
        cache = json.load(f)
        if cache.get(query) is None:
            cache[query] = {}
        for doc in cached_results:
            doc_id = doc['id']
            if cache[query].get(doc_id) is None:
                cache[query][doc_id] = {
                    "contents": doc['contents'],
                    "score": doc['score']
                }
        # Write back to the file
        f.seek(0)
        json.dump(cache, f, indent=4)
        f.truncate()

    return _passages2string(new_docs)

# Initialize stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# read the processed questions from output file if it exists
processed_questions = set()
if os.path.exists(output_file_path):
    with open(output_file_path, "r") as f:
        for line in f:
            if line.startswith("Question: "):
                processed_questions.add(line.strip().removeprefix("Question: ").strip())


# Update the main loop to track retrieved documents for each question
with open(output_file_path, "a") as f:
    for i, question in enumerate(questions if verbose else tqdm(questions)):
        file_output = ""    # entry to output to file

        # clean the question and skip them if they have been processed and output to output_file_path
        cleaned_question = question.strip()
        if cleaned_question in processed_questions:
            print(f"Skipping already processed question: {cleaned_question}")
            continue

        prompt = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as you want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
        
        q = "Question: " + question
        file_output += q + "\n"

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        
        if verbose:
            print(f'\n\n################# [Processing Question] ##################\n\n')
            print(f'Question: {question}')
        
        retrieved_docs = set()  # Track retrieved document IDs for this question
        while True:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)
            
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

            if outputs[0][-1].item() in curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if verbose:
                    print(output_text)
                file_output += output_text + "\n"
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
            # retrieved_docs will be updated since it's passed by reference
            search_results = search(tmp_query, retrieved_docs) if tmp_query else ''
            
            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            if verbose:
                print(search_text)

            file_output += search_text + "\n"
        answer = "Correct Answer (s): " + answers[i]
        file_output += "\n" + answer + "\n"
        file_output += "\n" + "---------------------------------------" + "\n\n"
        f.write(file_output)
        f.flush()
