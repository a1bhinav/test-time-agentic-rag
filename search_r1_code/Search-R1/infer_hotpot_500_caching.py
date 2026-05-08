# HotpotQA val-500 inference loop with cache-extracted retrieval and optional CPU inference.
import os
import re
from tqdm import tqdm
import transformers
import torch
import requests
from datasets import load_dataset

# if verbose is set to true, it will print out all the reasoning traces as they are generated
# if verbose is set to false, it will show a progress bar of the overall generation progress
verbose = True
output_file_path = "results/inference_w_caching.txt"

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

# Auxiliary extractor model that compresses retrieved passages into a focused cache.
# Set MODEL_HUB_DIR to a local snapshot path to avoid re-downloading; otherwise the
# model id falls back to the public HuggingFace identifier.
cache_model_id = "Qwen/Qwen2.5-7B-Instruct"
cache_model_path = os.environ.get("MODEL_HUB_DIR", cache_model_id)
cache_model = transformers.AutoModelForCausalLM.from_pretrained(
    cache_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 1},
)

def generate_cache_tags(question, information):
    # cache_prompt = f"""Your objective is to extract relevant and helpful information for the Question from the Information. Identify factual information that is relevant to the Question and can aid in the reasoning process for the original question. 
    # Guidelines: 
    # 1. Extract Relevant Information: 
    # - Select the information from the <information> </information> that directly contributes to answering the question. 
    # - Ensure that the extracted information is accurate and relevant.
    # 2. Output Format: 
    # - If there is helpful information to answer the Question: Present the information beginning with <cache> and ending with </cache> as shown below.
    # <cache> Helpful information </cache>
    # - If there is no helpful information to answer the Question: Output the following text. <cache> No helpful information found </cache>.
    # """
    cache_prompt = f"""You are an helpful information extractor. Given a query inside <query> and </query> and some retrieved document snippets inside <information></information>, your task is to extract any information that is helpful to answer the query and output them inside <cache> and </cache>.
    If no relevant or helpful information is found, output <cache>No helpful information found</cache>. Make sure you only output <cache> and </cache> with helpful information inside and nothing else."""
    cache_prompt += f"<query>{question}<query>\n"
    cache_prompt += f"<information>{information}</information>\n"
    if tokenizer.chat_template:
        cache_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    print("-------" * 5)
    print(f"Cache model prompt: {cache_prompt}")
    # Move the cache_prompt tensor to the device used by cache_model (GPU 1)
    cache_prompt = tokenizer.encode(cache_prompt, return_tensors='pt').to("cuda:1")
    cache_attention_mask = torch.ones_like(cache_prompt)
    cache_outputs = cache_model.generate(
        cache_prompt,
        attention_mask=cache_attention_mask,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3
    )
    generated_tokens = cache_outputs[0][cache_prompt.shape[1]:]
    cache_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"cache model output: {cache_output_text}")
    print("-------" * 5)
    cache_matches = re.findall(r"<cache>(.*?)</cache>", cache_output_text, re.DOTALL)
    if cache_matches:
        return " ".join(cache_matches).strip()
    else:
        return None


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

def search(query: str):
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference
    
    return _passages2string(results[0])

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


with open(output_file_path, "a") as f:
    # filter to 2 questions for testing
    questions = questions[:2]
    for i, question in enumerate(questions if verbose else tqdm(questions)):
        file_output = ""    # entry to output to file

        # clean the question and skip them if they have been processed and output to output_file_path
        # cleaned_question = question.strip()
        # if cleaned_question in processed_questions:
        #     print(f"Skipping already processed question: {cleaned_question}")
        #     continue

        # Initialize cumulative cache for the current question
        cumulative_cache = ""

        prompt = f"""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find there is a lack of knowledge or missing information, you can call the search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        After every search relevant information will be provided in <cache> and </cache>. \
        Perform reasoning on the relevant information. If more information is needed to answer the question, perform a search. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question:{question}"""

        q = "Question: " + question
        file_output += q + "\n"

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        
        if verbose:
            print(f'\n\n################# [Processing Question] ##################\n\n')
            print(f'Question: {question}')
        
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
            search_results = search(tmp_query) if tmp_query else ''
            
            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            if verbose:
                print(search_text)
            file_output += search_text + "\n"

            # Generate cache tags
            cache_tags = generate_cache_tags(question, search_results)
            if cache_tags:
                cumulative_cache += cache_tags + "\n"
            prompt += f"<cache>{cumulative_cache}</cache>"
            if verbose:
                print(f"<cache>{cumulative_cache}</cache>")

        answer = "Correct Answer (s): " + answers[i]
        file_output += "\n" + answer + "\n"
        file_output += "\n" + "---------------------------------------" + "\n\n"
        f.write(file_output)
        f.flush()
