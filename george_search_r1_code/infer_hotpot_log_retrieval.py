# this program is the same as the infer_hotpot.py, except it logs the retrieval in a file called retrieval_log.json for analysis on confidence score
import transformers
import torch
import time
import requests
from datasets import load_dataset

# path to the retrieval log file (csv)
retrieval_log_path = "./retrieval_log.csv"
# use the current utc time (integer seconds) as run_id in retrieval log
run_id = int(time.time())

# Load dataset
dataset = load_dataset("hotpot_qa", "fullwiki", split='validation')

questions = []
answers = []

for example in dataset.select(range(20)):
    question = example['question'].strip()
    if question[-1] != '?':
        question += '?'
    questions.append(question)
    answers.append(example['answer'])

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

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
        with open(retrieval_log_path, 'a') as log_file:
            writer = csv.writer(log_file)   # csv writer
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                score = doc_item['score']
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
                writer.writerow([run_id, score, title, text])
        return format_reference
    
    return _passages2string(results[0])

# Initialize stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

with open("results/inference20.txt", "w") as f:
    for i, question in enumerate(questions):
        prompt = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as you want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
        
        q = "Question: " + question
        f.write(q + "\n")

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        
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
                print(output_text)
                f.write(output_text + "\n")
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
            search_results = search(tmp_query) if tmp_query else ''
            
            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            print(search_text)

            f.write(search_text + "\n")
        answer = "Correct Answer: " + answers[i]
        f.write("\n" + answer + "\n")
        f.write("\n" + "---------------------------------------" + "\n")

