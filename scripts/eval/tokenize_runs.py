import os, argparse, json
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def count_tokens(text_list, tokenizer):
    enc = tokenizer(text_list, add_special_tokens=False, return_length=True)
    return sum(enc["length"])

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_prompt_response(prompt, response, pipeline):
    if pipeline in {"direct-pddl", "direct-plan", "scene-graph"}:
        return prompt, response

    start_str, end_str = "", ""

    if pipeline == "caption":
        start_str = "SCENE CAPTION:"
        end_str = "Based on the above scene caption"

    if pipeline in {"scene-graph-multi-step-batch", "scene-graph-multi-step-no-batch"}:
        start_str = "The following are some and (maybe) all the objects and their states:"
        end_str = "For the task instruction"

    start_idx = prompt.find(start_str)

    if start_idx == -1:
        return prompt, response
        
    end_idx = prompt[start_idx:].find(end_str)
    if end_idx == -1:
        return prompt, response

    intermediate_response = prompt[start_idx + len(start_str):start_idx+end_idx].strip()
    response += intermediate_response
    return prompt, response


def determine_info(basename):
    model_name = None
    if "gpt" in basename:
        model_name = "gpt-4.1"
    elif "qwen" in basename:
        model_name = "qwenvl"

    pipeline = None
    if "vila" in basename:
        pipeline = "direct-plan"
    elif "no_batch" in basename:
        pipeline = "scene-graph-multi-step-no-batch"
    elif "multi_step" in basename:
        pipeline = "scene-graph-multi-step-batch"
    elif "scene_graph_hard" in basename:
        pipeline = "scene-graph"
    elif "caption" in basename:
        pipeline = "caption"
    elif "direct_pddl" in basename:
        pipeline = "direct-pddl"

    dataset = None
    if "blocksworld_" in basename:
        dataset = "blocksworld-small"
    elif "blocksworld-real_" in basename:
        dataset = "blocksworld-real"
    elif "alfred_" in basename:
        dataset = "alfred"
    elif "cooking_" in basename:
        dataset = "cooking"

    if any(name is None for name in {model_name, pipeline, dataset}):
        return None

    if dataset in {"blocksworld-real", "alfred"}:
        return {
            "model_name": model_name, "pipeline": pipeline, "dataset": dataset
        }
    
    run = None
    if "run_" in basename:
        run = int(basename.split("run_")[1][0]) # assume always single digit <= 3
    
    if run is None:
        return None

    return {
        "model_name": model_name, "pipeline": pipeline, "dataset": dataset, "run": run
    }

def iterate_run(run_dir, tokenizer, pipeline):
    total_prompt_tok = 0
    total_response_tok = 0

    prompts_cumulative = []
    response_cumulative = []

    n_tasks = 0

    for task_dir in tqdm(sorted(os.listdir(run_dir)), desc="Iterating tasks"):
        if not os.path.isdir(os.path.join(run_dir, task_dir)):
            continue
        task_dir_path = os.path.join(run_dir, task_dir)

        try:
            prompt = read_txt(os.path.join(task_dir_path, "prompt.txt"))
            response = read_txt(os.path.join(task_dir_path, "response.txt"))
        except FileNotFoundError:
            try:
                prompt = read_txt(os.path.join(task_dir_path, "prompt-try-2.txt"))
                response = read_txt(os.path.join(task_dir_path, "response-try-2.txt"))
            except FileNotFoundError:
                continue

        parsed_prompt, parsed_response = parse_prompt_response(prompt, response, pipeline)

        prompts_cumulative.append(parsed_prompt)
        response_cumulative.append(parsed_response)
        n_tasks += 1
        
    total_prompt_tok = count_tokens(prompts_cumulative, tokenizer) if prompts_cumulative else 0
    total_response_tok = count_tokens(response_cumulative, tokenizer) if response_cumulative else 0

    if n_tasks == 0 or total_prompt_tok == 0 or total_response_tok == 0:
        return None

    return {
        "prompt_token_count": total_prompt_tok,
        "response_token_count": total_response_tok,
        "total_token_count": total_prompt_tok + total_response_tok,
        "n_tasks": n_tasks,
        "prompt_tokens_average": total_prompt_tok / n_tasks,
        "response_tokens_average": total_response_tok / n_tasks,
        "total_tokens_average": (total_prompt_tok + total_response_tok) / n_tasks,
    }

def iterate_runs(result_dir):
    tokenizer = get_tokenizer("Qwen/Qwen2.5-VL-72B-Instruct")

    total_results = []

    for run_dir in tqdm(sorted(os.listdir(result_dir)), desc="Iterating runs"):
        dir_path = os.path.join(result_dir, run_dir)
        if not os.path.isdir(dir_path):
            continue

        info = determine_info(os.path.basename(run_dir))
        if info is None:
            continue

        result = iterate_run(dir_path, tokenizer, info["pipeline"])
        if result is None:
            continue

        total_results.append({
            **info,
            **result
        })

    return total_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("output_file")
    args = parser.parse_args()

    total_results = iterate_runs(args.result_dir)
    with open(args.output_file, "w") as f:
        json.dump(total_results, f, indent=2)

if __name__ == "__main__":
    main()



        