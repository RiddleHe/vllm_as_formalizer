import os
import re
import argparse
from tkinter import FALSE
from tqdm import tqdm

from find_goal import (
    parse_domain, parse_problem, parse_plan,
    Domain, Problem, Action,
    simulate,
)

def open_file(path):
    with open(path, "r") as f:
        return f.read()

def translate_plan(raw_plan, mapping):
    return [(name, [mapping[arg] for arg in args]) for name, args in raw_plan]

def _token_set(name):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return set(re.findall(r'[a-z]+', s.lower()))

def _score(raw_tokens, candidate_tokens):
    if raw_tokens == candidate_tokens:
        return 100.0
    inter = len(raw_tokens & candidate_tokens)
    union = len(raw_tokens | candidate_tokens)
    jaccard = inter / union if union else 0.0

    containment = 0.0
    if raw_tokens and raw_tokens.issubset(candidate_tokens):
        containment = max(containment, len(raw_tokens) / len(candidate_tokens))
    if candidate_tokens and candidate_tokens.issubset(raw_tokens):
        containment = max(containment, len(candidate_tokens) / len(raw_tokens))
    return 5.0 * jaccard + 4.0 * containment

def _build_privileged():
    return {
        'r1': 'robot',
        'robby': 'robot',
        'agent': 'robot',
    }

def build_mapping(raw_plan, problem, privileged=None):
    raw_objs = {a for _, args in raw_plan for a in args}
    candidate_names = list(problem.objects.keys())
    candidate_tokens_map = {c: _token_set(c) for c in candidate_names}
    mapping = {}

    if privileged:
        for raw, target in privileged.items():
            if raw not in raw_objs:
                continue
            if target not in problem.objects:
                raise ValueError(f'Privileged target {target!r} not in problem objects.')
            mapping[raw] = target

    for raw in sorted(raw_objs):
        if raw in mapping:
            continue
        if raw in problem.objects:
            mapping[raw] = raw
            continue
        raw_tokens = _token_set(raw)
        best_candidate = max(candidate_names, key=lambda c: _score(raw_tokens, candidate_tokens_map[c]))
        mapping[raw] = best_candidate
    return mapping

def iter_dir(result_dir, data_dir):
    success_count = 0
    task_count = 0

    for task_dir_path in tqdm(sorted(os.listdir(result_dir)), desc='Checking plans'):
        if not os.path.isdir(os.path.join(result_dir, task_dir_path)):
            continue

        gt_task_dir = os.path.join(data_dir, task_dir_path)
        task_dir = os.path.join(result_dir, task_dir_path)

        if not os.path.exists(gt_task_dir):
            raise FileNotFoundError(f"Task {task_dir_path} is not defined.")

        domain_str = open_file(os.path.join(gt_task_dir, "domain.pddl"))
        problem_str = open_file(os.path.join(gt_task_dir, "problem.pddl"))
        
        plan_path = os.path.join(task_dir, 'plan.txt')
        if not os.path.exists(plan_path):
            continue
        plan_str = open_file(plan_path)

        domain = parse_domain(domain_str)
        problem = parse_problem(problem_str)

        raw_plan = parse_plan(plan_str)
        mapping = build_mapping(raw_plan, problem, privileged=_build_privileged())
        translated_plan = translate_plan(raw_plan, mapping)

        print(f"task {task_dir_path}: mapping {mapping}")
        _, success = simulate(domain, problem, translated_plan, trace=False, stop_on_invalid=False)
        if success:
            success_count += 1
        task_count += 1

    print(f"Success rate: {success_count / task_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("data_dir")
    args = parser.parse_args()

    iter_dir(args.result_dir, args.data_dir)

if __name__ == "__main__":
    main()