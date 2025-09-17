import os
import re
import argparse
import statistics
from tqdm import tqdm
import json

from find_goal import (
    parse_domain, parse_problem, parse_plan,
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
        'white-robotic-arm': 'robot',
        'white_robotic_arm': 'robot',
        'robo': 'robot',
        'peach': 'orange',
        'pink': 'red',
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
        
        best_candidate, best_score = None, float('-inf')
        for c in candidate_names:
            s = _score(raw_tokens, candidate_tokens_map[c])
            if s > best_score:
                best_score = s
                best_candidate = c
        if best_score <= 0.0:
            raise ValueError(f"No candidate found for {raw!r}")

        mapping[raw] = best_candidate
        
    return mapping

def iter_dir(result_dir, data_dir):
    tasks_out = []
    successes = 0
    with_plan = 0
    planned_steps = []
    executed_steps = []

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
        has_plan = os.path.exists(plan_path)
        if not has_plan:
            tasks_out.append({
                'task_id': task_dir_path,
                'has_plan': has_plan,
                'mapping': {},
                'plan_len': 0,
            })
            continue
        plan_str = open_file(plan_path)
        with_plan += 1

        domain = parse_domain(domain_str)
        problem = parse_problem(problem_str)

        raw_plan = parse_plan(plan_str)
        planned_steps.append(len(raw_plan))

        try:
            mapping = build_mapping(raw_plan, problem, privileged=_build_privileged())
        except ValueError as e:
            print(f"Task {task_dir_path}: {e}")
            tasks_out.append({
                'task_id': task_dir_path,
                'has_plan': has_plan,
                'mapping': {},
                'plan_len': len(raw_plan),
                'error': str(e),
            })
            continue

        translated_plan = translate_plan(raw_plan, mapping)

        info = simulate(domain, problem, translated_plan, trace=False, stop_on_invalid=True)
        if info['success']:
            successes += 1
        executed_steps.append(int(info['stopped_step']))

        tasks_out.append({
            "task_id": task_dir_path,
            "has_plan": has_plan,
            "mapping": mapping,
            'plan_len': len(raw_plan),
            **info,
        })

    n_tasks = len(tasks_out)
    summary = {
        "tasks_total": n_tasks,
        "tasks_with_plan": with_plan,
        "successes": successes,
        "failures": n_tasks - successes,
        "success_rate": successes / n_tasks if n_tasks else 0.0,
        'avg_steps_planned': statistics.fmean(planned_steps) if planned_steps else 0.0, 
        'avg_steps_executed': statistics.fmean(executed_steps) if executed_steps else 0.0,
    }

    report = {
        "result_dir": os.path.abspath(result_dir),
        "data_dir": os.path.abspath(data_dir),
        'summary': summary,
        'tasks': tasks_out,
    }

    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("data_dir")
    parser.add_argument("output_file")
    args = parser.parse_args()

    report = iter_dir(args.result_dir, args.data_dir)

    with open(args.output_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()