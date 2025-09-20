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

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def open_file(path):
    with open(path, "r") as f:
        return f.read()

def translate_plan(raw_plan, mapping):
    out = []
    for name, args in raw_plan:
        missing = [a for a in args if a not in mapping]
        if missing:
            # print(f"[SKIP] unmapped args {missing} in action: {name}({', '.join(args)})")
            continue
        out.append((name, [mapping[a] for a in args]))
    return out

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
        'robot': 'agent1',
    }

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _lev(a, b): # dp implementation
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        cur = [i]
        for j, char_b in enumerate(b, 1):
            insert = cur[-1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (char_a != char_b)
            cur.append(min(insert, delete, sub))
        prev = cur
    return prev[-1]

def _lev_similarity(a, b):
    if not a and not b: return 1.0
    d = _lev(a, b)
    return 1.0 - d / max(len(a), len(b))

def suggest_mapping_with_llm(targets, leftovers):
    if not leftovers:
        return {}, []
    prompt = {
        "role": "user",
        "content": json.dumps({
            "instruction": (
                "Map each leftover object to ONE existing problem object or mark as unmappable. "
                "Do not invent names. Do not reuse a problem object. "
                "Each leftover object should be mapped to a single problem object and vice versa. "
                "If there is no reasonable mapping, mark as unmappable. "
                "For example, all names beginning with '?' is a variable and should not map to anything. " 
                "Return only JSON with keys 'mapping' and 'unmappable'."
            ),
            "unmapped_raw_tokens": leftovers,
            "problem_objects": targets,
            "example": {
                "input": {
                    "unmapped_raw_tokens": ["robo-arm", "peach"],
                    "problem_objects": ["agent1", "orange"],
                },
                "output": {
                    "mapping": {"robo-arm": "agent1", "peach": "orange"},
                    "unmappable": [],
                }
            }
        }, ensure_ascii=False)
    }
    resp = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[prompt],
        temperature=0.2,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end+1]) if start != -1 and end != -1 else {}
    mapping = {k: v for k, v in data.get("mapping", {}).items() if v in set(targets)}
    unmappable = data.get("unmappable", [])
    return mapping, unmappable

def build_mapping(raw_plan, problem, privileged=None, best_score_threshold=1.0, edit_sim_threshold=0.6, allow_unmapped=False):
    raw_objs = {a for _, args in raw_plan for a in args}
    candidate_names = list(problem.objects.keys())
    candidate_tokens_map = {c: _token_set(c) for c in candidate_names}
    mapping = {}

    if privileged:
        for raw, target in privileged.items():
            if raw not in raw_objs:
                continue
            if target not in problem.objects:
                continue
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
        if best_score > best_score_threshold:
            mapping[raw] = best_candidate
            continue

        nraw = _norm(raw) # edit distance as last resort to avoid overwrites
        best_edit_candidate, best_edit_sim = None, -1.0
        for c in candidate_names:
            sim = _lev_similarity(nraw, _norm(c))
            if sim > best_edit_sim:
                best_edit_sim, best_edit_candidate = sim, c
        if best_edit_sim >= edit_sim_threshold:
            mapping[raw] = best_edit_candidate
    
    leftovers = sorted([raw for raw in raw_objs if raw not in mapping])
    if leftovers:
        targets = [o for o in problem.objects.keys() if o not in set(mapping.values())]
        llm_map, llm_unmappable = suggest_mapping_with_llm(targets, leftovers)
        used_targets = set(mapping.values())
        for k, v in llm_map.items():
            if k in leftovers and k not in mapping and v not in used_targets: 
                mapping[k] = v
                used_targets.add(v)
        leftovers = [raw for raw in leftovers if raw not in mapping]
        if leftovers:
            if allow_unmapped:
                print(f"[WARN] no mapping for {leftovers!r}")
            else:
                raise ValueError(f"No candidate for {leftovers!r}")
        
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
            mapping = build_mapping(raw_plan, problem, privileged=_build_privileged(), allow_unmapped=True)
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