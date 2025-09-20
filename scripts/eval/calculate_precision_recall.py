import os, re, json, argparse

from find_goal import literal_to_atom, _flatten_and, parse_problem

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def mean(xs):
    return 0.0 if not xs else sum(xs) / len(xs)

def load_mapping_index(report_path):
    data = json.loads(read_text(report_path))
    tasks = data.get("tasks", {})
    idx = {}
    for t in tasks:
        tid = t.get("task_id")
        mapping = t.get(f"mapping", {}) or {}
        if tid is not None: 
            idx[tid] = mapping
    return idx

# normalization
def norm_objects(problem, *, typed=False):
    if typed:
        return {(name, typ) for name, typ in problem.objects.items()}
    return set(problem.objects.keys())

def norm_init(problem):
    return set(problem.init)

def norm_goal(problem):
    out = set()
    for literal in _flatten_and(problem.goals):
        # no need to check equality of literals in current version
        is_neg, atom = literal_to_atom(literal)
        if not is_neg:
            out.add(atom) # positive atoms only
    return out

# mapping
def map_atom(atom, mapping):
    pred, args = atom
    mapped = []
    ok = True
    for a in args:
        if a in mapping:
            mapped.append(mapping[a])
        else:
            mapped.append(a)
            ok = False # mark current atom as invalid
    return ok, (pred, tuple(mapped))

def map_atom_set(atoms, mapping):
    out = set()
    unmapped = 0
    for atom in atoms:
        ok, mapped_atom = map_atom(atom, mapping)
        if ok:
            out.add(mapped_atom)
        else:
            unmapped += 1
    return out, unmapped

def map_typed_objects(pred_objs_dict, mapping):
    out, unmapped = set(), 0
    for pred_name, pred_type in pred_objs_dict.items():
        if pred_name in mapping:
            out.add((mapping[pred_name], pred_type))
        else:
            unmapped += 1
    return out, unmapped

# scoring
def score_sets(gt, pred):
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return {
        "sizes": {"gt": len(gt), "pred_mapped": len(pred)},
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
    }

def _pick_problem_file(base_dir, task_id):
    primary = os.path.join(base_dir, task_id, "problem.pddl")
    if os.path.exists(primary):
        return primary, None
    task_dir = os.path.join(base_dir, task_id)
    if not os.path.isdir(task_dir):
        return None, f"Missing task dir: {task_dir}"
    problem_files = [
        os.path.join(task_dir, f) 
        for f in sorted(os.listdir(task_dir)) 
        if f.lower().endswith(".pddl") and f.lower().startswith("problem")
    ]
    if not problem_files:
        return None, f"Missing problem file in {task_dir}"
    return problem_files[-1], None # return last attempt

# per task eval
def eval_task(task_id, result_dir, data_dir, mapping):
    gt_path, gt_err = _pick_problem_file(data_dir, task_id)
    pred_path, pred_err = _pick_problem_file(result_dir, task_id)

    if gt_err:
        return {"task_id": task_id, "problem_eval": {"error": f"GT: {gt_err}"}}
    if not os.path.exists(pred_path):
        return {"task_id": task_id, "problem_eval": {"error": f"Predicted: {pred_err}"}}

    # TODO: no mapping means precision/recall is 0, not a bug
    if mapping is None:
        return {"task_id": task_id, "problem_eval": {"error": "Missing mapping for this task"}}

    try:
        gt_prob = parse_problem(read_text(gt_path))
        pred_prob = parse_problem(read_text(pred_path))
    except Exception as e:
        return {"task_id": task_id, "problem_eval": {"error": f"parse_problem failed: {e}"}}
    
    gt_objs = norm_objects(gt_prob, typed=True)
    gt_init = norm_init(gt_prob)
    gt_goal = norm_goal(gt_prob)

    pred_objs_raw = norm_objects(pred_prob, typed=typed_objects)
    pred_init_raw = norm_init(pred_prob)
    pred_goal_raw = norm_goal(pred_prob)

    pred_objs, unmapped_objs = map_object_names(pred_objs_raw, mapping, exclude_unmapped=True)
    pred_init, unmapped_init = map_atom_set(pred_init_raw, mapping)
    pred_goal, unmapped_goal = map_atom_set(pred_goal_raw, mapping)

    sec_objects = score_sets(gt_objs, pred_objs)
    sec_init = score_sets(gt_init, pred_init)
    sec_goal = score_sets(gt_goal, pred_goal)

    macro_precision = mean([sec_objects['precision'], sec_init['precision'], sec_goal['precision']])
    macro_recall = mean([sec_objects['recall'], sec_init['recall'], sec_goal['recall']])

    tp_all = sec_objects['true_positives'] + sec_init['true_positives'] + sec_goal['true_positives']
    fp_all = sec_objects['false_positives'] + sec_init['false_positives'] + sec_goal['false_positives']
    fn_all = sec_objects['false_negatives'] + sec_init['false_negatives'] + sec_goal['false_negatives']

    micro_precision = 1.0 if (tp_all + fp_all) == 0 else tp_all / (tp_all + fp_all)
    micro_recall = 1.0 if (tp_all + fn_all) == 0 else tp_all / (tp_all + fn_all)

    return {
        "task_id": task_id,
        "problem_eval": {
            "config": {
                "gt_problem_path": os.path.abspath(gt_path),
                "pred_problem_path": os.path.abspath(pred_path),
                "typed_objects": bool(typed_objects),
            },
            "sections": {
                "objects": sec_objects,
                "init": sec_init,
                "goal": sec_goal,
            },
            "overall": {
                "macro": {"precision": macro_precision, "recall": macro_recall},
                "micro": {"true_positives": tp_all, "false_positives": fp_all, "false_negatives": fn_all, 
                        "precision": micro_precision, "recall": micro_recall},
            },
            "unmapped": {
                "object_count": unmapped_objs,
                "init_count": unmapped_init,
                "goal_count": unmapped_goal
            }
        }
    }

# top summary
def build_summary(task_results):
    sections = ["objects", "init", "goal"]
    tp_s = {s: 0 for s in sections}
    fp_s = {s: 0 for s in sections}
    fn_s = {s: 0 for s in sections}
    macro_ps = {s: [] for s in sections}
    macro_rs = {s: [] for s in sections}

    counts = {
        "task_total": len(task_results),
        "task_scored": 0,
        "task_missing_pred_problem": 0,
        "task_with_parse_error": 0,
    }

    for task_result in task_results:
        problem_eval = task_result.get("problem_eval", {})
        if "error" in problem_eval:
            msg = str(problem_eval["error"]).lower()

            # TODO: double check have covered all error types
            if "missing predicted" in msg:
                counts["task_missing_pred_problem"] += 1
            else:
                counts["task_with_parse_error"] += 1
            continue

        counts["task_scored"] += 1
        for s in sections:
            sec = problem_eval["sections"][s]
            tp_s[s] += sec["true_positives"]
            fp_s[s] += sec["false_positives"]
            fn_s[s] += sec["false_negatives"]
            macro_ps[s].append(sec["precision"])
            macro_rs[s].append(sec["recall"])

    by_section = {}
    for s in sections:
        tp, fp, fn = tp_s[s], fp_s[s], fn_s[s]
        micro_p = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
        micro_r = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
        by_section[s] = {
            "tp": tp, "fp": fp, "fn": fn,
            "macro_precision": mean(macro_ps[s]),
            "macro_recall": mean(macro_rs[s]),
            "micro_precision": micro_p,
            "micro_recall": micro_r,
        }

    tp_all = sum(tp_s.values())
    fp_all = sum(fp_s.values())
    fn_all = sum(fn_s.values())
    overall_micro_p = 1.0 if (tp_all + fp_all) == 0 else tp_all / (tp_all + fp_all)
    overall_micro_r = 1.0 if (tp_all + fn_all) == 0 else tp_all / (tp_all + fn_all)
    overall_macro_p = mean([by_section[s]['macro_precision'] for s in sections])
    overall_macro_r = mean([by_section[s]['macro_recall'] for s in sections])

    return {
        "by_section": by_section,
        "overall": {
            "tp": tp_all, "fp": fp_all, "fn": fn_all,
            "macro_precision": overall_macro_p, "macro_recall": overall_macro_r,
            "micro_precision": overall_micro_p, "micro_recall": overall_micro_r,
        },
        "counts": counts,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("data_dir")
    parser.add_argument("mapping_report_path")
    parser.add_argument("output_file")
    args = parser.parse_args()

    mapping_index = load_mapping_index(args.mapping_report_path)
    task_results = []
    for task_id in sorted(os.listdir(args.result_dir)):
        task_path = os.path.join(args.result_dir, task_id)
        if not os.path.isdir(task_path):
            continue
        mapping = mapping_index.get(task_id)
        task_results.append(eval_task(task_id, args.result_dir, args.data_dir, mapping))

    report = {
        "result_dir": args.result_dir,
        "data_dir": args.data_dir,
        "mapping_report_path": args.mapping_report_path,
        "summary": build_summary(task_results),
        "tasks": task_results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()