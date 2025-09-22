import os, json, argparse

from find_goal import literal_to_atom, _flatten_and, parse_problem

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def mean(xs):
    return 0.0 if not xs else sum(xs) / len(xs)

def load_mapping_index(report_dict):
    tasks = report_dict.get("tasks", [])
    idx = {}
    for t in tasks:
        tid = t.get("task_id")
        mapping = t.get("mapping", {}) or {}
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

def map_object_set(objs, mapping, *, typed=False, exclude_unmapped=True):
    out = set()
    unmapped = 0
    if typed:
        for name, typ in objs:
            if name in mapping:
                out.add((mapping[name], typ))
            else:
                unmapped += 1
                if not exclude_unmapped:
                    out.add((name, typ))
    else:
        for name in objs:
            if name in mapping:
                out.add(mapping[name])
            else:
                unmapped += 1
                if not exclude_unmapped:
                    out.add(name)
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

def _zero_section(gt_set, pred_set):
    return {
        "sizes": {"gt": len(gt_set), "pred_mapped": 0},
        "true_positives": 0, "false_positives": len(pred_set), "false_negatives": len(gt_set),
        "precision": 0.0, "recall": 0.0,
    }

# per task eval
def eval_task(task_id, result_dir, data_dir, mapping):
    gt_path, gt_err = _pick_problem_file(data_dir, task_id)
    pred_path, pred_err = _pick_problem_file(result_dir, task_id)

    if gt_err:
        return {"task_id": task_id, "problem_eval": {"error": f"GT: {gt_err}"}}
    if pred_err:
        return {"task_id": task_id, "problem_eval": {"error": f"Predicted: {pred_err}"}}

    try:
        gt_prob = parse_problem(read_text(gt_path))
        pred_prob = parse_problem(read_text(pred_path))
    except Exception as e:
        return {"task_id": task_id, "problem_eval": {"error": f"parse_problem failed: {e}"}}
    
    gt_objs = norm_objects(gt_prob, typed=False)
    gt_init = norm_init(gt_prob)
    gt_goal = norm_goal(gt_prob)

    pred_objs_raw = norm_objects(pred_prob, typed=False)
    pred_init_raw = norm_init(pred_prob)
    pred_goal_raw = norm_goal(pred_prob)

    if mapping is None: # have lots of fp in original generation, so p/r are 0
        sec_objects = _zero_section(gt_objs, pred_objs_raw)
        sec_init = _zero_section(gt_init, pred_init_raw)
        sec_goal = _zero_section(gt_goal, pred_goal_raw)
        return {
            "task_id": task_id,
            "problem_eval": {
                "config": {
                    "gt_problem_path": os.path.abspath(gt_path),
                    "pred_problem_path": os.path.abspath(pred_path),
                },
                "sections": {
                    "objects": sec_objects, 
                    "init": sec_init, 
                    "goal": sec_goal, 
                    "type": {"objects_checked": 0, "mismatches": 0, "agreement": 0.0}
                },
                "overall": {
                    "macro": {"precision": 0.0, "recall": 0.0}, 
                    "micro": {
                        "true_positives": 0, 
                        "false_positives": sec_objects["false_positives"] + sec_init["false_positives"] + sec_goal["false_positives"], 
                        "false_negatives": sec_objects["false_negatives"] + sec_init["false_negatives"] + sec_goal["false_negatives"], 
                        "precision": 0.0, "recall": 0.0
                    }
                },
                "unmapped": {"object_count": len(pred_objs_raw), "init_count": len(pred_init_raw), "goal_count": len(pred_goal_raw)}
            }
        }

    pred_objs, unmapped_objs = map_object_set(pred_objs_raw, mapping, typed=False, exclude_unmapped=True)
    pred_init, unmapped_init = map_atom_set(pred_init_raw, mapping)
    pred_goal, unmapped_goal = map_atom_set(pred_goal_raw, mapping)

    sec_objects = score_sets(gt_objs, pred_objs)
    sec_init = score_sets(gt_init, pred_init)
    sec_goal = score_sets(gt_goal, pred_goal)

    type_checked, type_mismatches = 0, 0
    for pred_name, pred_type in pred_prob.objects.items():
        if pred_name in mapping:
            gt_name = mapping[pred_name]
            gt_type = gt_prob.objects.get(gt_name)
            if gt_type is not None:
                type_checked += 1
                if gt_type != pred_type:
                    type_mismatches += 1
    type_agreement = (1.0 if type_checked == 0 else (type_checked - type_mismatches) / type_checked)
    sec_type = {
        "objects_checked": type_checked,
        "mismatches": type_mismatches,
        "agreement": type_agreement,
    }

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
            },
            "sections": {
                "objects": sec_objects,
                "init": sec_init,
                "goal": sec_goal,
                "type": sec_type,
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

    types_checked_total = 0
    types_mismatches_total = 0
    types_agreements = []

    counts = {
        "task_total": len(task_results),
        "task_scored": 0,
        "task_missing_pred_problem": 0,
        "task_missing_gt_problem": 0,
        "task_with_parse_error": 0,
        "task_missing_mapping": 0,
        "task_other_error": 0,
    }

    for task_result in task_results:
        problem_eval = task_result.get("problem_eval", {})
        if "error" in problem_eval:
            msg = str(problem_eval["error"]).lower()

            if "predicted:" in msg:
                counts["task_missing_pred_problem"] += 1
            elif "gt:" in msg:
                counts["task_missing_gt_problem"] += 1
            elif "parse_problem failed" in msg:
                counts["task_with_parse_error"] += 1
            elif "missing mapping" in msg:
                counts["task_missing_mapping"] += 1
            else:
                counts["task_other_error"] += 1
            continue

        counts["task_scored"] += 1
        for s in sections:
            sec = problem_eval["sections"][s]
            tp_s[s] += sec["true_positives"]
            fp_s[s] += sec["false_positives"]
            fn_s[s] += sec["false_negatives"]
            macro_ps[s].append(sec["precision"])
            macro_rs[s].append(sec["recall"])

        type_sec = problem_eval["sections"]["type"]
        types_checked_total += type_sec["objects_checked"]
        types_mismatches_total += type_sec["mismatches"]
        if type_sec["objects_checked"] > 0:
            types_agreements.append(type_sec["agreement"])

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

    types_micro_agreement = (1.0 if types_checked_total == 0 else (types_checked_total - types_mismatches_total) / types_checked_total)
    types_macro_agreement = mean(types_agreements)

    return {
        "by_section": by_section,
        "overall": {
            "tp": tp_all, "fp": fp_all, "fn": fn_all,
            "macro_precision": overall_macro_p, "macro_recall": overall_macro_r,
            "micro_precision": overall_micro_p, "micro_recall": overall_micro_r,
        },
        "types": {
            "objects_checked": types_checked_total,
            "mismatches": types_mismatches_total,
            "macro_agreement": types_macro_agreement,
            "micro_agreement": types_micro_agreement,
        },
        "counts": counts,
    }

def _merge_same_key(key, a, b):
    va, vb = a.get(key), b.get(key)
    if va is None and vb is None:
        return None
    if va == vb:
        return va
    return [va, vb]

def merge_reports(check_plan_report, p_r_report):
    summary = {
        **check_plan_report.get("summary", {}),
        **p_r_report.get("summary", {}),
    }
    plan_tasks = {task["task_id"]: task for task in check_plan_report.get("tasks", [])}
    p_r_tasks = {task["task_id"]: task for task in p_r_report.get("tasks", [])}
    merged_tasks = []
    all_ids = sorted(set(plan_tasks) | set(p_r_tasks))
    for tid in all_ids:
        merged_task = {"task_id": tid}
        if tid in plan_tasks:
            merged_task.update(plan_tasks[tid])
        if tid in p_r_tasks:
            for k, v in p_r_tasks[tid].items():
                if k != "task_id":
                    merged_task[k] = v
        merged_tasks.append(merged_task)
    
    merged = {
        "result_dir": _merge_same_key("result_dir", check_plan_report, p_r_report),
        "data_dir": _merge_same_key("data_dir", check_plan_report, p_r_report),
        # drop the mapping report path
        "summary": summary,
        "tasks": merged_tasks,
    }
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("data_dir")
    parser.add_argument("mapping_report_path")
    parser.add_argument("output_file")
    args = parser.parse_args()

    mapping_report_dict = json.loads(read_text(args.mapping_report_path))
    mapping_index = load_mapping_index(mapping_report_dict)
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

    merged_report = merge_reports(mapping_report_dict, report)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(merged_report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()