from typing import Any, Tuple, List, Dict, Set
from dataclasses import dataclass
import argparse
import sys

# parsing expressions

def _strip_comments(text):
    lines = []
    for line in text.splitlines():
        i = line.find(';')
        if i >= 0:
            line = line[:i]
        lines.append(line)
    return '\n'.join(lines)

def _tokenize(text):
    text = text.replace('(', ' ( ').replace(')', ' ) ')
    tokens = [token for token in text.split() if token]
    return [token.lower() for token in tokens]

def parse_s_expressions(text):
    tokens = _tokenize(_strip_comments(text))
    pos = 0

    def parse_one():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of tokens.")
        token = tokens[pos]
        pos += 1

        if token == '(':
            lst = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("Unbalanced parentheses, missing ')'.")
                if tokens[pos] == ')':
                    pos += 1
                    return lst
                lst.append(parse_one())
        elif token == ')':
            raise ValueError("Unexpected ')'.")
        else:
            return token

    forms = []
    while pos < len(tokens):
        forms.append(parse_one())
    return forms

# parsing lists

def parse_variable_typed_list(seq):
    out = []
    i = 0
    while i < len(seq):
        token = seq[i]
        if not isinstance(token, str) or not token.startswith('?'):
            raise ValueError(f"Expected variable starting with '?', got {token!r}")
        varname = token
        i += 1

        if i < len(seq) and seq[i] == '-':
            i += 1
            if i >= len(seq):
                raise ValueError("Expected type name after '-'.")
            type_name = seq[i]
            i += 1

        else:
            type_name = "object"

        out.append((varname, type_name))
    return out

def parse_objects_typed_list(seq):
    out = {}
    buffer = []
    i = 0
    while i < len(seq):
        if seq[i] == '-':
            if not buffer:
                raise ValueError("Found '-' without a collected object name.")
            i += 1
            if i >= len(seq):
                raise ValueError("Expected type after '-'.")

            type_name = seq[i]
            i += 1
            for name in buffer:
                out[name] = type_name
            buffer = []
        else:
            buffer.append(seq[i])
            i += 1
    
    for name in buffer:
        out[name] = "object"
    return out

def is_var(x):
    return isinstance(x, str) and x.startswith('?')

# data structures

Atom = Tuple[str, Tuple[str, ...]]

@dataclass
class Action:
    name: str
    params: List[Tuple[str, str]]
    preconditions: Any
    effects: Any

@dataclass
class Domain:
    name: str
    predicates: Dict[str, List[Tuple[str, str]]]
    actions: Dict[str, Action]

@dataclass
class Problem:
    name: str
    domain_name: str
    objects: Dict[str, str]
    init: Set[Atom]
    goals: Any

# parsing pddl file

def _find_block(forms, head):
    for f in forms:
        if isinstance(f, list) and f:
            if f[0] == head:
                return f
    return None

def parse_domain(text):
    forms = parse_s_expressions(text)
    if not forms or not isinstance(forms[0], list) or not forms[0] or forms[0][0] != 'define':
        raise ValueError("Domain file must start with (define ...)")
    define = forms[0][1:]

    dom_head = _find_block(define, "domain")
    if not dom_head or len(dom_head) != 2 or not isinstance(dom_head[1], str):
        raise ValueError("Cannot find (domain NAME)")
    dom_name = dom_head[1]

    predicates = {}
    actions = {}

    for block in define:
        if not (isinstance(block, list) and block):
            continue
        tag = block[0]
        if tag == ":predicates":
            for predicate_def in block[1:]:
                if not (isinstance(predicate_def, list) and predicate_def):
                    raise ValueError("Bad predicate declaration.")
                pred_name = predicate_def[0]
                params = parse_variable_typed_list(predicate_def[1:])
                predicates[pred_name] = params

        elif tag == ":action":
            name = block[1] if len(block) > 1 and isinstance(block[1], str) else None
            if not name:
                raise ValueError("Action missing name.")

            params_list = []
            preconditions = ["and"]
            effects = ["and"]

            i = 2
            while i < len(block):
                key = block[i]
                i += 1

                if key == ":parameters":
                    if i >= len(block) or not isinstance(block[i], list):
                        raise ValueError(f"Action {name}: params must be a list.")
                    params_list = parse_variable_typed_list(block[i])
                    i += 1

                elif key == ":precondition":
                    if i >= len(block):
                        raise ValueError(f"Action {name}: missing precondition value.")
                    preconditions = block[i]
                    i += 1

                elif key == ":effect":
                    if i >= len(block):
                        raise ValueError(f"Action {name}: missing effect value.")
                    effects = block[i]
                    i += 1

                else:
                    pass

            actions[name] = Action(name=name, params=params_list, preconditions=preconditions, effects=effects)

    return Domain(name=dom_name, predicates=predicates, actions=actions)

def _parse_init_facts(ast_list):
    facts = set()
    for literal in ast_list:
        if not isinstance(literal, list) or not literal:
            raise ValueError("Init entries must be atoms.")
        predicate = literal[0]
        args = tuple(str(x) for x in literal[1:])
        facts.add((predicate, args))
    return facts

def parse_problem(text):
    forms = parse_s_expressions(text)
    if not forms or not isinstance(forms[0], list) or not forms[0] or forms[0][0] != 'define':
        raise ValueError("Problem must start with (define ...)")
    define = forms[0][1:]
    
    prob_head = _find_block(define, "problem")
    if not prob_head or len(prob_head) != 2:
        raise ValueError("Cannot find (problem NAME)")
    prob_name = prob_head[1]

    dom_head = _find_block(define, ":domain")
    if not dom_head or len(dom_head) != 2:
        raise ValueError("Cannot find (:domain NAME)")
    dom_name = dom_head[1]

    objects = {}
    init = set()
    goals = ["and"]

    for block in define:
        if not (isinstance(block, list) and block):
            continue
        tag = block[0]

        if tag == ":objects":
            flat = []
            for item in block[1:]:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            objects = parse_objects_typed_list(flat)

        elif tag == ":init":
            init_list = block[1:]
            init = _parse_init_facts(init_list)
        
        elif tag == ":goal":
            if len(block) != 2:
                raise ValueError("Goal must be a single expression.")
            goals = block[1]

    return Problem(name=prob_name, domain_name=dom_name, objects=objects, init=init, goals=goals)

# grounding predicates

def substitute(expression, theta):
    if isinstance(expression, str):
        return theta.get(expression, expression)
    if isinstance(expression, list):
        return [substitute(x, theta) for x in expression]
    return expression

def _flatten_and(expression):
    if isinstance(expression, list) and expression and expression[0] == "and":
        out = []
        for expr in expression[1:]:
            out.extend(_flatten_and(expr))
        return out
    else:
        return [expression]

def literal_to_atom(literal):
    if not isinstance(literal, list) or not literal:
        raise ValueError(f"Bad literal: {literal}")

    if literal[0] == "not":
        if len(literal) != 2 or not isinstance(literal[1], list) or not literal[1]:
            raise ValueError(f"Bad (not ...) literal: {literal}")
        atom = literal[1]
        predicate = atom[0]
        args = tuple(str(x) for x in atom[1:])
        return True, (predicate, args)

    else:
        predicate = literal[0]
        args = tuple(str(x) for x in literal[1:])
        return False, (predicate, args)

def holds(state, condition):
    """Currently only check conditions with connectives 'and' and 'not'"""
    if isinstance(condition, list) and condition:
        head = condition[0]
        if head == "and":
            return all(holds(state, x) for x in condition[1:])
        
        if head == "not":
            if len(condition) != 2:
                raise ValueError(f"Bad (not ...) condition: {condition}")
            _, atom = literal_to_atom(condition)
            return atom not in state

        else:
            _, atom = literal_to_atom(condition)
            return atom in state

    else:
        raise ValueError(f"Unexpected condition form: {condition}")

def apply_effects(state, effects):
    adds = set()
    dels = set()
    literals = _flatten_and(effects)

    for literal in literals:
        is_neg, atom = literal_to_atom(literal)
        if is_neg:
            dels.add(atom)
        else:
            adds.add(atom)
    
    new_state = (state - dels) | adds
    return new_state, adds, dels

def parse_plan(text):
    forms = parse_s_expressions(text)
    actions = []
    for form in forms:
        if isinstance(form, list) and form:
            actions.append(form)
    
    out = []
    for action in actions:
        if not isinstance(action, list) or not action:
            continue
        name = action[0]
        args = [str(x) for x in action[1:]]
        out.append((name, args))
    return out

# simulation driver

def type_check_binding(action, objects, binding):
    for var, type_name in action.params:
        const = binding.get(var)
        if const is None:
            raise ValueError(f"Missing binding for {var} in action {action.name}")
        if const not in objects:
            raise ValueError(f"Unknown object '{const}' for var {var} in action {action.name}")
        obj_type = objects[const]
        if type_name != "object" and type_name != obj_type:
            raise ValueError(f"Type mismatch for {var}: expected {type_name}, got {obj_type} (object {const})")

def simulate(domain, problem, plan, *, trace=True, stop_on_invalid=True):
    if domain.name != problem.domain_name:
        raise ValueError(f"Domain mismatch: domain '{domain.name}' != problem's domain '{problem.domain_name}'")

    state = set(problem.init)
    if trace:
        print(f"Initial state ({len(state)} facts):")
        for predicate, args in sorted(state):
            print(f"  ({predicate} {' '.join(args)})")
        print()
    
    for t, (action_name, args) in enumerate(plan, start=1):
        if action_name not in domain.actions:
            raise ValueError(f"Unknown action '{action_name}' at step {t}")
        action = domain.actions[action_name]
        if len(args) != len(action.params):
            raise ValueError(f"Arity mismatch at step {t} for '{action_name}': "
                             f"expected {len(action.params)}, got {len(args)}")
        
        binding = {var: const for (var, _), const in zip(action.params, args)}
        type_check_binding(action, problem.objects, binding)

        grounded_preconditions = substitute(action.preconditions, binding)
        grounded_effects = substitute(action.effects, binding)

        ok = holds(state, grounded_preconditions)

        if trace:
            print(f"Step {t}: ({action_name} {' '.join(args)})")
            print(f"  Precondition: {'OK' if ok else 'FAILED'}")
        if not ok and stop_on_invalid:
            raise RuntimeError(f"Precondition failed at step {t}: ({action_name} {' '.join(args)})")

        new_state, adds, dels = apply_effects(state, grounded_effects)
        if trace:
            if adds:
                print(f"  Add: " + ", ".join([f"({p} {' '.join(a)})" for p, a in sorted(adds)]))
            if dels:
                print(f"  Del: " + ", ".join([f"({p} {' '.join(a)})" for p, a in sorted(dels)]))
        state = new_state
        if trace:
            print()

    goal_ok = holds(state, problem.goals)
    if trace:
        print(f"Final state: ")
        for p, a in sorted(state):
            print(f"  ({p} {' '.join(a)})")
        print()
        print(f"Goal satisfied: {goal_ok}")
    return state, goal_ok 

# test

def run_test():
    dom_txt = """
(define (domain d)
  (:requirements :strips :typing)
  (:types block robot)
  (:predicates (ontable ?x - block) (clear ?x - block) (handempty ?r - robot) (handfull ?r - robot) (holding ?x - block))
  (:action pick-up
    :parameters (?x - block ?r - robot)
    :precondition (and (clear ?x) (ontable ?x) (handempty ?r))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty ?r)) (handfull ?r) (holding ?x))
  )
)"""

    prob_txt = """
(define (problem p)
  (:domain d)
  (:objects a - block r1 - robot)
  (:init (ontable a) (clear a) (handempty r1))
  (:goal (and (holding a)))
)"""

    plan_txt = "(pick-up a r1)"

    domain = parse_domain(dom_txt)
    problem = parse_problem(prob_txt)
    plan = parse_plan(plan_txt)
    _, ok = simulate(domain, problem, plan, trace=True)
    return 0 if ok else 1

# CLI

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.required = True

    p_run = subparsers.add_parser("run")
    p_run.add_argument("domain")
    p_run.add_argument("problem")
    p_run.add_argument("plan")
    p_run.add_argument("--no_trace", action="store_true")
    p_run.add_argument("--no_stop_on_invalid", action="store_true")

    p_test = subparsers.add_parser("test")

    args = parser.parse_args()

    if args.cmd == "test":
        exit_code = run_test()
        sys.exit(exit_code)

    with open(args.domain, "r") as f:
        domain_txt = f.read()
    with open(args.problem, "r") as f:
        prob_txt = f.read()
    with open(args.plan, "r") as f:
        plan_txt = f.read()

    domain = parse_domain(domain_txt)
    problem = parse_problem(prob_txt)
    plan = parse_plan(plan_txt)

    try:
        simulate(domain, problem, plan, trace=not args.no_trace, stop_on_invalid=not args.no_stop_on_invalid)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()