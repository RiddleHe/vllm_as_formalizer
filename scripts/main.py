import argparse, importlib, os, logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datafmt="%Y-%m-%d %H:%M:%S",
    )

class Subtask:
    def __init__(self, root):
        self.root = root
        self.name = root.name
        self.observation_dir = root / "observation"
        self.domain_path = root / "domain.pddl"
        self.instruction_path = root / "instruction.txt"

    def load_inputs(self):
        imgs = sorted(self.observation_dir.glob("*.png"))
        domain_pddl = self.domain_path.read_text(encoding="utf-8")
        instruction = self.instruction_path.read_txt(encoding="utf-8")
        return imgs, domain_pddl, instruction

def discover_subtasks(root):
    subtasks = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidate = Subtask(child)
            subtasks.append(candidate)
    if not subtasks:
        return RuntimeError(f"No valid subtasks found in {root}")
    return subtasks

def import_object(module_path, attr_name):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except Exception as e:
        raise

def load_model_interface(model_module, model_name):
    load_model = import_object(model_module, "load_model")
    model = load_model(model_name)
    return model

def load_pipeline_interface(pipeline_module, pipeline_name):
    load_pipeline = import_object(pipeline_module, "load_pipeline")
    pipeline = load_pipeline(pipeline_name)
    return pipeline

def build_results_root(dataset, pipeline, model):
    name = f"{dataset}_{pipeline}_{model}"
    return (Path(__file__).resolve().parent / ".." / "results" / name).resolve()

def write_prediction(out_root, subtask, predicted_pddl):
    subtask_out_dir = out_root / subtask.name
    subtask_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = subtask_out_dir / f"problem.pddl"
    out_path.write_text(predicted_pddl, encoding="utf-8", newline="\n")
    return out_path

def run(
    dataset_dir, dataset_name, 
    model_module, model_name,
    pipeline_module, pipeline_name,
):
    subtasks = discover_subtasks(dataset_dir)
    logging.info(f"Found {len(subtasks)} subtasks in {dataset_dir}")

    model = load_model_interface(model_module, model_name)
    pipeline = load_pipeline_interface(pipeline_module, pipeline_name)

    results_root = build_results_root(dataset_name, pipeline_name, model_name)
    results_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results will be saved to {results_root}")

    saved_paths = []
    for subtask in tqdm(subtasks, desc="Subtasks", unit="task"):
        try:
            observations, domain_pddl, instruction = subtask.load_inputs()
            predicted_pddl = pipeline.predict_problem_pddl(
                observations=observations,
                domain_pddl=domain_pddl,
                instruction=instruction,
                model=model,
            )
            out_path = write_prediction(results_root, subtask, predicted_pddl)
            saved_paths.append(out_path)
            logging.info(f"Saved prediction to {out_path}")
        except Exception as e:
            logging.error(f"Failed on subtask {subtask.name}: {e}")

    return saved_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_module", type=str, default="pipelines.load_models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pipeline_module", type=str, default="pipelines.load_pipelines")
    parser.add_argument("--pipeline", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()

    try:
        saved = run(
            dataset_dir=args.dataset_dir,
            dataset_name=args.dataset,
            model_module=args.model_module,
            model_name=args.model,
            pipeline_module=args.pipeline_module,
            pipeline_name=args.pipeline,
        )
        logging.info(f"Completed with {len(saved)} predictions.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()