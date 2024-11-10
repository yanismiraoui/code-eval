from typing import Dict, Any
from human_eval.data import write_jsonl, read_problems
# Using a slightly modified version of the human eval evaluation script to evaluate on a subset of the dataset (GPU poor :sad:)
from ..human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
import os

from . import BaseEvaluator

class HumanEvalEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, k: int, num_problems: int = None, temperature: float = 0.2, max_length: int = 512, num_samples: int = 1):
        super().__init__(model_name, temperature, max_length, k, num_problems, num_samples)

    def load_dataset(self):
        self.problems = read_problems()
        if self.num_problems:
            self.problems = {k: v for k, v in list(self.problems.items())[:self.num_problems]}

    def evaluate(self) -> Dict[str, Any]:
        if not self.model or not self.problems:
            self.load_model()
            self.load_dataset()

        # Stop tokens for HumanEval (source: https://github.com/bigcode-project/bigcode-evaluation-harness/pull/125/commits/4110d7d9a2c0a840ebc870f629ea2a2dd01c2147)
        stopping_list = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]

        samples_results = []
        for _ in range(self.num_samples):
            # Generate completions
            completions = []
            for task_id, problem in tqdm(self.problems.items(), desc="Human-eval progress"):
                for _ in range(self.k):
                    completion = self.generate_completion(problem["prompt"], stopping_list=stopping_list)[len(problem["prompt"]) :].replace("<|file_separator|>", "")
                    completions.append({
                        "task_id": task_id,
                        "completion": completion
                    })

            # Save completions temporarily
            write_jsonl("temp_completions.jsonl", completions)

            # Evaluate using HumanEval's built-in evaluator
            results = evaluate_functional_correctness(
                "temp_completions.jsonl",
                k=[self.k],
                n_workers=4
            )
            samples_results.append(results)

        # Clean up temporary file
        os.remove("temp_completions.jsonl")

        return {
            "pass@k": sum(results[f"pass@{self.k}"] for results in samples_results) / len(samples_results),
            "num_problems": len(self.problems),
            "model": self.model_name,
            "k": self.k,
            "temperature": self.temperature
        } 