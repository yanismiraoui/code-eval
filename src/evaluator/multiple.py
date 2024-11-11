from typing import Dict, Any
import json
from tqdm import tqdm
import os
from . import BaseEvaluator
import datasets

from ..multiple.evaluation import evaluate_problem

DATASET_REVISION = "8a4cb75204eb3d5855a81778db6b95bfc80c9136"

DATASET_NAMES = ['humaneval-sh', 'humaneval-r', 'humaneval-js', 'humaneval-java', 'humaneval-cs', 'humaneval-cpp', 'humaneval-d', 'humaneval-elixir', 'humaneval-go', 'humaneval-hs', 'humaneval-jl', 'humaneval-lua', 'humaneval-ml', 'humaneval-php', 'humaneval-pl', 'humaneval-r', 'humaneval-rb', 'humaneval-rkt', 'humaneval-rs', 'humaneval-scala', 'humaneval-swift', 'humaneval-ts', 'mbpp-cpp', 'mbpp-cs', 'mbpp-d', 'mbpp-elixir', 'mbpp-go', 'mbpp-hs', 'mbpp-java', 'mbpp-jl', 'mbpp-js', 'mbpp-lua', 'mbpp-ml', 'mbpp-php', 'mbpp-pl', 'mbpp-r', 'mbpp-rb', 'mbpp-rkt', 'mbpp-rs', 'mbpp-scala', 'mbpp-sh', 'mbpp-swift', 'mbpp-ts']

class MultiplEEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, k: int, num_problems: int = None, temperature: float = 0.2, max_length: int = 512, num_samples: int = None):
        super().__init__(model_name, temperature, max_length, k, num_problems, num_samples)

    def load_dataset(self):
        """ Load the MultiPL-E dataset """
        self.problems = []
        # Load all datasets
        for name in DATASET_NAMES:
            temp = datasets.load_dataset(
                "nuprl/MultiPL-E", name, revision=DATASET_REVISION
            )
            self.problems.extend(temp["test"])
        # If num_problems is specified, take only the first num_problems problems
        if self.num_problems:
            self.problems = self.problems[:self.num_problems]

    def evaluate(self) -> Dict[str, Any]:
        """ Evaluate the model on the MultiPL-E dataset """
        if not self.model or not self.problems:
            self.load_model()
            self.load_dataset()

        total = len(self.problems)
        correct_problems = 0
        list_files = []

        for problem in tqdm(self.problems, desc="MultiPL-E progress"):
            # Build stopping list from stop tokens available in the dataset
            stopping_list = problem["stop_tokens"] + ["<file_sep>"]
            completion = self.generate_completion(problem["prompt"].strip(), stopping_list=stopping_list)[len(problem["prompt"]):]

            # Save problem
            problem = {
                "name": problem["name"],
                "language": problem["language"],
                "prompt": problem["prompt"],
                "completions": completion,
                "tests": problem["tests"],
            }
            # each problem is save in a json file in the temp_problems_multiple folder
            temp_dir = "temp_problems_multiple"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_file_name = os.path.join(temp_dir, f"{problem['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)

        # Evaluate problems
        output_dir = "temp_problems_multiple_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in tqdm(list_files, desc="Evaluating problems"):
            evaluate_problem(output_dir, file, 1)

        # Read results in output_dir/temp_problems_multiple 
        results = []
        for file in tqdm(os.listdir(output_dir+"/temp_problems_multiple"), desc="Reading results"):
            with open(os.path.join(output_dir+"/temp_problems_multiple", file), "r") as f:
                results.append(json.load(f)["results"])
       
        # Calculate pass@k
        for result in results:
            # Exclude results with exit code 1 (Exception)
            result = [r for r in result if r["exit_code"] != 1]
            if len(result) == 0:
                continue
            # check if one of the results passed
            if any(r["status"] == "OK" for r in result):
                correct_problems += 1

        return {
            "pass@k": correct_problems / total,
            "num_problems": total,
            "model": self.model_name,
            "k": self.k
        }
