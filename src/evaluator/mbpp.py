from typing import Dict, Any
import json
import requests
from tqdm import tqdm

from . import BaseEvaluator

class MBPPEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, k: int, num_problems: int = None, temperature: float = 0.2, max_length: int = 512, num_samples: int = 1):
        super().__init__(model_name, temperature, max_length, k, num_problems, num_samples)

    def load_dataset(self):
        # Data source: https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/MBPP/data/mbpp.jsonl
        mbpp_url = "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Coder/main/Evaluation/MBPP/data/mbpp.jsonl"
        response = requests.get(mbpp_url)
        all_problems = [json.loads(line) for line in response.text.strip().split('\n')]
        self.problems = all_problems[:self.num_problems] if self.num_problems else all_problems


    def evaluate(self) -> Dict[str, Any]:
        if not self.model or not self.problems:
            self.load_model()
            self.load_dataset()

        # Generate stopping criteria for MBPP (source: https://github.com/bigcode-project/bigcode-evaluation-harness/pull/125/commits/4110d7d9a2c0a840ebc870f629ea2a2dd01c2147)
        stopping_list = ["[DONE]", "[END]", "[STOP]", "[END OF CODE]", "\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]

        total = len(self.problems)
        correct_problems = 0
        
        for problem in tqdm(self.problems, desc="MBPP progress"):
            # Try up to num_samples times to better estimate the pass@k (the more samples, the better the estimate)
            correct_solution = 0
            for _ in range(self.num_samples):
                # Try up to k times until we find a working solution
                for _ in range(self.k):
                    prompt = f"You are an expert Python programmer, and here is your task: {problem['text']}\n Your code should pass these tests:\n\n{problem['test_list']}\n[BEGINNING OF CODE]\n"
                    solution = self.generate_completion(prompt, stopping_list=stopping_list)
                    
                    try:
                        solution_code = solution.split("[BEGINNING OF CODE]")[1].split("[END OF CODE]")[0]
                        exec(solution_code, globals())
                        
                        # Check if all tests pass
                        all_tests_pass = all(
                            eval(test_case.replace('assert ', ''), globals())
                            for test_case in problem['test_list']
                        )
                        
                        if all_tests_pass:
                            correct_solution += 1
                            break  # No need to try more potential solutions for this problem
                            
                    except:
                        continue
            
            correct_problems += correct_solution / self.num_samples
        
        return {
            "pass@k": correct_problems / total,
            "num_problems": total,
            "model": self.model_name,
            "k": self.k,
            "temperature": self.temperature
        }
