from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

from src.config import HF_TOKEN, MODEL_PATHS, DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE, DEFAULT_NUM_PROBLEMS, DEFAULT_NUM_SAMPLES
from ..utils.device import get_device

class StopOnTokens(StoppingCriteria):
    """ Stopping criteria for generation """
    def __init__(self, stop_ids_list: List[int]):
        self.stop_ids_list = stop_ids_list

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        for stop_ids in self.stop_ids_list:
            if len(stop_ids) > len(input_ids[0]):
                continue
            if torch.equal(input_ids[0][-len(stop_ids):], torch.tensor(stop_ids, device=input_ids.device)):
                return True
        return False

class BaseEvaluator(ABC):
    """ Base class for evaluators """
    def __init__(self, model_name: str, temperature: float = DEFAULT_TEMPERATURE, max_length: int = DEFAULT_MAX_LENGTH, k: int = 1, num_problems: int = DEFAULT_NUM_PROBLEMS, num_samples: int = DEFAULT_NUM_SAMPLES):
        """
        Initialize base evaluator
        Args:
            model_name: Either 'deepseek' or 'codegemma'
            temperature: Temperature for generation (default: 0.2)
            max_length: Maximum length for generation (default: 512)
            k: k value for Pass@k metric (1, 3, or 5) (default: 1)
            num_problems: Number of problems to evaluate (default: 50)
            num_samples: Number of samples to evaluate (default: 1)
        """
        self.model_name = model_name
        self.k = k
        self.num_problems = num_problems
        self.temperature = temperature
        self.max_length = max_length
        self.num_samples = num_samples
        self.model = None
        self.tokenizer = None
        self.problems = None
        self.device = get_device()

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_model(self):
        """Load the specified model using HF token"""
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_path = MODEL_PATHS[self.model_name]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map=None
        ).to(self.device)

    def generate_completion(self, prompt: str, stopping_list: List = None) -> str:
        """Generate completion using the model"""
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(self.model.device)

        if stopping_list:
            stop_ids_list = [self.tokenizer.encode(word, add_special_tokens=False) for word in stopping_list]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids_list)])
        else:
            stopping_criteria = None
        
        if inputs['input_ids'].shape[1] > self.max_length:
            print(f"Warning: Input length {inputs['input_ids'].shape[1]} exceeds max length {self.max_length}. Truncating input.")
            inputs['input_ids'] = inputs['input_ids'][:, :self.max_length]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :self.max_length]

        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.max_length,
            temperature=self.temperature,
            use_cache=True,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @abstractmethod
    def load_dataset(self):
        """Load the benchmark dataset"""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return results"""
        pass

    def save_results(self, results: Dict[str, Any], benchmark: str):
        """Save evaluation results to JSON file"""
        filename = f"{self.results_dir}/{benchmark}_{self.model_name}_k{self.k}_temperature{self.temperature}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)