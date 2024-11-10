import pytest
from unittest.mock import Mock, patch
from src.evaluator.human_eval import HumanEvalEvaluator

def test_human_eval_initialization():
    evaluator = HumanEvalEvaluator(
        model_name="codegemma",
        k=1,
        num_problems=2,
        temperature=0.2
    )
    assert evaluator.model_name == "codegemma"
    assert evaluator.k == 1
    assert evaluator.num_problems == 2
    assert evaluator.temperature == 0.2

def test_human_eval_dataset_loading():
    evaluator = HumanEvalEvaluator(
        model_name="codegemma",
        k=1,
        num_problems=5
    )
    evaluator.load_dataset()
    assert len(evaluator.problems) == 5
    assert isinstance(evaluator.problems, dict)
    assert all(isinstance(p, dict) for p in evaluator.problems.values())
    assert all('prompt' in p for p in evaluator.problems.values())

def test_human_eval_single_problem_evaluation():
    with patch('src.evaluator.human_eval.HumanEvalEvaluator.generate_completion') as mock_generate:
        # Mock a successful code completion
        mock_generate.return_value = """def add(a, b):
    return a + b"""
        
        evaluator = HumanEvalEvaluator(
            model_name="codegemma",
            k=1,
            num_problems=1,
            num_samples=1
        )
        
        # Create a simple test problem
        evaluator.problems = {
            "HE_test": {
                "prompt": "def add(a, b):",
                "entry_point": "add",
                "test": """
def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(-1, 1) == 0
"""
            }
        }
        
        result = evaluator.evaluate()
        
        assert isinstance(result, dict)
        assert 'pass@k' in result
        assert result['num_problems'] == 1
        assert result['k'] == 1
        assert result['model'] == "codegemma" 