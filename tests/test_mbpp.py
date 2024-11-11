import pytest
from unittest.mock import Mock, patch
from src.evaluator.mbpp import MBPPEvaluator

def test_mbpp_initialization():
    """ Test the initialization of the MBPPEvaluator """
    evaluator = MBPPEvaluator(
        model_name="deepseek",
        k=1,
        num_problems=2,
        temperature=0.2
    )
    assert evaluator.model_name == "deepseek"
    assert evaluator.k == 1
    assert evaluator.num_problems == 2
    assert evaluator.temperature == 0.2

def test_mbpp_dataset_loading():
    """ Test the loading of the MBPP dataset """
    evaluator = MBPPEvaluator(
        model_name="deepseek",
        k=1,
        num_problems=5
    )
    evaluator.load_dataset()
    assert len(evaluator.problems) == 5
    assert isinstance(evaluator.problems, list)
    assert all(isinstance(p, dict) for p in evaluator.problems)
    assert all('text' in p and 'test_list' in p for p in evaluator.problems)

def test_mbpp_single_problem_evaluation():
    """ Test the evaluation of a single problem """
    with patch('src.evaluator.mbpp.MBPPEvaluator.generate_completion') as mock_generate:
        # Mock a successful code completion
        mock_generate.return_value = """
            [BEGINNING OF CODE]
            def add_numbers(a, b):
                return a + b
            [END OF CODE]
        """
        
        evaluator = MBPPEvaluator(
            model_name="deepseek",
            k=1,
            num_problems=1,
            num_samples=1
        )
        
        # Create a simple test problem
        evaluator.problems = [{
            'text': 'Write a function to add two numbers',
            'test_list': [
                'assert add_numbers(2, 3) == 5',
                'assert add_numbers(-1, 1) == 0'
            ]
        }]
        
        result = evaluator.evaluate()
        
        assert isinstance(result, dict)
        assert 'pass@k' in result
        assert result['num_problems'] == 1
        assert result['k'] == 1
        assert result['model'] == "deepseek"