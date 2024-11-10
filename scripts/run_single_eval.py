import argparse
import logging
from typing import Dict, Any
from src.utils.device import print_device_info

from src.evaluator.human_eval import HumanEvalEvaluator
from src.evaluator.mbpp import MBPPEvaluator
from src.evaluator.multiple import MultiplEEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_evaluator(benchmark: str, model: str, k: int, num_problems: int = None, temperature: float = 0.2, max_length: int = 512, num_samples: int = 1):
    """Get the appropriate evaluator based on benchmark name"""
    evaluators = {
        'humaneval': HumanEvalEvaluator,
        'mbpp': MBPPEvaluator,
        'multiple': MultiplEEvaluator
    }
    
    if benchmark not in evaluators:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(evaluators.keys())}")
    
    return evaluators[benchmark](model_name=model, k=k, num_problems=num_problems, temperature=temperature, max_length=max_length, num_samples=num_samples)

def run_evaluation(
    model: str,
    benchmark: str,
    k: int = 1,
    num_problems: int = None,
    temperature: float = 0.2,
    max_length: int = 512,
    num_samples: int = 1
) -> Dict[str, Any]:
    """
    Run evaluation for a specific configuration
    
    Args:
        model: Model name ('deepseek' or 'codegemma')
        benchmark: Benchmark name ('humaneval', 'mbpp', or 'multiple')
        k: k value for Pass@k metric (1, 3, or 5) (default: 1)
        num_problems: Number of problems to evaluate (default: 50)
        temperature: Temperature for generation (default: 0.2)
        max_length: Max length for generation (default: 512)
        num_samples: Number of samples to evaluate (default: 1)
    Returns:
        Dictionary containing evaluation results
    """
    # Print device info at the start of evaluation
    print_device_info(logger)
    
    # Run the evaluation
    logger.info(f"Starting evaluation for {model} on {benchmark} with k={k}, num_problems={num_problems}, temperature={temperature}, max_length={max_length}, num_samples={num_samples}")
    evaluator = get_evaluator(benchmark, model, k, num_problems, temperature, max_length)
    results = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(results, benchmark)
    logger.info(f"Evaluation completed. Pass@{k}: {results['pass@k']:.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run model evaluation on a specific benchmark')
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['deepseek', 'codegemma'],
        required=True,
        help='Model to evaluate'
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        choices=['humaneval', 'mbpp', 'multiple'],
        required=True,
        help='Benchmark to evaluate on'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Temperature for generation (default: 0.2)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Max length for generation (default: 512)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        choices=[1, 3, 5],
        required=True,
        help='k value for Pass@k metric'
    )
    
    parser.add_argument(
        '--num-problems',
        type=int,
        default=50,
        help='Number of problems to evaluate (default: 50)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to evaluate (default: 1). The higher the number, the more accurate the pass@k metric will be.'
    )

    args = parser.parse_args()
    run_evaluation(
        model=args.model,
        benchmark=args.benchmark,
        k=args.k,
        num_problems=args.num_problems,
        temperature=args.temperature,
        max_length=args.max_length,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main() 