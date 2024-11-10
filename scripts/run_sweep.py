import argparse
import itertools
import logging
import json
import os
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

from src.evaluator.human_eval import HumanEvalEvaluator
from src.evaluator.mbpp import MBPPEvaluator
from src.evaluator.multiple import MultiplEEvaluator
from src.utils.device import print_device_info
from run_single_eval import run_evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS = ['codegemma', 'deepseek']
BENCHMARKS = ['humaneval', 'mbpp']
K_VALUES = [1, 3, 5]

def get_evaluator(benchmark: str, model: str, k: int, num_problems: int = None, temperature: float = 0.2, max_length: int = 512):
    """Get the appropriate evaluator based on benchmark name"""
    evaluators = {
        'humaneval': HumanEvalEvaluator,
        'mbpp': MBPPEvaluator,
        'multiple': MultiplEEvaluator
    }
    
    if benchmark not in evaluators:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(evaluators.keys())}")
    
    return evaluators[benchmark](model_name=model, k=k, num_problems=num_problems, temperature=temperature, max_length=max_length)

def run_sweep(
    models: List[str] = MODELS,
    benchmarks: List[str] = BENCHMARKS,
    k_values: List[int] = K_VALUES,
    num_problems: int = None,
    temperature: float = 0.2,
    max_length: int = 1024,
    max_workers: int = 1,
    output_file: str = "sweep_results.json"
) -> List[Dict[str, Any]]:
    """
    Run evaluations for all combinations of models, benchmarks, and k values
    
    Args:
        models: List of models to evaluate
        benchmarks: List of benchmarks to run
        k_values: List of k values for Pass@k metric
        num_problems: Number of problems to evaluate per benchmark
        temperature: Temperature for generation
        max_length: Max length for generation
        max_workers: Maximum number of parallel processes
        output_file: File to save complete sweep results
    """
    # Print device info at the start of sweep
    print_device_info(logger)
    
    # Generate all combinations
    configs = [
        {
            'model': model,
            'benchmark': benchmark,
            'k': k,
            'num_problems': num_problems,
            'temperature': temperature,
            'max_length': max_length
        }
        for model, benchmark, k in itertools.product(models, benchmarks, k_values)
    ]
    
    results = []
    start_time = datetime.now()
    
    logger.info(f"Starting sweep with {len(configs)} configurations")
    logger.info(f"Max workers: {max_workers}")
    
    # Add progress bar for the sweep
    for params in tqdm(configs, desc="Evaluating combinations"):
        # Run evaluations in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(run_evaluation, **params): params
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {len(results)}/{len(configs)} evaluations")
                except Exception as e:
                    logger.error(f"Evaluation failed for {config}: {str(e)}")
                    results.append({
                        **config,
                        'error': str(e),
                        'status': 'failed'
                    })
    
    # Save complete sweep results
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', output_file), 'w') as f:
        json.dump({
            'results': results,
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'num_configurations': len(configs),
                'successful_evaluations': len([r for r in results if 'error' not in r])
            }
        }, f, indent=2)
    
    logger.info(f"Sweep completed. Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run a sweep of model evaluations')
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=MODELS,
        default=MODELS,
        help='Models to evaluate'
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        choices=BENCHMARKS,
        default=BENCHMARKS,
        help='Benchmarks to run (default: humaneval, mbpp, multiple)'
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
        '--k-values',
        nargs='+',
        type=int,
        choices=K_VALUES,
        default=K_VALUES,
        help='k values for Pass@k metric (default: 1, 3, 5)'
    )
    
    parser.add_argument(
        '--num-problems',
        type=int,
        default=50,
        help='Number of problems to evaluate per benchmark (default: 50)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel processes'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='sweep_results.json',
        help='File to save complete sweep results'
    )
    
    args = parser.parse_args()
    
    try:
        run_sweep(
            models=args.models,
            benchmarks=args.benchmarks,
            k_values=args.k_values,
            num_problems=args.num_problems,
            temperature=args.temperature,
            max_length=args.max_length,
            max_workers=args.max_workers,
            output_file=args.output_file
        )
    except Exception as e:
        logger.error(f"Sweep failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 