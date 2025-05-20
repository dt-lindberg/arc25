#!/usr/bin/env python3
# filepath: /home/dt-lindberg/arc25/evaluation.py

import os
import json
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

# Import our data loader
from src.arc_data_loader import ARCDataLoader


def format_task_for_model(task_data):
    """
    Format an ARC task into a text prompt for the model.
    
    Args:
        task_data: Dictionary containing task data with 'train' and 'test' examples
        
    Returns:
        String containing the formatted task
    """
    prompt = "I'll provide you with examples of input-output pairs for a pattern recognition task. "
    prompt += "Each example contains a transformation from an input grid to an output grid. "
    prompt += "Your job is to understand the pattern and apply it to a new test input.\n\n"
    
    # Add training examples
    for i, example in enumerate(task_data["train"]):
        prompt += f"Training Example {i+1}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    # Add test input
    test_example = task_data["test"][0]  # Assuming we're evaluating one test case at a time
    prompt += "Now, apply the pattern you've learned to this new input:\n"
    prompt += f"Test Input: {test_example['input']}\n\n"
    prompt += "Generate the correct output following the pattern. Your response should be formatted exactly as a Python list of lists."
    
    return prompt


def evaluate_grid_equality(predicted, actual):
    """
    Check if the predicted grid matches the actual grid.
    
    Args:
        predicted: Model's predicted grid (list of lists)
        actual: Actual grid (list of lists)
        
    Returns:
        Boolean indicating whether the grids match
    """
    if not isinstance(predicted, list) or not all(isinstance(row, list) for row in predicted):
        return False
        
    # Convert to numpy arrays for easier comparison
    try:
        pred_array = np.array(predicted)
        actual_array = np.array(actual)
        
        # Check if shapes and values match
        return pred_array.shape == actual_array.shape and np.array_equal(pred_array, actual_array)
    except:
        return False


def parse_model_output(output_text):
    """
    Parse the model's output text to extract the predicted grid.
    
    Args:
        output_text: The raw output text from the model
        
    Returns:
        Grid as a list of lists, or None if parsing failed
    """
    try:
        # Find the grid in the output - typically it's a list of lists
        # This is a simple approach - we assume the model outputs the grid directly
        # For more complex outputs, we'd need more sophisticated parsing
        
        # Try direct eval first (dangerous but controlled here)
        clean_output = output_text.strip()
        
        # Find the first [ and the matching last ]
        start_idx = clean_output.find('[')
        if start_idx == -1:
            return None
            
        # Track brackets to find the matching closing bracket
        open_brackets = 0
        end_idx = -1
        
        for i, char in enumerate(clean_output[start_idx:]):
            if char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
                if open_brackets == 0:
                    end_idx = start_idx + i + 1
                    break
        
        if end_idx == -1:
            return None
            
        grid_str = clean_output[start_idx:end_idx]
        
        # Safely evaluate the string as Python code
        grid = eval(grid_str)
        
        # Validate that it's a list of lists
        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
            return grid
        return None
    except:
        return None


def evaluate_model(model, evaluation_data, num_tasks=None, verbose=False, output_file=None):
    """
    Evaluate the model's performance on ARC tasks.
    
    Args:
        model: Model that takes a task prompt and returns a prediction
        evaluation_data: ARCDataLoader instance with loaded evaluation data
        num_tasks: Number of tasks to evaluate (None to evaluate all)
        verbose: Whether to print detailed information
        output_file: Optional file to save results
    
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
        "total": 0,
        "task_results": {}
    }
    
    # Get tasks to evaluate
    task_ids = evaluation_data.task_ids
    if num_tasks:
        task_ids = task_ids[:num_tasks]
    
    start_time = time.time()
    
    for task_id in tqdm(task_ids, desc="Evaluating tasks"):
        task_data = evaluation_data.get_task(task_id)
        task_start_time = time.time()
        
        # Format task for model
        prompt = format_task_for_model(task_data)
        
        # Get model prediction
        try:
            model_output = model(prompt)
            predicted_grid = parse_model_output(model_output)
            
            # Get actual answer
            actual_grid = task_data["test"][0]["output"]
            
            # Check if prediction is correct
            is_correct = False
            if predicted_grid:
                is_correct = evaluate_grid_equality(predicted_grid, actual_grid)
            
            # Record result
            if predicted_grid is None:
                results["parse_errors"] += 1
                status = "PARSE_ERROR"
            elif is_correct:
                results["correct"] += 1
                status = "CORRECT"
            else:
                results["incorrect"] += 1
                status = "INCORRECT"
                
            task_time = time.time() - task_start_time
            
            # Store individual task results
            results["task_results"][task_id] = {
                "status": status,
                "time_seconds": task_time,
                "input": task_data["test"][0]["input"],
                "expected": actual_grid,
                "predicted": predicted_grid,
                "raw_model_output": model_output
            }
            
            if verbose:
                print(f"Task {task_id}: {status} (in {task_time:.2f}s)")
                if status == "INCORRECT" and predicted_grid:
                    print(f"  Expected: {actual_grid}")
                    print(f"  Predicted: {predicted_grid}")
                elif status == "PARSE_ERROR":
                    print(f"  Could not parse model output: {model_output[:200]}...")
                print()
                
        except Exception as e:
            results["parse_errors"] += 1
            results["task_results"][task_id] = {
                "status": "ERROR",
                "error": str(e),
                "time_seconds": time.time() - task_start_time
            }
            if verbose:
                print(f"Task {task_id}: ERROR - {str(e)}")
    
    # Calculate overall metrics
    results["total"] = len(task_ids)
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["parse_error_rate"] = results["parse_errors"] / results["total"] if results["total"] > 0 else 0
    results["total_time_seconds"] = time.time() - start_time
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def print_summary(results):
    """Print a summary of the evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total tasks evaluated: {results['total']}")
    print(f"Correct predictions: {results['correct']} ({results['accuracy']:.2%})")
    print(f"Incorrect predictions: {results['incorrect']} ({results['incorrect']/results['total']:.2%})")
    print(f"Parse errors: {results['parse_errors']} ({results['parse_error_rate']:.2%})")
    print(f"Total time: {results['total_time_seconds']:.2f}s (avg: {results['total_time_seconds']/results['total']:.2f}s per task)")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance on ARC tasks")
    parser.add_argument("--data-dir", type=str, default="/home/dt-lindberg/arc25/src/ARC-AGI-2/data/evaluation",
                        help="Directory containing evaluation data")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Number of tasks to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--output-file", type=str, default=None,
                        help="File to save evaluation results")
    parser.add_argument("--model", type=str, default="default",
                        help="Model to use for evaluation (default: mock model for testing)")
    
    args = parser.parse_args()
    
    # Load evaluation data
    print(f"Loading evaluation data from {args.data_dir}...")
    evaluation_data = ARCDataLoader(args.data_dir)
    print(f"Loaded {len(evaluation_data.task_ids)} evaluation tasks")
    
    # Create a mock model if no real model is specified (for testing)
    if args.model == "default":
        print("Using mock model for testing")
        def mock_model(prompt):
            # This mock model just returns the first test input as the output (will be wrong)
            import re
            match = re.search(r"Test Input: (\[\[.*?\]\])", prompt, re.DOTALL)
            if match:
                return f"Based on the pattern, the output should be: {match.group(1)}"
            return "[[0, 0], [0, 0]]"
        model = mock_model
    else:
        # Here you would load your actual model
        # For now, we'll just use the mock model
        model = mock_model
    
    # Run evaluation
    print(f"Starting evaluation on {args.num_tasks if args.num_tasks else 'all'} tasks...")
    results = evaluate_model(
        model, 
        evaluation_data, 
        num_tasks=args.num_tasks, 
        verbose=args.verbose,
        output_file=args.output_file
    )
    
    # Print summary
    print_summary(results)
    
    if args.output_file:
        print(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()