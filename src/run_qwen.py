import os
import json
import time
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import torch

# Import our modules
from src.arc_data_loader import ARCDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_task_for_model(task_data):
    """Format an ARC task into a text prompt for the model."""
    prompt = """You are tasked with solving an ARC-AGI reasoning challenge. CRITICALLY IMPORTANT: Your response must ONLY contain a list of matrices representing your solution.

DO NOT include ANY explanations, reasoning, or text after you stop thinking.
DO NOT describe your approach or explain your solution after you stop thinking.
ONLY output the solution in the format [[a,b,c], [d,e,f], [g,h,i]] where the values represent your answer.

For example, if your solution is a 3x3 grid, your entire response should look like:
[[1,2,3],[4,5,6],[7,8,9]]

NOTHING ELSE. No text before or after. ONLY the list of lists with your solution.

Here are pattern examples:
"""
    
    # Add training examples
    for i, example in enumerate(task_data["train"]):
        prompt += f"Training Example {i+1}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    # Add test input
    test_example = task_data["test"][0]
    prompt += f"Now, apply the pattern you've learned to this new input:\n"
    prompt += f"Test Input: {test_example['input']}\n\n"
    prompt += f"Your response should ONLY be a list of lists representing the output grid."
    
    return prompt

def format_retry_prompt(task_data, previous_output):
    """Format a retry prompt when the model's output was not in the correct format."""
    prompt = """You previously gave an incorrect response format for an ARC-AGI reasoning challenge. 
    
Your previous response was:
"""
    prompt += previous_output
    prompt += """

This format is INCORRECT. Your response must ONLY contain a list of matrices representing your solution.
NO explanations or text. ONLY the python-formatted list of lists with your solution.

For example, if your solution is a 3x3 grid, your entire response should be EXACTLY like this:
[[1,2,3],[4,5,6],[7,8,9]]

Let's try again with the same problem:

"""
    # Add training examples
    for i, example in enumerate(task_data["train"]):
        prompt += f"Training Example {i+1}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    # Add test input
    test_example = task_data["test"][0]
    prompt += f"Test Input: {test_example['input']}\n\n"
    prompt += f"ONLY respond with a list of lists representing the output grid. Nothing else."
    
    return prompt

def parse_model_output(output_text):
    """Parse the model's output text to extract the predicted grid."""
    try:
        # Find the grid in the output - typically it's a list of lists
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
    except Exception as e:
        print(f"Error parsing output: {e}")
        return None

def evaluate_grid_equality(predicted, actual):
    """Check if the predicted grid matches the actual grid."""
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

def run_model_with_prompt(model, tokenizer, prompt):
    """Run the model with the given prompt and extract the output."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Conduct text completion
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # Parse thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    return content, thinking_content

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-4B on ARC tasks")
    parser.add_argument("--data-dir", type=str, 
                        default="src/ARC-AGI-2/data/evaluation",
                        help="Directory containing evaluation data")
    parser.add_argument("--num-tasks", type=int, default=10,
                        help="Number of tasks to evaluate (default: 10)")
    parser.add_argument("--output-file", type=str, default="qwen_evaluation_results.json",
                        help="File to save evaluation results")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Maximum number of retries if output format is incorrect")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    args = parser.parse_args()
    
    print("Starting Qwen3-4B evaluation on ARC tasks...")
    
    # Step 1: Load the model
    print("Loading Qwen3-4B model...")
    start_time = datetime.now()
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", 
        device_map="auto"
    )
    print(f"Model loaded in {datetime.now() - start_time}")
    
    # Step 2: Load evaluation data
    print(f"Loading evaluation data from {args.data_dir}...")
    evaluation_data = ARCDataLoader(args.data_dir)
    print(f"Loaded {len(evaluation_data.task_ids)} evaluation tasks")
    
    # Limit the number of tasks if specified
    task_ids = evaluation_data.task_ids
    if args.num_tasks and args.num_tasks < len(task_ids):
        task_ids = task_ids[:args.num_tasks]
        
    print(f"Evaluating {len(task_ids)} tasks...")
    
    # Step 3: Set up results storage
    results = {
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
        "total": 0,
        "task_results": {},
        "metadata": {
            "model": model_name,
            "date": datetime.now().isoformat(),
            "num_tasks": len(task_ids),
            "max_retries": args.max_retries
        }
    }
    
    # Step 4: Evaluate tasks
    total_start_time = time.time()
    for task_id in tqdm(task_ids, desc="Evaluating tasks"):
        task_data = evaluation_data.get_task(task_id)
        actual_grid = task_data["test"][0]["output"]
        task_start_time = time.time()
        
        # Format initial prompt for model
        prompt = format_task_for_model(task_data)
        
        # Try with initial prompt
        content, thinking_content = run_model_with_prompt(model, tokenizer, prompt)
        predicted_grid = parse_model_output(content)
        retry_count = 0
        
        # Retry if needed and if we haven't reached max retries
        while predicted_grid is None and retry_count < args.max_retries:
            retry_count += 1
            if args.verbose:
                print(f"Retry {retry_count} for task {task_id} - previous output format incorrect")
            
            # Format retry prompt
            retry_prompt = format_retry_prompt(task_data, content)
            
            # Run model with retry prompt
            content, retry_thinking = run_model_with_prompt(model, tokenizer, retry_prompt)
            predicted_grid = parse_model_output(content)
            
            # Combine thinking content
            thinking_content += f"\n\n--- RETRY {retry_count} THINKING ---\n\n{retry_thinking}"
        
        # Check if prediction is correct
        is_correct = False
        if predicted_grid is not None:
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
            "model_output": content,
            "retries": retry_count,
            "thinking": thinking_content
        }
        
        if args.verbose:
            print(f"Task {task_id}: {status} (in {task_time:.2f}s, retries: {retry_count})")
            if status == "INCORRECT" and predicted_grid:
                print(f"  Expected: {actual_grid}")
                print(f"  Predicted: {predicted_grid}")
            elif status == "PARSE_ERROR":
                print(f"  Could not parse model output: {content[:200]}...")
            print()
    
    # Calculate overall metrics
    results["total"] = len(task_ids)
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["parse_error_rate"] = results["parse_errors"] / results["total"] if results["total"] > 0 else 0
    results["total_time_seconds"] = time.time() - total_start_time
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total tasks evaluated: {results['total']}")
    print(f"Correct predictions: {results['correct']} ({results['accuracy']:.2%})")
    print(f"Incorrect predictions: {results['incorrect']} ({results['incorrect']/results['total']:.2%})")
    print(f"Parse errors: {results['parse_errors']} ({results['parse_error_rate']:.2%})")
    print(f"Total time: {results['total_time_seconds']:.2f}s")
    print(f"Avg time per task: {results['total_time_seconds']/results['total']:.2f}s")
    print("="*50)
    
    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.output_file}")
    
if __name__ == "__main__":
    main()