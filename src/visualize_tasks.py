import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from arc_data_loader import ARCDataLoader


def visualize_task(task_data, task_id=None, save_path=None):
    """
    Visualize an ARC task with its train and test examples.
    
    Args:
        task_data: Task data containing 'train' and 'test' examples
        task_id: Task ID for the plot title (optional)
        save_path: Path to save the visualization (optional)
    """
    train_examples = task_data['train']
    test_examples = task_data['test']
    
    # Create a figure with dynamic size based on number of examples
    n_train = len(train_examples)
    n_test = len(test_examples)
    
    # Calculate layout
    fig_width = 12
    fig_height = 1.5 * (n_train + n_test)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create a title for the entire figure
    if task_id:
        fig.suptitle(f"ARC Task: {task_id}", fontsize=16)
    
    # Create a grid layout
    gs = gridspec.GridSpec(n_train + n_test, 2, width_ratios=[1, 1])
    
    # Define a colormap that matches ARC colors
    # ARC uses 10 colors (0-9)
    cmap = colors.ListedColormap([
        'white',        # 0
        'blue',         # 1
        'red',          # 2
        'green',        # 3
        'yellow',       # 4
        'gray',         # 5
        'purple',       # 6
        'pink',         # 7
        'orange',       # 8
        'cyan'          # 9
    ])
    
    # Plot training examples
    for i, example in enumerate(train_examples):
        # Input
        ax_in = plt.subplot(gs[i, 0])
        input_grid = np.array(example['input'])
        ax_in.imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
        ax_in.set_title(f"Train {i+1} Input")
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        
        # Add grid lines
        for x in range(input_grid.shape[1] + 1):
            ax_in.axvline(x - 0.5, color='black', linewidth=0.5)
        for y in range(input_grid.shape[0] + 1):
            ax_in.axhline(y - 0.5, color='black', linewidth=0.5)
            
        # Output
        ax_out = plt.subplot(gs[i, 1])
        output_grid = np.array(example['output'])
        ax_out.imshow(output_grid, cmap=cmap, vmin=0, vmax=9)
        ax_out.set_title(f"Train {i+1} Output")
        ax_out.set_xticks([])
        ax_out.set_yticks([])
        
        # Add grid lines
        for x in range(output_grid.shape[1] + 1):
            ax_out.axvline(x - 0.5, color='black', linewidth=0.5)
        for y in range(output_grid.shape[0] + 1):
            ax_out.axhline(y - 0.5, color='black', linewidth=0.5)
    
    # Plot test examples
    for i, example in enumerate(test_examples):
        # Input
        ax_in = plt.subplot(gs[n_train + i, 0])
        input_grid = np.array(example['input'])
        ax_in.imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
        ax_in.set_title(f"Test {i+1} Input")
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        
        # Add grid lines
        for x in range(input_grid.shape[1] + 1):
            ax_in.axvline(x - 0.5, color='black', linewidth=0.5)
        for y in range(input_grid.shape[0] + 1):
            ax_in.axhline(y - 0.5, color='black', linewidth=0.5)
            
        # Output
        ax_out = plt.subplot(gs[n_train + i, 1])
        output_grid = np.array(example['output'])
        ax_out.imshow(output_grid, cmap=cmap, vmin=0, vmax=9)
        ax_out.set_title(f"Test {i+1} Output")
        ax_out.set_xticks([])
        ax_out.set_yticks([])
        
        # Add grid lines
        for x in range(output_grid.shape[1] + 1):
            ax_out.axvline(x - 0.5, color='black', linewidth=0.5)
        for y in range(output_grid.shape[0] + 1):
            ax_out.axhline(y - 0.5, color='black', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def visualize_task_by_id(task_id, data_loader, save_dir=None):
    """
    Visualize a specific task by its ID.
    
    Args:
        task_id: Task ID to visualize
        data_loader: ARCDataLoader instance containing the task
        save_dir: Directory to save the visualization (optional)
    
    Returns:
        True if the task was found and visualized, False otherwise
    """
    task_data = data_loader.get_task(task_id)
    
    if task_data is None:
        print(f"Task {task_id} not found in the data loader.")
        return False
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{task_id}.png")
    
    visualize_task(task_data, task_id, save_path)
    return True


def visualize_random_tasks(data_loader, n=5, save_dir=None):
    """
    Visualize n random tasks from the data loader.
    
    Args:
        data_loader: ARCDataLoader instance
        n: Number of random tasks to visualize
        save_dir: Directory to save the visualizations (optional)
    """
    import random
    
    task_ids = data_loader.task_ids
    if not task_ids:
        print("No tasks available in the data loader.")
        return
    
    n = min(n, len(task_ids))
    selected_ids = random.sample(task_ids, n)
    
    for task_id in selected_ids:
        visualize_task_by_id(task_id, data_loader, save_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ARC tasks")
    parser.add_argument('--dataset', choices=['training', 'evaluation'], default='training',
                      help='Which dataset to use (training or evaluation)')
    parser.add_argument('--task', type=str, help='Specific task ID to visualize')
    parser.add_argument('--random', type=int, help='Number of random tasks to visualize')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Base directories
    base_dir = "/home/dt-lindberg/arc25/src/ARC-AGI-2/data"
    training_dir = os.path.join(base_dir, "training")
    evaluation_dir = os.path.join(base_dir, "evaluation")
    
    # Determine which dataset to load
    data_dir = training_dir if args.dataset == 'training' else evaluation_dir
    data_loader = ARCDataLoader(data_dir)
    
    print(f"Loaded {len(data_loader.task_ids)} tasks from {args.dataset} dataset")
    
    # Visualize specific task if provided
    if args.task:
        found = visualize_task_by_id(args.task, data_loader, args.save_dir)
        if not found:
            print(f"Task '{args.task}' not found in {args.dataset} dataset.")
            sys.exit(1)
    # Otherwise visualize random tasks if requested
    elif args.random:
        visualize_random_tasks(data_loader, args.random, args.save_dir)
    # Default: show one random task
    else:
        visualize_random_tasks(data_loader, 1, args.save_dir)