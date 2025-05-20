import os
import json
# from typing import Dict, List, Any


class ARCDataLoader:
    """
    A class to load and store ARC tasks from JSON files.
    The data is accessible as a dictionary with the JSON filename as the key.
    
    This class is designed to load one type of data (training or evaluation) at a time.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the ARCDataLoader with one data directory.
        
        Args:
            data_dir: Directory containing task JSON files
        """
        self.data = {}
        
        if data_dir and os.path.isdir(data_dir):
            self.load_data(data_dir)
    
    def load_data(self, directory: str) -> None:
        """
        Load all JSON files from a directory into the data dictionary.
        
        Args:
            directory: Directory containing JSON task files
        """
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r') as f:
                        # Store using filename without extension as the key
                        key = os.path.splitext(filename)[0]
                        self.data[key] = json.load(f)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    def get_task(self, task_id: str) -> dict:
        """
        Get a specific task by ID.
        
        Args:
            task_id: The task ID (filename without extension)
            
        Returns:
            The task data or None if not found
        """
        return self.data.get(task_id)
    
    def __getitem__(self, key):
        """
        Allow dictionary-like access to tasks.
        Example: loader['task_id']
        """
        return self.data.get(key)
        
    @property
    def task_ids(self) -> list[str]:
        """
        Get a list of all task IDs.
        """
        return list(self.data.keys())
        
    def __len__(self):
        """
        Return the number of tasks in the loader.
        """
        return len(self.data)
    
    def __iter__(self):
        """
        Allow iteration over task IDs.
        """
        return iter(self.data)


# Example usage:
if __name__ == "__main__":
    # Assuming the paths based on the workspace structure
    training_dir = "/home/dt-lindberg/arc25/src/ARC-AGI-2/data/training"
    evaluation_dir = "/home/dt-lindberg/arc25/src/ARC-AGI-2/data/evaluation"
    
    # Load the training and evaluation data separately
    training_data = ARCDataLoader(training_dir)
    evaluation_data = ARCDataLoader(evaluation_dir)
    
    # Print some stats
    print(f"Loaded {len(training_data.task_ids)} training tasks")
    print(f"Loaded {len(evaluation_data.task_ids)} evaluation tasks")
    
    # Example: access a specific task
    example_task_id = training_data.task_ids[0] if training_data.task_ids else None
    if example_task_id:
        print(f"\nExample task {example_task_id}:")
        task_data = training_data.get_task(example_task_id)
        print(f"- Training examples: {len(task_data['train'])}")
        print(f"- Test examples: {len(task_data['test'])}")
        
        # Demonstrate dictionary-like access
        print("\nSame data using dictionary access:")
        print(f"- Training examples: {len(training_data[example_task_id]['train'])}")
        print(f"- Test examples: {len(training_data[example_task_id]['test'])}")
