"""
Native Python data generation script designed to be fully portable across Windows/Linux.
Bypasses Bash to prevent WSL path resolution issues with Windows Python.
"""
import os
import subprocess
import sys
from pathlib import Path

# Paths based on the script location
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT_BASE = PROJECT_ROOT.parent / "data" 

# Inject GraphInstruct root into PYTHONPATH
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

# Generation specs
tasks = [
    'shortest_path', 'DFS', 'BFS', 'connectivity', 'topological_sort',
    'cycle', 'bipartite', 'MST', 'maximum_flow', 'connected_component'
]
configs = [
    {'tag': 'mini', 'range': '(5,7)'},
    {'tag': 'small', 'range': '(8,15)'},
    {'tag': 'medium', 'range': '(16,25)'}
]
num_sample = '500'

def run_generation():
    print(f"Project Root resolved to: {PROJECT_ROOT}")
    print("Starting native Python data generation pipeline...\n")
    
    for config in configs:
        tag = config['tag']
        n_range = config['range']
        data_root = DATA_ROOT_BASE / tag
        data_root.mkdir(parents=True, exist_ok=True)
        
        print(f"=== Generating {tag} dataset ===")
        for task in tasks:
            print(f" -> Task: {task} {n_range}")
            task_dir = data_root / task
            task_dir.mkdir(parents=True, exist_ok=True)
            raw_csv = task_dir / f"{task}.csv"
            
            # Step 1: Generate Graph
            cmd_gen = [sys.executable, "-m", "GTG.generation", "--task", task, "--file_output", str(raw_csv),
                       "--num_samples", num_sample, "--num_nodes_range", n_range, "--hash_str", tag]
            subprocess.run(cmd_gen, env=env, check=True, stdout=subprocess.DEVNULL)
            
            # Step 2: Int ID representation
            out_int = task_dir / f"{task}-int_id.csv"
            cmd_int = [sys.executable, "-m", "GTG.process_node_id.main", "--id_type", "int_id", 
                       "--file_input", str(raw_csv), "--file_output", str(out_int)]
            subprocess.run(cmd_int, env=env, check=True, stdout=subprocess.DEVNULL)
            
            # Step 3: Letter ID representation
            out_letter = task_dir / f"{task}-letter_id.csv"
            cmd_letter = [sys.executable, "-m", "GTG.process_node_id.main", "--id_type", "letter_id", 
                          "--file_input", str(raw_csv), "--file_output", str(out_letter)]
            subprocess.run(cmd_letter, env=env, check=True, stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    try:
        run_generation()
        print("\nSUCCESS: All datasets successfully generated!")
    except subprocess.CalledProcessError as e:
        print(f"\nFATAL ERROR: A subprocess failed with exit code {e.returncode}")
        print(f"Command run: {' '.join(e.cmd)}")
        sys.exit(1)