import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to the JSON files
RESULTS_DIR = "/home/ramji.purwar/DecompGTI/DecompGTI/evaluation/results/qwen7b_v3"
FILES = {
    "Mini": os.path.join(RESULTS_DIR, "mini_results.json"),
    "Small": os.path.join(RESULTS_DIR, "small_results.json"),
    "Medium": os.path.join(RESULTS_DIR, "medium_results.json")
}

sizes = list(FILES.keys())
overall_success = []
tasks_data = {}

# Load data
for size in sizes:
    file_path = FILES[size]
    with open(file_path, "r") as f:
        data = json.load(f)
        
    overall_success.append(data["metrics"]["task_success_rate"])
    
    for task_name, task_metrics in data["metrics"]["by_task"].items():
        if task_name not in tasks_data:
            tasks_data[task_name] = []
        tasks_data[task_name].append(task_metrics["task_success"])

# Define modern aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 1. Plot Overall Accuracy
plt.figure(figsize=(8, 6))
bars = plt.bar(sizes, overall_success, color=colors, width=0.5)
plt.title("DecompGTI + Qwen2.5-7B: Overall Graph Reasoning Accuracy by Graph Size", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Graph Scale", fontsize=12)
plt.ylabel("Task Success Rate (%)", fontsize=12)
plt.ylim(0, 100)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.savefig(os.path.join(RESULTS_DIR, "overall_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Plot Per-Task Accuracy Grouped Bar Chart
x = np.arange(len(tasks_data.keys()))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

for i, size in enumerate(sizes):
    # Extract success rates for this size across all tasks
    rates = [tasks_data[task][i] if i < len(tasks_data[task]) else 0 for task in tasks_data.keys()]
    offset = width * i - width
    bars = ax.bar(x + offset, rates, width, label=size, color=colors[i])
    
    # Optional: Add tiny labels on these bars too
    # for bar in bars:
    #     yval = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{int(yval)}', ha='center', va='bottom', fontsize=8)

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Task Success Breakdown by Graph Algorithm and Graph Size', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
formatted_labels = [task.replace('_', ' ').title() for task in tasks_data.keys()]
ax.set_xticklabels(formatted_labels, rotation=45, ha="right", fontsize=11)
ax.legend(title='Dataset Size', fontsize=11, title_fontsize=12)
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "task_breakdown.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Graphs successfully generated and saved matching 'mini', 'small', and 'medium' datasets!")
