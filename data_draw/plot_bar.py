import matplotlib.pyplot as plt
import numpy as np

# Data processing
data = {
    0.01: {
        "Train": [74, 13],
        "Model Adjustment after Testing": [23, 8, 49, 37, 24],
        "Test": [9, 14]
    },
    0.005: {
        "Train": [66, 115],
        "Model Adjustment after Testing": [23],
        "Test": [9, 12]
    },
    0.001: {
        "Train": [74, 151],
        "Model Adjustment after Testing": [0],
        "Test": [25, 12]
    },
    0.0005: {
        "Train": [233, 188],
        "Model Adjustment after Testing": [39],
        "Test": [15, 36]
    }
    # 0.0001: {
    #     "Train": [1011, 76, 1468 ,439 ,624, 82],
    #     "Model Adjustment after Testing": [0],
    #     "Test": [149, 45]
    # }
}

# Calculate total time
total_times = {}
for lr, times in data.items():
    total_times[lr] = {k: sum(v) for k, v in times.items()}

# Set common scientific colors
colors = {
    "Train": "#F1B656",   # Blue
    "Model Adjustment after Testing": "#397FC7",  # Orange
    "Test": "#040676"  # Green
}

# Use Times New Roman font and increase font size
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,            # Increase the base font size
    'axes.titlesize': 16,       # Increase the title font size
    'axes.labelsize': 14,       # Increase the x and y label font size
    'xtick.labelsize': 12,      # Increase the x tick label font size
    'ytick.labelsize': 12,      # Increase the y tick label font size
    'legend.fontsize': 12,      # Increase the legend font size
})

# Plotting the bar chart
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(total_times))

# Stacked bar chart
train_times = [total_times[lr]["Train"] for lr in total_times]
test_times = [total_times[lr]["Model Adjustment after Testing"] for lr in total_times]
adjust_times = [total_times[lr]["Test"] for lr in total_times]

bars1 = ax.bar(index, train_times, bar_width, label='Train', color=colors["Train"])
bars2 = ax.bar(index, test_times, bar_width, bottom=train_times, label='Model Adjustment after Testing', color=colors["Model Adjustment after Testing"])
bars3 = ax.bar(index, adjust_times, bar_width, bottom=np.array(train_times)+np.array(test_times), label='Test', color=colors["Test"])

# Add labels
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Time(S)')
ax.set_title('Training and Testing Time for Different Learning Rates')
ax.set_xticks(index)
ax.set_xticklabels(total_times.keys())
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
