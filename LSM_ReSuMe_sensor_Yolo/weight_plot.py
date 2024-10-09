import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_weights_from_csv(file_path):
    timestamps = []
    weights_over_time = []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)
        for row in csvreader:
            timestamps.append(float(row[0]))
            weights = np.array([float(value) for value in row[1:]])
            weights_over_time.append(weights.reshape(3, 25 * 40))

    return timestamps, weights_over_time

def plot_all_weights(timestamps, weights_over_time, colors):
    plt.figure(figsize=(10, 6))

    for i in range(3):  # 3 output neurons
        for j in range(25 * 40):  # 1000 synapses
            neuron_weight_abs = [weight[i, j] for weight in weights_over_time]
            plt.plot(timestamps, neuron_weight_abs, color=colors[j], linewidth=0.5, alpha=0.8)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Weight Value')
    plt.title('Weight Changes Over Time for All Neurons and Synapses')

def plot_individual_neuron_weights(timestamps, weights_over_time, neuron_index, colors):
    plt.figure(figsize=(10, 6))

    for j in range(25 * 40):  # 1000 synapses
        neuron_weight_abs = [weight[neuron_index, j] for weight in weights_over_time]
        plt.plot(timestamps, neuron_weight_abs, color=colors[j], linewidth=0.5, alpha=0.8)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Weight Value')
    plt.title(f'Weight Changes Over Time for Neuron {neuron_index}')

if __name__ == '__main__':
    file_path = "weights_over_time.csv"
    timestamps, weights_over_time = read_weights_from_csv(file_path)

    # Use a colormap to get unique colors
    colors = cm.hsv(np.linspace(0, 1, 25 * 40))

    # Plot all weights
    plot_all_weights(timestamps, weights_over_time, colors)

    # Plot weights for individual neurons
    for neuron_index in range(3):
        plot_individual_neuron_weights(timestamps, weights_over_time, neuron_index, colors)

    # Show all plots together
    plt.show()
