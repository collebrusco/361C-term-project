import csv

def data_to_csv(data, filename='output.csv'):
    # Split the data into lines
    lines = data.strip().split('\n')
    
    # Prepare to collect the entries
    entries = []
    current_size = None  # To track the current input size

    # Iterate through the lines of data
    for line in lines:
        if line.isdigit():
            current_size = line  # Update the current input size when a digit line is found
        else:
            parts = line.split()
            solver_name = parts[0]  # Solver type
            num_dimensions = '1' if '1d' == parts[1] else '2'
            mean_time = parts[3]    # Mean time
            std_dev = parts[5]      # Standard deviation
            entries.append([solver_name, mean_time, std_dev, current_size, num_dimensions])
    
    # Write the entries to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Solver Name', 'Mean', 'Std Dev', 'Input Size', 'Dimensions'])
        # Write all entries
        writer.writerows(entries)
    
    print(f"Data has been written to {filename}")


# Example usage
data = """
8
fftw 1d m 0.20625 s 0.705752
fftw 2d m 1.12187 s 0.163429
seq 1d m 6.97031 s 1.45406
seq 2d m 106.869 s 2.4199
gpu 1d m 113.773 s 113.762
gpu 2d m 98.4984 s 2.86157
16
fftw 1d m 0.379688 s 1.27563
fftw 2d m 5.22969 s 0.171099
seq 1d m 15.4062 s 0.679125
seq 2d m 490.908 s 3.29091
gpu 1d m 107.95 s 26.1772
gpu 2d m 101.447 s 1.48566
32
fftw 1d m 0.982812 s 3.442
fftw 2d m 25.375 s 0.293684
seq 1d m 34.5531 s 0.786799
seq 2d m 2505.59 s 618.203
gpu 1d m 107.364 s 19.5033
gpu 2d m 112.233 s 3.32303
64
fftw 1d m 1.97656 s 0.383545
fftw 2d m 161.656 s 6.19745
seq 1d m 74.425 s 0.681222
seq 2d m 9715.45 s 150.937
gpu 1d m 120.828 s 25.4139
gpu 2d m 220.927 s 10.84
128
fftw 1d m 4.99531 s 4.03239
fftw 2d m 1178.26 s 52.1
seq 1d m 165.592 s 15.6844
seq 2d m 41726.1 s 469.801
gpu 1d m 128.136 s 14.9359
gpu 2d m 554.939 s 6.68958
256
fftw 1d m 8.49844 s 2.20946
fftw 2d m 4341.41 s 119.483
seq 1d m 345.419 s 15.327
seq 2d m 178785 s 1269.73
gpu 1d m 1027.37 s 107.114
gpu 2d m 4218.55 s 4864.22
512
fftw 1d m 18.1594 s 1.68168
fftw 2d m 18961.8 s 243.462
seq 1d m 742.192 s 41.0734
seq 2d m 774465 s 9056.11
gpu 1d m 1839.18 s 303.357
gpu 2d m 9779.36 s 6997.08
1024
fftw 1d m 39.275 s 8.80261
fftw 2d m 89248.8 s 1310.76
seq 1d m 1612.77 s 25.7685
seq 2d m 3352370 s 80772.7
gpu 1d m 3539.49 s 1099.37
gpu 2d m 31910.2 s 1857.69
"""

data_to_csv(data)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_fft_from_csv(filename):
    # Load data from CSV
    data = pd.read_csv(filename)
    
    # Split the data into 1D and 2D
    data_1d = data[data['Dimensions'] == 1]
    data_2d = data[data['Dimensions'] == 2]

    # print(data_1d)
    
    # Function to plot data
    def plot_data(data, title):
        plt.figure(figsize=(16, 12))
        solvers = data['Solver Name'].unique()
        colors = ['Red', 'Green', 'Blue']
        
        for solver, color in zip(solvers, colors):
            # if solver == 'seq':
            #     continue
            subset = data[data['Solver Name'] == solver]
            subset['Mean'] = subset['Mean'] / 1000
            subset['Std Dev'] = subset['Std Dev'] / 1000
            # subset = subset[subset['Input Size'] < 256]
            plt.errorbar(subset['Input Size'], subset['Mean'], yerr=subset['Std Dev'], label=solver,
                         fmt='-o', capsize=2, color=color)
        
        plt.title(f'FFT Execution Times for {title}')
        plt.xlabel('Input Size')
        plt.ylabel('Execution Time (ms)')
        plt.xscale('log', base=2)
        # plt.yscale('log')

        # Create formatter and apply to both axes
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.legend()
        plt.grid(True)
        plt.savefig(title + '.png')

    # Plotting 1D and 2D FFT execution times
    plot_data(data_1d, 'Solver Comparison 1D')
    plot_data(data_2d, 'Solver Comparison 2D')

# Example usage
plot_fft_from_csv('output.csv')


