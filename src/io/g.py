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
fftw 1d m 0.153125 s 0.347297
fftw 2d m 1.05312 s 0.128657
seq 1d m 6.54843 s 1.58292
seq 2d m 103.05 s 4.05813
gpu 1d m 117.614 s 160.212
gpu_nocpy 1d m 3.43437 s 3.45078
gpu 2d m 95.9 s 2.19858
gpu_nocpy 2d m 3.11562 s 0.706271
16
fftw 1d m 0.29375 s 0.669159
fftw 2d m 5.09063 s 0.0630445
seq 1d m 15.6531 s 0.609551
seq 2d m 484.077 s 5.91787
gpu 1d m 109.161 s 73.1194
gpu_nocpy 1d m 3.22969 s 1.00866
gpu 2d m 99.1359 s 5.91284
gpu_nocpy 2d m 2.98594 s 0.25487
32
fftw 1d m 4.18125 s 27.7954
fftw 2d m 27.4219 s 3.43568
seq 1d m 33.0234 s 0.472276
seq 2d m 2183.25 s 109.628
gpu 1d m 108.467 s 21.5073
gpu_nocpy 1d m 3.13125 s 0.772552
gpu 2d m 115.305 s 7.01417
gpu_nocpy 2d m 3.07656 s 0.229634
64
fftw 1d m 1.96094 s 0.421536
fftw 2d m 172.664 s 62.3333
seq 1d m 73.5016 s 1.49692
seq 2d m 9635.74 s 246.279
gpu 1d m 116.17 s 17.5891
gpu_nocpy 1d m 2.98594 s 0.268018
gpu 2d m 222.409 s 4.71665
gpu_nocpy 2d m 3.15781 s 0.182692
128
fftw 1d m 14.2906 s 69.2947
fftw 2d m 1172.16 s 7.87469
seq 1d m 168.703 s 23.7958
seq 2d m 42999.9 s 2340.18
gpu 1d m 135.18 s 25.0423
gpu_nocpy 1d m 3.28594 s 0.873546
gpu 2d m 560.698 s 14.5978
gpu_nocpy 2d m 3.72656 s 1.24263
256
fftw 1d m 16.6906 s 63.4367
fftw 2d m 4586.3 s 108.721
seq 1d m 364.28 s 8.13551
seq 2d m 189003 s 3884.73
gpu 1d m 1039.46 s 112.074
gpu_nocpy 1d m 3.39375 s 0.585468
gpu 2d m 4374.88 s 5096.11
gpu_nocpy 2d m 4.16562 s 1.52277
512
fftw 1d m 17.9984 s 0.748017
fftw 2d m 19290.7 s 267.742
seq 1d m 752.042 s 26.0675
seq 2d m 782326 s 11413.6
gpu 1d m 1933.31 s 167.946
gpu_nocpy 1d m 3.03906 s 0.289729
gpu 2d m 8542.93 s 7439.95
gpu_nocpy 2d m 4.32344 s 1.66853
1024
fftw 1d m 40.9875 s 5.75273
fftw 2d m 89166.2 s 1774.47
seq 1d m 1593.46 s 18.3961
seq 2d m 3.31477e+06 s 46026.2
gpu 1d m 3872.05 s 1131.88
gpu_nocpy 1d m 3.68438 s 1.18219
gpu 2d m 32157.3 s 2046.94
gpu_nocpy 2d m 7.81719 s 1.9637
"""

data_to_csv(data)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_fft_from_csv_all(filename):
    # Load data from CSV
    data = pd.read_csv(filename)
    
    # Split the data into 1D and 2D
    data_1d = data[data['Dimensions'] == 1]
    data_2d = data[data['Dimensions'] == 2]

    # print(data_1d)
    
    # Function to plot data
    def plot_data(data, title):
        plt.figure(figsize=(8, 6))
        solvers = data['Solver Name'].unique()
        
        for solver in solvers:
            color = ''
            if solver == 'fftw':
                color = 'Blue'
            if solver == 'seq':
                color = 'Green'
            if solver == 'gpu':
                color = 'Red'
            if solver == 'gpu_nocpy':
                color = 'Black'
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
    plot_data(data_1d, 'All Solvers 1D')
    plot_data(data_2d, 'All Solvers 2D')


def plot_fft_from_csv_fc_s(filename):
    # Load data from CSV
    data = pd.read_csv(filename)
    
    # Split the data into 1D and 2D
    data_1d = data[data['Dimensions'] == 1]
    data_2d = data[data['Dimensions'] == 2]

    # print(data_1d)
    
    # Function to plot data
    def plot_data(data, title):
        plt.figure(figsize=(8, 6))
        solvers = data['Solver Name'].unique()
        
        for solver in solvers:
            color = ''
            if solver == 'fftw':
                color = 'Blue'
            if solver == 'seq':
                color = 'Green'
            if solver == 'gpu':
                color = 'Red'
            if solver == 'gpu_nocpy':
                color = 'Black'
            if solver == 'seq':
                continue
            subset = data[data['Solver Name'] == solver]
            subset['Mean'] = subset['Mean'] / 1000
            subset['Std Dev'] = subset['Std Dev'] / 1000
            subset = subset[subset['Input Size'] < 256]
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
    # plot_data(data_1d, 'Solver Comparison 1D')
    # plot_data(data_2d, 'Solver Comparison 2D')
    # plot_data(data_1d, 'FFTW v CUDA 1D')
    # plot_data(data_2d, 'FFTW v CUDA 2D')
    plot_data(data_1d, 'FFTW v CUDA 1D (small inputs)')
    plot_data(data_2d, 'FFTW v CUDA 2D (small inputs)')


def plot_fft_from_csv_fc(filename):
    # Load data from CSV
    data = pd.read_csv(filename)
    
    # Split the data into 1D and 2D
    data_1d = data[data['Dimensions'] == 1]
    data_2d = data[data['Dimensions'] == 2]

    # print(data_1d)
    
    # Function to plot data
    def plot_data(data, title):
        plt.figure(figsize=(8, 6))
        solvers = data['Solver Name'].unique()
        
        for solver in solvers:
            color = ''
            if solver == 'fftw':
                color = 'Blue'
            if solver == 'seq':
                color = 'Green'
            if solver == 'gpu':
                color = 'Red'
            if solver == 'gpu_nocpy':
                color = 'Black'
            if solver == 'seq':
                continue
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
    # plot_data(data_1d, 'Solver Comparison 1D')
    # plot_data(data_2d, 'Solver Comparison 2D')
    plot_data(data_1d, 'FFTW v CUDA 1D')
    plot_data(data_2d, 'FFTW v CUDA 2D')
    # plot_data(data_1d, 'FFTW v CUDA 1D (small inputs)')
    # plot_data(data_2d, 'FFTW v CUDA 2D (small inputs)')


def plot_fft_from_csv_c(filename):
    # Load data from CSV
    data = pd.read_csv(filename)
    
    # Split the data into 1D and 2D
    data_1d = data[data['Dimensions'] == 1]
    data_2d = data[data['Dimensions'] == 2]

    # print(data_1d)
    
    # Function to plot data
    def plot_data(data, title):
        plt.figure(figsize=(8, 6))
        solvers = data['Solver Name'].unique()
        
        for solver in solvers:
            color = ''
            if solver == 'fftw':
                color = 'Blue'
            if solver == 'seq':
                color = 'Green'
            if solver == 'gpu':
                color = 'Red'
            if solver == 'gpu_nocpy':
                color = 'Black'
            if solver != 'gpu_nocpy':
                continue
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
    # plot_data(data_1d, 'Solver Comparison 1D')
    # plot_data(data_2d, 'Solver Comparison 2D')
    plot_data(data_1d, 'CUDA (no memcpy) 1D')
    plot_data(data_2d, 'CUDA (no memcpy) 2D')
    # plot_data(data_1d, 'FFTW v CUDA 1D (small inputs)')
    # plot_data(data_2d, 'FFTW v CUDA 2D (small inputs)')


def plot_fft_from_csv_fcn(filename):
    # Load data from CSV
    data = pd.read_csv(filename)
    
    # Split the data into 1D and 2D
    data_1d = data[data['Dimensions'] == 1]
    data_2d = data[data['Dimensions'] == 2]

    # print(data_1d)
    
    # Function to plot data
    def plot_data(data, title):
        plt.figure(figsize=(8, 6))
        solvers = data['Solver Name'].unique()

        for solver in solvers:
            color = ''
            if solver == 'fftw':
                color = 'Blue'
            if solver == 'seq':
                continue
            if solver == 'gpu':
                continue
            if solver == 'gpu_nocpy':
                color = 'Black'
            subset = data[data['Solver Name'] == solver]
            subset['Mean'] = subset['Mean'] / 1000
            subset['Std Dev'] = subset['Std Dev'] / 1000
            # subset = subset[subset['Input Size'] < 256]
            plt.plot(subset['Input Size'], subset['Mean'], '-o', label=solver,
                         color=color)
        
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
    # plot_data(data_1d, 'Solver Comparison 1D')
    # plot_data(data_2d, 'Solver Comparison 2D')
    # plot_data(data_1d, 'CUDA (no memcpy) 1D')
    # plot_data(data_2d, 'CUDA (no memcpy) 2D')
    plot_data(data_1d, 'FFTW v CUDA (no memcpy) 1D')
    plot_data(data_2d, 'FFTW v CUDA (no memcpy) 2D')



# Example usage
# plot_fft_from_csv_all('output.csv')
# plot_fft_from_csv_fc_s('output.csv')
# plot_fft_from_csv_fc('output.csv')
plot_fft_from_csv_fcn('output.csv')
# plot_fft_from_csv_c('output.csv')


