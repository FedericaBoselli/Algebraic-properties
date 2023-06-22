## IMPORT PACKAGES
import argparse
import numpy as np
import time
import math
from multiprocessing import Pool
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set the backend to Agg for faster plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from utils import test_equality


# COMMAND LINE INPUT PARAMETERS
def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', action="store", type=int, default=10, help="A positive integer for matrix size, used to create N*N square matrices.")
    parser.add_argument('--c', action="store", type=float, default=3.0, help="A numeric scalar, must be a float.")
    parser.add_argument('--num_processes', action="store", type=int, default=12, help="Number of threads to use for multiprocessing, must be an integer > 10.")
    parser.add_argument('--create_report', action="store",type=bool, default=False, help="Whether to create a report for the comparison of the execution times between the two algorithms (sequential and parallel).")

    args = parser.parse_args()

    # Args checks
    if args.N <= 0:
        raise argparse.ArgumentTypeError("--N is the matrix size, so it must be a positive integer!")
    
    if args.num_processes <= 10:
        raise argparse.ArgumentTypeError("--num_processes must be an integer greater than 10!")

    return args


# SEQUENTIAL ALGORITHM
def test_property_sequential(N, c):

    # Matrix generation: generate 10 random matrices N*N (A1, A2, A3, ..., A10)
    A = [np.round(np.random.rand(N, N)) for _ in range(10)]

    # Generate 10 matrices as: B1=cA1, B2=cA2, ..., B10=cA10
    B = [c * A[i] for i in range(10)]

    # Create argument list for test_equality function (see utils.py)
    args_list = list(zip(A, B))

    # Test the equality that AiBi = BiAi for i = 1, 2, 3, ..., 10
    results = test_equality(args_list)

    # Return the True or False result 
    return all(results)


# MULTIPROCESSING ALGORITHM
def test_property_parallel(N, c, num_processes):

    # Matrix generation: generate 10 random matrices N*N (A1, A2, A3, ..., A10)
    A = [np.round(np.random.rand(N, N)) for _ in range(10)]

    # Generate 10 matrices as: B1=cA1, B2=cA2, ..., B10=cA10
    B = [c * A[i] for i in range(10)]

    # Create a pool of worker processes with the specified number of processes
    with Pool(processes=num_processes) as pool:
        # Determine the chunk size based on the total number of argument tuples and number of processes
        total_tuples = len(A) * len(B)
        
        # In chunk_size variable definition the max function to ensure that the chunk size is at least 1, even if the division result is less than 1. 
        # This ensures that each process receives at least one chunk of work to process. 
        # (it guarantees that each process will have work to do, and so no process remains unused)
        chunk_size = max(math.ceil(total_tuples / num_processes), 1)

        # Divide the list of argument tuples into equal-sized chunks
        chunks = [list(zip(A[i:i + chunk_size], B[i:i + chunk_size])) for i in range(0, total_tuples, chunk_size)]

        # Apply the test_equality function to each chunk of argument tuples in parallel
        results = pool.map(test_equality, chunks)

    # Flatten the list of results
    results = [result for sublist in results for result in sublist]

    # Check if the equality property holds for all matrices and get the result (True/False)
    return all(results)


def main():

    # Command line inputs inputs
    args = process_arguments()

#--------------------------------------------------------------------

    # Calculate the execution time for the sequential algorithm 
    start_time_sequential = time.process_time()

    result_sequential = test_property_sequential(args.N, args.c)

    end_time_sequential = time.process_time()
    execution_time_sequential = end_time_sequential - start_time_sequential

    # Print the result (True/false and execution time) for the sequential algorithm
    print("Algebraic property verified for sequential algorithm? ", result_sequential, "\nExecution Time:", execution_time_sequential)

#--------------------------------------------------------------------

    # Calculate the execution time for the sequential algorithm 
    start_time_parallel = time.process_time()

    result_parallel = test_property_parallel(args.N, args.c, args.num_processes)

    end_time_parallel = time.process_time()
    execution_time_parallel = end_time_parallel - start_time_parallel

    # Print the result (True/false and execution time) for the parallel algorithm
    print("Algebraic property verified for parallel algorithm? ", result_parallel, "\nExecution Time:", execution_time_parallel) 

 #--------------------------------------------------------------------   

    # GRAPHICAL VISUALIZATION 
    if args.create_report:

        # values for N
        N_values = [100, 300, 500, 700, 900, 1200, 1500, 1700, 1900, 2100]  
        # value for c
        c = 3 
        # values for num_processes 
        num_processes_values = [15, 25, 35] 

        # Create empty lists for storing execution times and results 
        execution_times_sequential = []
        execution_times_parallel = []
        results_list = []

        # Sequential Algorithm
        for N in N_values:
            # Run the sequential algorithm and calculate the execution time
            start_time_sequential = time.process_time()
            result_sequential = test_property_sequential(N, c)
            end_time_sequential = time.process_time()

            # Calculate execution time
            execution_time_sequential = end_time_sequential - start_time_sequential

            # Append results to the list of results
            results_list.append({'N': N, 'c': c, 'num_processes': 0, 'algorithm': 'Sequential', 'execution_time': execution_time_sequential})

            # Append execution time to the list of execution times
            execution_times_sequential.append(execution_time_sequential)

        # Parallel Algorithm
        for N in N_values:
            for num_processes in num_processes_values:
                # Run the parallel algorithm and calculate the execution time
                start_time_parallel = time.process_time()
                result_parallel = test_property_parallel(N, c, num_processes)
                end_time_parallel = time.process_time()

                # Calculate execution time
                execution_time_parallel = end_time_parallel - start_time_parallel

                # Append results to the results list
                results_list.append({'N': N, 'c': c, 'num_processes': num_processes, 'algorithm': 'Parallel', 'execution_time': execution_time_parallel})

                # Append execution time to the list of execution times
                execution_times_parallel.append(execution_time_parallel)

        # Create DataFrame from results list
        results_df = pd.DataFrame(results_list)

        # Create a PDF file to save the plots and DataFrame
        with PdfPages('AlgebraicProperties_Report.pdf') as pdf:

            # Create a figure with 2x2 subplots 
            fig, axs = plt.subplots(2, 2, figsize=(8, 11))  

            # Plot for sequential algorithm
            ax = axs[0, 0]
            sns.lineplot(data=results_df[results_df['algorithm'] == 'Sequential'], x='N', y='execution_time', markers=True, ax=ax)
            ax.set_xlabel("Matrix Size (N)", fontsize=8)
            ax.set_ylabel("Execution Time (Sequential)", fontsize=8)
            ax.set_title("Sequential Algorithm Execution Time", fontsize=10)

            # Plot for parallel algorithm
            ax = axs[0, 1]
            sns.lineplot(data=results_df[results_df['algorithm'] == 'Parallel'], x='N', y='execution_time', hue='num_processes', markers=True, palette='Set2', ax=ax)
            ax.set_xlabel("Matrix Size (N)", fontsize=8)
            ax.set_ylabel("Execution Time (Parallel)", fontsize=8)
            ax.set_title("Parallel Algorithm Execution Time", fontsize=10)

            # Plot for sequential and parallel algorithms - log scale
            ax = axs[1, 0]
            sns.lineplot(data=results_df, x='N', y='execution_time', hue='algorithm', style='num_processes', markers=True, palette='Set2', ax=ax)
            ax.set_yscale('log')
            ax.set_xlabel("Matrix Size (N)", fontsize=8)
            ax.set_ylabel("Execution Time - log scale", fontsize=8)
            ax.set_title("Sequential and Parallel comparison - log scale", fontsize=10)
            ax.legend(fontsize='small')

            # Plot for sequential and parallel algorithms
            ax = axs[1, 1]
            sns.lineplot(data=results_df, x='N', y='execution_time', hue='algorithm', style='num_processes', markers=True, palette='Set2', ax=ax)
            ax.set_xlabel("Matrix Size (N)", fontsize=8)
            ax.set_ylabel("Execution Time", fontsize=8)
            ax.set_title("Sequential and Parallel comparison", fontsize=10)
            ax.legend(fontsize='small')

            fig.suptitle("Comparison of the execution times between Sequental and Parallel algorithms", fontsize=12, fontweight='bold', y=0.99)

            plt.tight_layout()

            # Save the figure 
            pdf.savefig(fig, dpi=300)  
            plt.close()

            # Create a new figure for the DataFrame table
            fig, ax = plt.subplots(figsize=(8, 11), dpi=300) 

            ax.axis('off')  
            table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
            ax.set_title("Summary DataFrame", fontsize=12, fontweight='bold')

            # Save the figure with the DataFrame table to the PDF
            pdf.savefig(fig, dpi=300) 
            plt.close()


if __name__ == "__main__":
    main()