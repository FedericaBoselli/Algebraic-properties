# **Python project - Federica Boselli**

## PROJECT 8 - Algebraic Properties

Scientific Programming, A.Y. 2022/2023

Federica Boselli - matriculation number: 995880, person code: 10605062


### **Short description** : 
- Programming language(s): Python
- Usually, given two square matrices A and B it is not true that AB = BA. However, if B=cA, where c is a scalar, then AB=BA.


### **Expected outcome** : 
Implement an algorithm for testing experimentally such hypotheses. It is required to:
- Take as input a integer N and a scalar c
- Generate 10 random matrices N*N: A1, A2, ..., A10
- Generate 10 matrices as: B1=cA1, B2=cA2, ..., B10=cA10
- Test the equality that AiBi = BiAi for i = 1, 2, 3, ..., 10
- Considering a number of threads num_processes > 10, design a second version of the algorithm that uses multiprocessing to speedup the execution.


### **Important notes** : 
The script allows users to customize the matrix size N, numeric scalar c, and number of processes (the latter is specific for the parallel algorithm) via the command line. This flexibility facilitates integration into automated pipelines. If no user-specified parameters are provided, the script will run with default values. The program executes and presents the analysis result for both sequential and parallel algorithm: True if the algebraic property is verified, and False otherwise. Additionally, by specifying the optional parameter "create_report = True", users can generate a report in PDF format. The report includes four plots comparing execution times of the sequential algorithm, parallel implementation, and both methods displayed in the same graph, along with a summary table showcasing the parameters used and the corresponding execution times.


### **Final comments** :
In this project, I implemented two algorithms (sequential and parallel) to experimentally test the hypothesis: AiBi = BiAi for i = 1, 2, 3, ..., 10, given B = cA. Both algorithms consistently verified the hypothesis, yielding True results for all cases, including those used to generate the final PDF report. The plots provide insights into how the execution times of the algorithms change with increasing matrix size N. Initially, the sequential algorithm outperforms the parallel algorithm for small or relatively small N values. However, around N=500, the behavior shifts, and the parallel algorithm becomes faster.

The slower execution time of the parallel algorithm for small problem sizes can be attributed to the overhead of process creation and management, which may outweigh the benefits of parallelism. The cost of synchronization and coordination among processes becomes significant compared to the actual computation, resulting in the sequential algorithm being faster. However, as the matrix size increases, the multiprocessing implementation starts to demonstrate its advantages.

Regarding the parallel algorithm, there is no significant difference observed with varying numbers of processes. Generally, a lower number of threads tends to perform better than a higher number. This suggests that the problem may not be sufficiently large or complex to fully exploit parallelization. When the problem size is small or the computations are not computationally intensive, increasing the number of processes beyond a certain point may not lead to significant performance improvements.