import numpy as np

def test_equality(args):
    # List to store the results of the equality test for each argument tuple
    results = []
    # Check if the rounded dot products are equal
    for A, B in args:
        AiBi = np.dot(A, B)
        BiAi = np.dot(B, A)
        # Append the result to the results list
        results.append(np.array_equal(np.round(AiBi, 2), np.round(BiAi, 2)))
    # Return the list of results
    return results






