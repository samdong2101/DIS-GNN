import numpy as np

def random_shuffle(data_dict):
    """
    Shuffle the contents of a dictionary in unison along the first axis.
    
    Args:
        data_dict (dict): Dictionary where each value is a list of samples.
    
    Returns:
        dict: New dictionary with all lists shuffled in the same order.
    """
    # Number of samples (assumes all lists have the same length)
    n = len(next(iter(data_dict.values())))
    
    # Generate a random permutation of indices
    perm = np.random.permutation(n)
    
    # Create a new dictionary with shuffled data
    shuffled_dict = {key: [data_dict[key][i] for i in perm] for key in data_dict}
    
    return shuffled_dict
