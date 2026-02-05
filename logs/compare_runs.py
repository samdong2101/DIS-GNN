#!/usr/bin/env python3

import argparse
import os
import numpy as np

ROOT_PATH = "."

def read_floats(filepath):
    """
    Read a text file containing one float per line
    and return a NumPy array of floats.
    """
    with open(filepath, "r") as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

def main():
    parser = argparse.ArgumentParser(description="Subtract two float files and report stats.")
    parser.add_argument("filename1", type=str, help="First filename (relative to logs root)")
    parser.add_argument("filename2", type=str, help="Second filename (relative to logs root)")
    args = parser.parse_args()

    path1 = os.path.join(ROOT_PATH, args.filename1)
    path2 = os.path.join(ROOT_PATH, args.filename2)

    arr1 = read_floats(path1)
    arr2 = read_floats(path2)

    n = min(len(arr1), len(arr2))
    diff = arr1[:n] - arr2[:n]

    last_10 = diff[-10:] if len(diff) >= 10 else diff
    avg_last_10 = np.mean(last_10)

    print(f"Average of last {len(last_10)} differences: {avg_last_10}")

if __name__ == "__main__":
    main()

