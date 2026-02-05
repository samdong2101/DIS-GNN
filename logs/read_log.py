#!/usr/bin/env python3

import os
import argparse

def read_logs(input_dir):
    input_dir = os.path.abspath(input_dir)

    for root, dirs, files in os.walk(input_dir):
        if "log.txt" in files:
            log_path = os.path.join(root, "log.txt")
            subdir = os.path.relpath(root, input_dir)

            print(f"\n=== Subdirectory: {subdir} ===")
            print(f"--- File: {log_path} ---")

            try:
                with open(log_path, "r") as f:
                    print(f.read())
            except Exception as e:
                print(f"Could not read {log_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively read and print all log.txt files in a directory"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"{args.input_dir} is not a valid directory")

    read_logs(args.input_dir)

if __name__ == "__main__":
    main()

