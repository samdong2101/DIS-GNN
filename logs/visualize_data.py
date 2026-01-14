import matplotlib.pyplot as plt
import os
import argparse
def plotter(rootdir, train_filename, val_filename, destination, desc='accuracy', ext='png'):
    """
    Reads a text file containing training and validation metrics per epoch,
    plots training vs validation curves, and saves the figure.
    
    Assumes the file has two columns: 
        first column = training metric
        second column = validation metric
    Each row corresponds to an epoch.
    """
    # Load data
    train_file = os.path.join(rootdir,train_filename)
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip():
                # Convert each line to floats
                values = [float(x) for x in line.strip().split()]
                train_data.append(values)

    val_file = os.path.join(rootdir,val_filename)
    val_data = []
    with open(val_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip():
                # Convert each line to floats
                values = [float(x) for x in line.strip().split()]
                val_data.append(values)


    # Convert to separate lists
    train = train_data
    val = val_data
    epochs = list(range(1, len(train)+1))

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train, label='Training')
    plt.plot(epochs, val, label='Validation')
    plt.title(f'Training/Validation {desc}')
    plt.xlabel('Epochs')
    plt.ylabel(desc.capitalize())
    plt.legend()
    plt.grid(True)

    # Ensure destination exists
    os.makedirs(destination, exist_ok=True)

    # Save figure
    save_path = os.path.join(destination, f'training_validation_{desc}.{ext}')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--root_dir",
            type=str,
            required=False,
            help="Directory containing log files"
    )
    cwd = os.getcwd()
    for i in os.listdir(cwd):
        try:
            root_dir = os.path.join(cwd,i)
            if os.path.isdir(root_dir):
                try:
                    plotter(root_dir,'training_accuracy.txt', 'validation_accuracy.txt', root_dir, desc = 'accuracy', ext = 'png')
                except:
                    pass
                plotter(root_dir,'training_loss.txt', 'validation_loss.txt', root_dir, desc = 'loss', ext = 'png')
            else:
                pass
        except Exception as e:
            print(f'error in {i} with error {e}')
            pass


if __name__ == "__main__":
    main()
