import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, save_path):
        """
        save_path: directory or file prefix where plots will be saved
        """
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def plot_losses(self, train_values, val_values, title="Training vs Validation", ylabel="Value", desc = "losses"):
        """
        train_values: list of training losses or accuracies
        val_values:   list of validation losses or accuracies
        """
        plt.figure(figsize=(6,4))
        plt.plot(train_values, label="Train")
        plt.plot(val_values, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        # Save the plot (e.g., "<save_path>_plot.png")
        outfile = f"{self.save_path}/{desc}.png"
        plt.savefig(outfile)
        plt.close()

        return outfile

    def plot_accuracies(self, train_values, val_values, title="Training vs Validation", ylabel="Value",desc = "accuracies"):
        """
        train_values: list of training losses or accuracies
        val_values:   list of validation losses or accuracies
        """
        plt.figure(figsize=(6,4))
        plt.plot(train_values, label="Train")
        plt.plot(val_values, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        # Save the plot (e.g., "<save_path>_plot.png")
        outfile = f"{self.save_path}/{desc}.png"
        plt.savefig(outfile)
        plt.close()

        return outfile
