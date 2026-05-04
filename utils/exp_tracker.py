import os
import json
import time
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Must be set before importing plt
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union


class LocalTracker:
    """
    Enhanced local experiment tracker:
    1. Records custom variables and auto-plots trends
    2. Can reload history from JSON and re-plot
    """

    def __init__(self, log_dir: str = "logs", auto_save: bool = True):
        """
        Args:
            log_dir: log output directory
            auto_save: auto-save on every record
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.auto_save = auto_save
        self.data: Dict[str, List[float]] = {}
        self.history: List[Dict[str, float]] = []

    def track(self, step: int, **kwargs: float):
        """Record a set of variables."""
        record = {"step": step, **kwargs}
        self.history.append(record)
        for key, value in kwargs.items():
            self.data.setdefault(key, []).append(value)
        if self.auto_save:
            self.save()

    def save(self, format: str = "json"):
        """Save data to file."""
        if format == "json":
            with open(f"{self.log_dir}/tracker_log.json", "w") as f:
                json.dump({"data": self.data, "history": self.history}, f, indent=2)
        else:
            pd.DataFrame(self.history).to_csv(f"{self.log_dir}/tracker_log.csv", index=False)

    def load(self, file_path: str) -> bool:
        """
        Load history from a JSON file.
        Args:
            file_path: JSON file path
        Returns:
            bool: whether load succeeded
        """
        try:
            with open(file_path, 'r') as f:
                loaded = json.load(f)
                self.data = loaded.get("data", {})
                self.history = loaded.get("history", [])
            return True
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    def plot(
            self,
            variables: Optional[List[str]] = None,
            output: str = "plot.png",
            figsize: list = [10, 6],
            style: str = "seaborn-v0_8",
            title: Optional[str] = None
    ):
        """
        Plot trends (from in-memory or loaded history).
        Args:
            title: optional overall title
            Other args: same as above
        """
        plt.style.use(style)
        df = pd.DataFrame(self.history)

        if df.empty:
            print("No data to plot!")
            return

        variables = variables or [col for col in df.columns if col != "step"]
        n_vars = len(variables)
        figsize[1] = figsize[1] * n_vars
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, var in enumerate(variables):
            axes[i].plot(df["step"], df[var], label=var, marker="o")
            axes[i].set_title(f"Trend of {var}")
            axes[i].legend()

        if title:
            plt.suptitle(title)

        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/{output}")
        plt.close()
        print(f"Plot saved to {self.log_dir}/{output}")

    def __str__(self):
        return f"LocalTracker(tracking {len(self.data)} variables: {list(self.data.keys())})"

class MainTracker:
    def __init__(self):
        # Storage for all records
        self.records = {}
        self.time_records = {}

    # Record start time
    def time_begin(self, var):
        if var not in self.time_records:
            self.time_records[var] = {}
        self.time_records[var]['start'] = time.time()

    # Record end time and compute duration
    def time_end(self, var):
        if var in self.time_records and 'start' in self.time_records[var]:
            end_time = time.time()
            start_time = self.time_records[var]['start']
            # Compute duration in seconds
            self.time_records[var]['duration'] = end_time - start_time
            self.records[var] = end_time - start_time
        else:
            print(f"Error: No start time recorded for {var}")

    # Merge external dict into this tracker
    def track(self, new_dict):
        if isinstance(new_dict, dict):
            self.records.update(new_dict)
        else:
            print("Error: Provided argument is not a dictionary")

    # Save records as JSON
    def save(self, path):
        try:
            with open(path, 'w') as f:
                json.dump(self.records, f, indent=4)
        except Exception as e:
            print(f"Error while saving file: {e}")

    def print_time(self):
        for var in self.time_records:
            print(f"{var}: {self.time_records[var]['duration']} seconds")


# ============== Usage examples ==============
if __name__ == "__main__":
    # Example 1: track and plot
    tracker = LocalTracker(log_dir="./logs")
    for epoch in range(10):
        for j in range(10):
            tracker.track(step=epoch, loss=0.1 * (epoch + 1) * j, accuracy= j + 0.8 + 0.02 * epoch)
    tracker.plot(output="training_curves.png", title="Training Metrics")

    # Example 2: reload from JSON and re-plot
    new_tracker = LocalTracker(log_dir="./logs_reloaded")
    if new_tracker.load("./logs/tracker_log.json"):  # Load saved file
        new_tracker.plot(
            output="reloaded_curves.png",
            variables=["accuracy"],  # Only plot accuracy
            title="Reloaded Accuracy Trend"
        )
