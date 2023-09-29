import tqdm

class ProgressLogger:
    """class to show progress bar"""
    def __init__(self, total_steps, description="Logging..."):
        self.total_steps = total_steps
        self.description = description

        self.pbar = tqdm.tqdm(total=total_steps, desc=description)

    def update(self, incremental=1):
        self.pbar.update(incremental)

    def close(self):
        self.pbar.close()