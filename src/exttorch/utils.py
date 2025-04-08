# Holy, holy, holy is the LORD of host, the whole earth is full of his glory.

# Importing necessary modules
import time


class ProgressBar:
    def __init__(
        self, length=40, show_val_metrics=False, verbose=None
    ):
        self._total = 0
        self.length = length
        self.current = 0
        self.start_time = time.time()
        self.progress_color = "\033[92m"
        self.reset_color = "\033[0m"
        self.show_val_metrics = show_val_metrics
        self.suffix = ""
        self.verbose = verbose
        
    @property
    def total(self):
        return self._total
    
    @total.setter
    def total(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Total must be a positive integer.")
        self._total = value

    def update(self, current, metric=None):

        if metric is not None:
            self.suffix = " - ".join([f"{key}: {value:.4f}" for key, value in metric])
            self.suffix = f" - {self.suffix}"

        seconds = int(time.time() - self.start_time)
        milliseconds = int((time.time() - self.start_time) * 1000) % 1000

        self.current = current
        filled_length = int(self.length * self.current // self.total)
        bar = f"{self.progress_color}━{self.reset_color}" * filled_length + "━" * (
            self.length - filled_length
        )
        print(
            f"\r{current}/{self.total} {bar} {seconds}s {milliseconds}ms/step {self.suffix}",
            end="\r",
        )
        if self.show_val_metrics:
            if self.current - 1 == self.total:
                print()  # New line after completion
        else:
            if self.current == self.total:
                # Rest the start time for the next progress bar
                self.start_time = time.time()
                
                print()

        time.sleep(0.1)  # Simulate some processing time

    def last_update(self, metric=None):

        if self.show_val_metrics == False:
            raise ValueError("show_validation must be True to use last_update")

        if metric is not None:
            suffix = " - ".join([f"{key}: {value:.4f}" for key, value in metric])
            suffix = f"{self.suffix} - {suffix}"

        seconds = int(time.time() - self.start_time)
        milliseconds = int((time.time() - self.start_time) * 1000) % 1000

        filled_length = int(self.length * self.current // self.total)
        bar = f"{self.progress_color}━{self.reset_color}" * filled_length + "━" * (
            self.length - filled_length
        )
        print(
            f"\r{self.current}/{self.total} {bar} {seconds}s {milliseconds}ms/step {suffix}",
            end="\r",
        )
        
        # Rest the start time for the next progress bar
        self.start_time = time.time()
        print()


# progress = ProgressBar(show_val_metrics=True)
# for i in range(50):
#     loss = i * 0.01
#     accuracy = i * 0.02
#     # if i < 98:
#     #     # print()
#     progress.total = 50
#     progress.update(i + 1, metric=[("loss", loss), ("accuracy", accuracy)])
#     # progress.update(i + 1, metric=[("loss", loss), ("accuracy", accuracy)])

# progress.last_update(metric=[("val_loss", 0.877), ("val_accuracy", 0.123)])
# print("Final update")
