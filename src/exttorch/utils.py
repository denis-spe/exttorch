# Holy, holy, holy is the LORD of host, the whole earth is full of his glory.

# Importing necessary modules
import time


class ProgressBar:
    def __init__(
        self,
        bar_width=40,
        show_val_metrics=False,
        verbose=None,
        show_diff_color=False,
        empty_color="\033[90m",
        progress_color="\033[92m",
        style="default",
        show_check_mark=True,
        show_suffix=True,
    ):
        self._total = 0
        self.length = bar_width
        self.current = 0
        self.start_time = time.time()
        self.progress_color = progress_color
        self.progress_empty_color = empty_color  # Dark Gray
        self.reset_color = "\033[0m"
        self.show_val_metrics = show_val_metrics
        self.suffix = ""
        self.style = style
        self.last_update_time = self.start_time
        self.verbose = verbose
        self.check_mark = show_check_mark
        self.show_diff_color = show_diff_color
        self.show_suffix = show_suffix

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Total must be a positive integer.")
        self._total = value

    def __progress_bar(self, current, bar, elapsed_time, step_time_formatted, percent):
        if self.check_mark:
            done = (
                f" {self.progress_color}✔️{self.reset_color} " if percent == 100 else " "
            )
        else:
            done = " "

        if self.verbose == "verbose":
            print(
                f"\r{current}/{self.total}{done}{bar} {elapsed_time}s {step_time_formatted} {self.suffix}",
                end="\r",
                flush=True
            )
        elif self.verbose == "silent":
            print(
                f"\r{current}/{self.total}{done}{elapsed_time}s {step_time_formatted} {self.suffix}",
                end="\r",
                flush=True
            )
        elif self.verbose == "silent_verbose":
            print(
                f"\r{current}/{self.total}{done}{bar} {elapsed_time}s {step_time_formatted}",
                end="\r",
                flush=True,
            )
        elif self.verbose == "silent_verbose_suffix":
            print(
                f"\r{current}/{self.total}{done}{self.suffix}",
                end="\r",
                flush=True,
            )
        elif self.verbose == "silent_epoch":
            print(
                f"\r{current}/{self.total}{done}{bar} {elapsed_time}s {step_time_formatted} {self.suffix}",
                end="\r",
                flush=True,
            )
        elif self.verbose == "silent_epoch_suffix":
            print(
                f"\r{current}/{self.total}{done}{self.suffix}",
                end="\r",
                flush=True,
            )
        elif self.verbose == None:
            pass
        else:
            raise ValueError(
                "Invalid verbose option. Choose from ['verbose', 'silent', 'silent_verbose', 'silent_verbose_suffix', 'silent_epoch', 'silent_epoch_suffix']"
            )
    def __bar(self, filled_length, percent):
        match self.style:
            case "default":
                bar = (
                    f"{self.progress_color}━{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )

            case "--":
                bar = (
                    f"{self.progress_color}-{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "==":
                bar = (
                    f"{self.progress_color}={self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "-=":
                bar = (
                    f"{self.progress_color}{'-' if percent <= 50 else '='}{self.reset_color}"
                    * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "airplane":
                bar = (
                    f"{self.progress_color}➤{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "circle":
                bar = (
                    f"{self.progress_color}◯{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "square":
                bar = (
                    f"{self.progress_color}■{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "triangle":
                bar = (
                    f"{self.progress_color}▲{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "diamond":
                bar = (
                    f"{self.progress_color}◆{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "star":
                bar = (
                    f"{self.progress_color}★{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "heart":
                bar = (
                    f"{self.progress_color}♥{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case "cross":
                bar = (
                    f"{self.progress_color}✖{self.reset_color}" * filled_length
                    + f"{self.progress_empty_color}━{self.reset_color}"
                    * (self.length - filled_length)
                )
            case _:
                raise ValueError(
                    "Invalid style option. Choose from ['default', '--', '==', '-=', 'airplane', 'circle', 'square', 'triangle', 'diamond', 'star', 'heart', 'cross']"
                )
        return bar

    def update(self, current, metric=None):

        if metric is not None and self.show_suffix:
            self.suffix = " - ".join([f"{key}: {value:.4f}" for key, value in metric])
            self.suffix = f" - {self.suffix}"
        else:
            self.suffix = ""

        # Calculate elapsed time
        self.current_time = time.time()
        elapsed_time = int(self.current_time - self.start_time)
        step_time = (
            self.current_time - self.last_update_time
        ) * 1000  # Time per step in ms

        # Format time per step
        step_time_formatted = f"{int(step_time)}ms/step"
        self.last_update_time = self.current_time

        self.current = current
        filled_length = int(self.length * self.current // self.total)

        percent = self.current / self.total * 100

        if self.show_diff_color:
            if percent < 25:
                color = "\033[31m"  # Dark Red
            elif percent < 50:
                color = "\033[91m"  # Light Red
            elif percent < 70:
                color = "\033[32m"  # Dark Green
            elif percent <= 100:
                color = "\033[92m"  # Light Green
        else:
            color = self.progress_color

        # Set the color based on the percentage
        self.progress_color = color
        
        # Different styles for the progress bar
        bar = self.__bar(filled_length, percent)

        # Print the progress bar
        self.__progress_bar(
            self.current, bar, elapsed_time, step_time_formatted, percent
        )

        if self.show_val_metrics:
            if self.current - 1 == self.total:
                # Rest the start time for the next progress bar
                self.start_time = time.time()
                self.last_update_time = self.start_time
                print()
        else:
            if self.current == self.total:
                # Rest the start time for the next progress bar
                self.start_time = time.time()
                print()

    def last_update(self, metric=None):

        if self.show_val_metrics == False:
            raise ValueError("show_validation must be True to use last_update")

        if metric is not None:
            suffix = " - ".join([f"{key}: {value:.4f}" for key, value in metric])
            self.suffix = f"{self.suffix} - {suffix}"

        # Calculate elapsed time
        current_time = time.time()
        step_time = (current_time - self.last_update_time) * 1000  # ms per step
        elapsed_time = int(current_time - self.start_time)
        self.last_update_time = current_time

        # Format time per step
        step_time_formatted = f"{int(step_time)}ms/step"
        
        # Calculate filled length
        filled_length = int(self.length * self.current // self.total)
        
        # Calculate percent
        percent = self.current / self.total * 100
        
        # Different styles for the progress bar
        bar = self.__bar(filled_length, percent)

        # Print the progress bar
        self.__progress_bar(
            self.current, bar, elapsed_time, step_time_formatted, percent=100
        )

        # Rest the start time for the next progress bar
        self.start_time = time.time()
        print()
