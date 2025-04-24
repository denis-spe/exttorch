# Holy, holy, holy is the LORD of host, the whole earth is full of his glory.

# Importing necessary modules
import time
import base64
from importlib.resources import files
from IPython.display import HTML, display

def _is_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return 'IPKernelApp' in ip.config
    except:
        return False

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
        epochs: int | None = None
    ):
        self._total = 0
        self.length = bar_width
        self.current = 0
        self.start_time = time.time()
        self.progress_color = (
            progress_color if not _is_notebook()
            else "<span style='color: lightgreen;'>"
        )
        self.progress_empty_color = (
            empty_color if not _is_notebook()
            else "<span style='color: darkgray;'>"
        )
        self.reset_color = "\033[0m" if not _is_notebook() else "</span>"
        self.show_val_metrics = show_val_metrics
        self.suffix = ""
        self.style = style
        self.last_update_time = self.start_time
        self.verbose = verbose
        self.check_mark = show_check_mark
        self.show_diff_color = show_diff_color
        self.show_suffix = show_suffix
        self.__pre_epoch = 1
        self.epochs = epochs

        # Grab the exttorch package root as a Traversable
        pkg_root = files("exttorch")

        # Navigate into the data-only folder
        gif_file = pkg_root.joinpath("assets", "spinner.gif")

        # Read the bytes
        data = gif_file.read_bytes()
        self.loader_img = base64.b64encode(data).decode()


        if _is_notebook():
            self._handle = display(HTML("<pre></pre>"), display_id=True)

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Total must be a positive integer.")
        self._total = value

    @property
    def add_epoch(self) -> int:
        return self.__pre_epoch

    @add_epoch.setter
    def add_epoch(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError(f"Except an int but received `{type(value)}`")

        self.__pre_epoch = value + 1 if value != 1 else value
        
        if not _is_notebook():
            print(f"\033[1mEpoch {self.__pre_epoch}/{self.epochs}\033[0m")

    def __format_bar(self, current, bar, elapsed_time, step_time, percent):
        if percent == 100 and self.check_mark:
            if _is_notebook():
                done = "<span style='color: lightgreen;'> &check; </span>"
            else:
                done = f" {self.progress_color}✔️{self.reset_color} "
        else:
            # https://i.gifer.com/ZZ5H.gif
            done = (
                " ⏳ " if not _is_notebook()
                else f" <img src='data:image/gif;base64,{self.loader_img}' height='20' style='vertical-align:middle;'> "
            )
        suffix = self.suffix if self.show_suffix else ""
        epoch = f"<strong>Epoch {self.__pre_epoch}/{self.epochs}</strong>\n" if _is_notebook() and self.epochs else ""

        format_map = {
            "verbose": f"{epoch}{current}/{self.total}{done}{bar} {elapsed_time}s {step_time} {suffix}",
            "silent": f"{epoch}{current}/{self.total}{done}{elapsed_time}s {step_time} {suffix}",
            "silent_verbose": f"{epoch}{current}/{self.total}{done}{bar} {elapsed_time}s {step_time}",
            "silent_verbose_suffix": f"{epoch}{current}/{self.total}{done}{suffix}",
            "silent_epoch": f"{epoch}{current}/{self.total}{done}{bar} {elapsed_time}s {step_time} {suffix}",
            "silent_epoch_suffix": f"{epoch}{current}/{self.total}{done}{suffix}",
            None: f"{epoch}{current}/{self.total}{done}{bar} {elapsed_time}s {step_time} {suffix}"
        }
        return format_map.get(self.verbose, "Invalid verbose setting")

    def __bar(self, filled_length, percent):
        style_chars = {
            "default": "━",
            "--": "-",
            "==": "=",
            "-=": "-" if percent <= 50 else "=",
            "airplane": "➤",
            "circle": "◯",
            "square": "■",
            "triangle": "▲",
            "diamond": "◆",
            "star": "★",
            "heart": "♥",
            "cross": "✖",
        }
        char = style_chars.get(self.style)
        if not char:
            raise ValueError("Invalid style option.")

        return (
            f"{self.progress_color}{char}{self.reset_color}" * filled_length +
            f"{self.progress_empty_color}━{self.reset_color}" * (self.length - filled_length)
        )

    def update(self, current, metric=None):
        if metric and self.show_suffix:
            self.suffix = " - ".join([f"{k}: {v:.4f}" for k, v in metric])
            self.suffix = f" - {self.suffix}"

        self.current_time = time.time()
        elapsed = int(self.current_time - self.start_time)
        step_time = int((self.current_time - self.last_update_time) * 1000)
        self.last_update_time = self.current_time

        self.current = current
        filled = int(self.length * self.current // self.total)
        percent = self.current / self.total * 100

        bar = self.__bar(filled, percent)
        progress = self.__format_bar(self.current, bar, elapsed, f"{step_time}ms/step", percent)

        if _is_notebook():
            self._handle.update(HTML(f"<pre>{progress}</pre>"))
            if self.current == self.total and not self.show_val_metrics:
                self.new_epoch()
        else:
            print(f"\r{progress}", end="", flush=True)
            if self.show_val_metrics:
                if self.current - 1 == self.total:
                    # Rest the start time for the next progress bar
                    self.new_epoch()
                    print()
            else:
                if self.current == self.total:
                    # Rest the start time for the next progress bar
                    self.new_epoch()
                    print()


    def last_update(self, metric=None):
        if not self.show_val_metrics:
            raise ValueError("show_val_metrics must be True to use last_update")

        if metric:
            val_suffix = " - ".join([f"{k}: {v:.4f}" for k, v in metric])
            self.suffix += f" - {val_suffix}"

        elapsed = int(time.time() - self.start_time)
        step_time = int((time.time() - self.last_update_time) * 1000)
        filled = int(self.length * self.current // self.total)
        percent = self.current / self.total * 100

        bar = self.__bar(filled, percent)
        final = self.__format_bar(self.current, bar, elapsed, f"{step_time}ms/step", 100)

        if _is_notebook():
            self._handle.update(HTML(f"<pre>{final}</pre>"))
            self.new_epoch()
        else:
            print(f"\r{final}", flush=True)
            print()

    def new_epoch(self):
        self.current = 0
        if self.show_val_metrics:
            self.start_time = time.time()
            self.last_update_time = self.start_time
            self.suffix = ""

        if _is_notebook():
            self._handle = display(HTML("<pre></pre>"), display_id=True)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.stop()


    def stop(self, error_message=None):
        if _is_notebook():
            done = "<span style='color: red;'>✖</span>"
        else:
            done = "\033[91m✖\033[0m"  # ANSI red X

        filled_length = int(self.length * self.current // max(self.total, 1))
        percent = self.current / max(self.total, 1) * 100 if self.total else 0

        bar = self.__bar(filled_length, percent)

        epoch = f"Epoch {self.__pre_epoch}/{self.epochs}\n" if _is_notebook() and self.epochs else ""

        # Custom message for notebook or terminal
        message = f"{epoch}{self.current}/{self.total} {done} {bar} {self.suffix}"
        if error_message:
            message += f" - {error_message}"

        if _is_notebook():
            self._handle.update(HTML(f"<pre>{message}</pre>"))
        else:
            print(message)
