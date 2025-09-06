# Bless be the LORD GOD of Host.

# Import libraries.
import torch as __torch__
import torch.nn as __nn__
import typing as __typing__
from src.exttorch import __types as __types__


class ModelModule(__nn__.Module):
    def __init__(
        self,
        layers: __typing__.List[__types__.Layer] | None = None,
        device: str = "cpu",
    ):
        """
        Initializes the ModelModule with the given layers and device.
        Args:
            layers (List[Layer], optional): A list of layers to be added to the model. Defaults to None.
            device (str, optional): The device on which the model will be run. Can be "TPU", "GPU", or "CPU". Defaults to "cpu".
        Raises:
            ValueError: If the device is not one of "TPU", "GPU", or "CPU".
            ImportError: If the required libraries for TPU or GPU are not available.
        """
        from exttorch.optimizers import Optimizer
        from exttorch.losses import Loss

        super().__init__()  # type: ignore
        self._xm = None

        match device:
            case "TPU" | "tpu":
                import torch_xla.core.xla_model as xm  # type: ignore

                self._xm = xm
            case "GPU" | "gpu" | "cuda" | "CUDA":

                if __torch__.cuda.is_available():
                    device = device if device.startswith("cuda") else "cuda"
                    self._device = __torch__.device(device)
                else:
                    raise ValueError("GPU is not available")
            case "CPU" | "cpu":
                self._device = __torch__.device("cpu")
            case _:
                raise ValueError("device must be either 'TPU', 'GPU' or 'CPU'.")

        self.loss: Loss | None = None
        self.loss_obj: Loss | str | None = None
        self.optimizer: Optimizer | None = None
        self.optimizer_obj: Optimizer | str | None = None
        self.layers = layers if layers else []
        self.metrics = None
        self.stop_training = False
        self._device = None
        self._model: ModelModule | __nn__.Sequential = self

    def model_state_dict(self) -> __types__.Weight:
        if self._model is not None:
            return self._model.state_dict()
        else:
            raise TypeError(
                "The model must be fitted before calling the get_weights method"
            )

    def load_model_state_dict(self, weight: __types__.Weight):
        if self._model is not None:
            self._model.load_state_dict(weight)
        else:
            raise TypeError(
                "The model must be fitted before calling the set_weights method"
            )

    def add(self, layer: __types__.Layer):
        self.layers.append(layer)

    def save(self, filepath: str):
        """
        Saves the model to the specified file path.
        Args:
            filepath (str): The path where the model will be saved.
        """
        import pickle, os

        if not os.path.exists(os.path.dirname(filepath)):
            print(f"Creating directory: {os.path.dirname(filepath)}")
            os.makedirs(os.path.dirname(filepath))

        if filepath.endswith(".ext"):
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

        elif filepath.endswith(".we"):
            weights = self.model_state_dict()
            __torch__.save(weights, filepath)
        else:
            raise ValueError("Filepath must end with .ext or .we")

    # def forward(self, *args: Any, **kwargs: Any) -> __Module__:
    # raise NotImplementedError("The forward method must be implemented in the subclass.")
