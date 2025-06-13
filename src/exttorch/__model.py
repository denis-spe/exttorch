# Bless be the LORD GOD of Host.

# Import libraries.
import torch as __torch__
from torch.nn import Module as __Module__
import typing as __typing__
from exttorch import __types as __types__


class ModelModule(__Module__):
    def __init__(self, layers: __typing__.List[__types__.Layer] | None = None, device: str = "cpu"):
        """
        Initializes the ModelModule with the given layers and device.
        Args:
            layers (List[Layer], optional): A list of layers to be added to the model. Defaults to None.
            device (str, optional): The device on which the model will be run. Can be "TPU", "GPU", or "CPU". Defaults to "cpu".
        Raises:
            ValueError: If the device is not one of "TPU", "GPU", or "CPU".
            ImportError: If the required libraries for TPU or GPU are not available.
        """
        from exttorch.callbacks import Callback
        super().__init__() # type: ignore
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
        
        self.loss = None
        self.loss_obj = None
        self.optimizer = None
        self.optimizer_obj = None
        self.layers = layers if layers else []
        self.metrics = None
        self.__callbacks: __typing__.List[Callback] | None = None
        self.__progressbar = None
        self.stop_training = False
        self._device = None
        self.__verbose = None
        self.__val_data_size = None
        self._model: ModelModule = self

    def get_weights(self) -> __types__.Weight:
        if self._model is not None:
            return self._model.state_dict()
        else:
            raise TypeError("The model must be fitted before calling the get_weights method")

    def set_weights(self, weight: __types__.Weight):
        if self._model is not None:
            self._model.load_state_dict(weight)
        else:
            raise TypeError("The model must be fitted before calling the set_weights method")

    def add(self, layer: __types__.Layer):
        self.layers.append(layer)

    def _handle_callbacks(self, callback_method: str, logs = None, epoch: int | None = None):

        if self.__callbacks is not None:
            for callback in self.__callbacks:
                # Set the model and stop_training to the callback
                callback.model = self

                # Check if the present callback method
                match callback_method:
                    case "on_train_begin":
                        callback.on_train_begin()
                    case "on_train_end":
                        callback.on_train_end(logs)
                    case "on_validation_begin":
                        callback.on_validation_begin()
                    case "on_validation_end":
                        callback.on_validation_end(logs)
                    case "on_batch_begin":
                        callback.on_batch_begin()
                    case "on_batch_end":
                        callback.on_batch_end(logs)
                    case "on_epoch_begin":
                        if epoch is None:
                            raise ValueError("epoch must be provided for on_epoch_begin callback method")
                        callback.on_epoch_begin(epoch)
                    case "on_epoch_end":
                        if epoch is None:
                            raise ValueError("epoch must be provided for on_epoch_end callback method")
                        callback.on_epoch_end(epoch, logs)
                    case _:
                        raise ValueError("Unknown callback_method name: {}".format(callback_method))
        
    # def forward(self, *args: Any, **kwargs: Any) -> __Module__:
        # raise NotImplementedError("The forward method must be implemented in the subclass.")
    