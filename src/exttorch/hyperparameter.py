# Praise Ye The Lord

# Import libraries
# import numpy as np
from dataclasses import dataclass

class ParamType:
    name: str
    default: any

    def set_default(self, default):
        """
        Sets the new default value
        
        Parameters
        ----------
        default : Any
            Default value
        """
        self.default = default
        return self

@dataclass
class Choice(ParamType):
    name: str
    values: list
    default: any

    @classmethod
    def use(cls, name: str, values: list):
        """
        Use the choice parameter type.
        
        Parameters
        ----------
        name : str
            Name of the parameter
        values : np.ndarray
            Parameter choices to try to run in the model
        """
        # Import numpy
        import numpy as np

        return cls(name, np.array(values), values[0])


@dataclass
class Boolean(ParamType):
    """
    Boolean parameter type.
    """
    name: str
    default: any

    @classmethod
    def use(cls, name: str):
        """
        Use the Boolean parameter type.
        
        Parameters
        ----------
        name : str
            Name of the parameter
        """
        return cls(name, default=False)

    @property
    def values(self):
        """
        Returns the values of the parameter.
        """
        # Import numpy
        import numpy as np
        
        return np.array([False, True])


@dataclass
class Int(ParamType):
    """
    Int parameter type.
    """
    name: str
    min_value: int
    max_value: int
    step: int
    default: any

    @classmethod
    def use(cls, name, min_value, max_value, step):
        """
        Use the Int parameter
        
        Parameters
        ----------
        name : str
            Name of the parameter
        min_value : int
            Minimum value of the parameter
        max_value : int
            Maximum value of the parameter
        step : int
            Step of the parameter
        """
        if (not isinstance(min_value, int) or
            not isinstance(max_value, int) or
            not isinstance(step, int)):
            raise ValueError("min_value, max_value and step must be a int type")
        return cls(name, min_value, max_value, step, min_value)

    @property
    def values(self):
        """
        Returns the values of the parameter.
        """
        # Import numpy
        import numpy as np
        
        return np.arange(
                self.min_value,
                self.max_value,
                self.step)


@dataclass
class Float(ParamType):
    """
    Float parameter type.
    """
    name: str
    min_value: float
    max_value: float
    step: float
    default: any

    @classmethod
    def use(cls, name, min_value, max_value, step):
        """
        Use the Float parameter
        
        Parameters
        ----------
        name : str
            Name of the parameter
        min_value : int
            Minimum value of the parameter
        max_value : int
            Maximum value of the parameter
        step : int
            Step of the parameter
        """
        if (not isinstance(min_value, float) or
            not isinstance(max_value, float) or
            not isinstance(step, float)):
            raise ValueError("min_value, max_value and step must "
            "be a float type")

        return cls(name, min_value, max_value, step, min_value)

    @property
    def values(self):
        """
        Returns the values of the parameter.
        """
        # Import numpy
        import numpy as np
        
        if self.step not in [1.0, 0.0]:
            # Returns an array of different step
            return np.arange(self.min_value,
                            self.max_value,
                            self.step).round(5)
        else:
            return np.linspace(self.min_value, self.max_value).round(5)
        

class HyperParameters:
    """
    The class represents hyperparameters for tuning the Sequential model.
    """
    def Choice(self, name, values):
        """
        Represent a list of choice for hyper tuning the model.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        values : list
            Parameter choices to try to run in the model.
        """
        if name not in self.__dict__:
            setattr(self, name, Choice.use(name, values))
        return self._get_attr(name).default

    def Int(self, name: int, min_value: int, max_value: int, step=1):
        """
        Represent int value for hyper tuning the models.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        min_value : int
            Minimum value of the parameter.
        max_value : int
            Maximum value of the parameter.
        step : int, optional
            Step of the parameter, by default 1
        """
        if name not in self.__dict__:
            # Creates a new attribute to the Hyperparameters class
            setattr(self, name, Int.use(name, min_value, max_value, step))
            
        return self._get_attr(name).default

    def Float(self, name, min_value, max_value, step=0.0):
        """
        Represents the float value for hyper tuning the models.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        min_value : float
            Minimum value of the parameter.
        max_value : float
            Maximum value of the parameter.
        step : float, optional
            Step of the parameter, by default 0.0
        """
        if name not in self.__dict__:
            setattr(self, name, Float.use(name, min_value, max_value, step))
        return self._get_attr(name).default

    def Boolean(self, name):
        """
        Represents the boolean value for hyper tuning the model.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        """
        if name not in self.__dict__:
            setattr(self, name, Boolean.use(name))
        return self._get_attr(name).default


    def change_default(self, name, value):
        """
        Changes the default value.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        value : Any
            New default value.
        """
        self.__dict__[name] = self._get_attr(name).set_default(value)

    def _get_attr(self, name):
        """
        Get the attribute from the HyperParameters class.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        """
        return self.__getattribute__(name)

    def __repr__(self):
        """
        Returns the string representation of the HyperParameters class.
        """
        return str(self.__dict__)