# Praise Ye The Lord

# Import libraries
import numpy as np
from dataclasses import dataclass
from typing import Any

class ParamType:
    name: str
    default: Any

    def set_default(self, default):
        self.default = default
        return self

@dataclass
class Choice(ParamType):
    name: str
    values: np.ndarray
    default: Any

    @classmethod
    def use(cls, name, values):
        return cls(name, np.array(values), values[0])


@dataclass
class Boolean(ParamType):
    name: str
    default: Any

    @classmethod
    def use(cls, name):
        return cls(name, default=False)

    @property
    def values(self):
        return np.array([False, True])


@dataclass
class Int(ParamType):
    name: str
    min_value: int
    max_value: int
    step: int
    default: Any

    @classmethod
    def use(cls, name, min_value, max_value, step):
        if (not isinstance(min_value, int) or
            not isinstance(max_value, int) or
            not isinstance(step, int)):
            raise ValueError("min_value, max_value and step must be a int type")
        return cls(name, min_value, max_value, step, min_value)

    @property
    def values(self):
        return np.arange(
                self.min_value,
                self.max_value,
                self.step)


@dataclass
class Float(ParamType):
    name: str
    min_value: float
    max_value: float
    step: float
    default: Any

    @classmethod
    def use(cls, name, min_value, max_value, step):
        if (not isinstance(min_value, float) or
            not isinstance(max_value, float) or
            not isinstance(step, float)):
            raise ValueError("min_value, max_value and step must "
            "be a float type")

        return cls(name, min_value, max_value, step, min_value)

    @property
    def values(self):
        if self.step not in [1.0, 0.0]:
            return np.arange(self.min_value,
                            self.max_value,
                            self.step).round(5)
        else:
            return np.linspace(self.min_value, self.max_value).round(5)
        

class HyperParameters:
    def Choice(self, name, values):
        if name not in self.__dict__:
            setattr(self, name, Choice.use(name, values))
        return self._get_attr(name).default

    def Int(self, name, min_value, max_value, step=1):
        if name not in self.__dict__:
            setattr(self, name, Int.use(name, min_value, max_value, step))
        return self._get_attr(name).default

    def Float(self, name, min_value, max_value, step=0.0):
        if name not in self.__dict__:
            setattr(self, name, Float.use(name, min_value, max_value, step))
        return self._get_attr(name).default

    def Boolean(self, name):
        if name not in self.__dict__:
            setattr(self, name, Boolean.use(name))
        return self._get_attr(name).default


    def _change_default(self, name, value):
        self.__dict__[name] = self._get_attr(name).set_default(value)

    def _get_attr(self, name):
        return self.__getattribute__(name)

    def __repr__(self):
        return str(self.__dict__)