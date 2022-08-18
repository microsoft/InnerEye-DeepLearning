#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import param
from param.parameterized import Parameter

from InnerEye.Common.common_util import is_private_field_name
from InnerEye.Common.type_annotations import T

# Need this as otherwise a description of all the params in a class is added to the class docstring
# which makes generated documentation with sphinx messy.
param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False


class ListOrDictParam(param.Parameter):
    """
    Wrapper class to allow either a List or Dict inside of a Parameterized object.
    """

    def _validate(self, val: Any) -> None:
        if not (self.allow_None and val is None):
            if not (isinstance(val, List) or isinstance(val, Dict)):
                raise ValueError(f"{val} must be an instance of List or Dict, found {type(val)}")
        super()._validate(val)


class StringOrStringList(param.Parameter):
    """
    Wrapper class to allow either a string or a list of strings. Internally represented always as a list.
    """

    def _validate(self, val: Any) -> None:
        if isinstance(val, str):
            return
        if isinstance(val, List):
            if all([isinstance(v, str) for v in val]):
                return
        raise ValueError(f"{val} must be a string or a list of strings")

    def set_hook(self, obj: Any, val: Any) -> Any:
        """
        Modifies the value before calling the setter. Here, we are converting all strings to lists of strings.
        """
        if isinstance(val, str):
            return [val]
        return val


class PathOrPathList(param.Parameter):
    """
    Wrapper class to allow either a Path or a list of Paths. Internally represented always as a list.
    """

    def _validate(self, val: Any) -> None:
        if isinstance(val, Path):
            return
        if isinstance(val, List):
            if all([isinstance(v, Path) for v in val]):
                return
        raise ValueError(f"{val} must be a Path object or a list of paths")

    def set_hook(self, obj: Any, val: Any) -> Any:
        """
        Modifies the value before calling the setter. Here, we are converting simple path to lists of path.
        """
        if isinstance(val, Path):
            return [val]
        return val


class IntTuple(param.NumericTuple):
    """
    Parameter class that must always have integer values
    """

    def _validate(self, val: Any) -> None:
        super()._validate(val)
        if val is not None:
            for i, n in enumerate(val):
                if not isinstance(n, int):
                    raise ValueError("{}: tuple element at index {} with value {} in {} is not an integer"
                                     .format(self.name, i, n, val))


class GenericConfig(param.Parameterized):
    """
    Base class for all configuration classes provides helper functionality to create argparser.
    """

    def __init__(self, should_validate: bool = True, throw_if_unknown_param: bool = False, **params: Any):
        """
        Instantiates the config class, ignoring parameters that are not overridable.

        :param should_validate: If True, the validate() method is called directly after init.
        :param throw_if_unknown_param: If True, raise an error if the provided "params" contains any key that does not
                                correspond to an attribute of the class.

        :param params: Parameters to set.
        """
        # check if illegal arguments are passed in
        legal_params = self.get_overridable_parameters()
        illegal = [k for k, v in params.items() if (k in self.params().keys()) and (k not in legal_params)]

        if illegal:
            raise ValueError(f"The following parameters cannot be overriden as they are either "
                             f"readonly, constant, or private members : {illegal}")
        if throw_if_unknown_param:
            # check if parameters not defined by the config class are passed in
            unknown = [k for k, v in params.items() if (k not in self.params().keys())]
            if unknown:
                raise ValueError(f"The following parameters do not exist: {unknown}")
        # set known arguments
        super().__init__(**{k: v for k, v in params.items() if k in legal_params.keys()})
        if should_validate:
            self.validate()

    def validate(self) -> None:
        """
        Validation method called directly after init to be overridden by children if required
        """
        pass

    def add_and_validate(self, kwargs: Dict[str, Any], validate: bool = True) -> None:
        """
        Add further parameters and, if validate is True, validate. We first try set_param, but that
        fails when the parameter has a setter.
        """
        for key, value in kwargs.items():
            try:
                self.set_param(key, value)
            except ValueError:
                setattr(self, key, value)
        if validate:
            self.validate()

    @classmethod
    def create_argparser(cls: Type[GenericConfig]) -> argparse.ArgumentParser:
        """
        Creates an ArgumentParser with all fields of the given argparser that are overridable.
        :return: ArgumentParser
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cls.add_args(parser)

        return parser

    @classmethod
    def add_args(cls: Type[GenericConfig],
                 parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Adds all overridable fields of the current class to the given argparser.
        Fields that are marked as readonly, constant or private are ignored.

        :param parser: Parser to add properties to.
        """

        def parse_bool(x: str) -> bool:
            """
            Parse a string as a bool. Supported values are case insensitive and one of:
            'on', 't', 'true', 'y', 'yes', '1' for True
            'off', 'f', 'false', 'n', 'no', '0' for False.

            :param x: string to test.
            :return: Bool value if string valid, otherwise a ValueError is raised.
            """
            sx = str(x).lower()
            if sx in ('on', 't', 'true', 'y', 'yes', '1'):
                return True
            if sx in ('off', 'f', 'false', 'n', 'no', '0'):
                return False
            raise ValueError(f"Invalid value {x}, please supply one of True, true, false or False.")

        def _get_basic_type(_p: param.Parameter) -> Union[type, Callable]:
            """
            Given a parameter, get its basic Python type, e.g.: param.Boolean -> bool.
            Throw exception if it is not supported.

            :param _p: parameter to get type and nargs for.
            :return: Type
            """
            if isinstance(_p, param.Boolean):
                p_type: Callable = parse_bool
            elif isinstance(_p, param.Integer):
                p_type = lambda x: _p.default if x == "" else int(x)
            elif isinstance(_p, param.Number):
                p_type = lambda x: _p.default if x == "" else float(x)
            elif isinstance(_p, param.String):
                p_type = str
            elif isinstance(_p, param.List):
                p_type = lambda x: [_p.class_(item) for item in x.split(',')]
            elif isinstance(_p, param.NumericTuple):
                float_or_int = lambda y: int(y) if isinstance(_p, IntTuple) else float(y)
                p_type = lambda x: tuple([float_or_int(item) for item in x.split(',')])
            elif isinstance(_p, param.ClassSelector):
                p_type = _p.class_
            elif isinstance(_p, ListOrDictParam):
                def list_or_dict(x: str) -> Union[Dict, List]:
                    import json
                    if x.startswith("{") or x.startswith('['):
                        res = json.loads(x)
                    else:
                        res = [str(item) for item in x.split(',')]
                    if isinstance(res, Dict):
                        return res
                    elif isinstance(res, List):
                        return res
                    else:
                        raise ValueError(f"Parameter of type {_p} should resolve to List or Dict")

                p_type = list_or_dict
            else:
                raise TypeError("Parameter of type: {} is not supported".format(_p))

            return p_type

        def add_boolean_argument(parser: argparse.ArgumentParser, k: str, p: Parameter) -> None:
            """
            Add a boolean argument.
            If the parameter default is False then allow --flag (to set it True) and --flag=Bool as usual.
            If the parameter default is True then allow --no-flag (to set it to False) and --flag=Bool as usual.

            :param parser: parser to add a boolean argument to.
            :param k: argument name.
            :param p: boolean parameter.
            """
            if not p.default:
                # If the parameter default is False then use nargs="?" (argparse.OPTIONAL).
                # This means that the argument is optional.
                # If it is not supplied, i.e. in the --flag mode, use the "const" value, i.e. True.
                # Otherwise, i.e. in the --flag=value mode, try to parse the argument as a bool.
                parser.add_argument("--" + k, help=p.doc, type=parse_bool, default=False,
                                    nargs=argparse.OPTIONAL, const=True)
            else:
                # If the parameter default is True then create an exclusive group of arguments.
                # Either --flag=value as usual
                # Or --no-flag to store False in the parameter k.
                group = parser.add_mutually_exclusive_group(required=False)
                group.add_argument("--" + k, help=p.doc, type=parse_bool)
                group.add_argument('--no-' + k, dest=k, action='store_false')
                parser.set_defaults(**{k: p.default})

        for k, p in cls.get_overridable_parameters().items():
            # param.Booleans need to be handled separately, they are more complicated because they have
            # an optional argument.
            if isinstance(p, param.Boolean):
                add_boolean_argument(parser, k, p)
            else:
                parser.add_argument("--" + k, help=p.doc, type=_get_basic_type(p), default=p.default)

        return parser

    @classmethod
    def parse_args(cls: Type[T], args: Optional[List[str]] = None) -> T:
        """
        Creates an argparser based on the params class and parses stdin args (or the args provided)
        """
        return cls(**vars(cls.create_argparser().parse_args(args)))  # type: ignore

    @classmethod
    def get_overridable_parameters(cls: Type[GenericConfig]) -> Dict[str, param.Parameter]:
        """
        Get properties that are not constant, readonly or private (eg: prefixed with an underscore).

        :return: A dictionary of parameter names and their definitions.
        """
        return dict((k, v) for k, v in cls.params().items()
                    if cls.reason_not_overridable(v) is None)

    @staticmethod
    def reason_not_overridable(value: param.Parameter) -> Optional[str]:
        """
        :param value: a parameter value
        :return: None if the parameter is overridable; otherwise a one-word string explaining why not.
        """
        if value.readonly:
            return "readonly"
        elif value.constant:
            return "constant"
        elif is_private_field_name(value.name):
            return "private"
        elif isinstance(value, param.Callable):
            return "callable"
        return None

    def apply_overrides(self, values: Optional[Dict[str, Any]], should_validate: bool = True,
                        keys_to_ignore: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Applies the provided `values` overrides to the config.
        Only properties that are marked as overridable are actually overwritten.

        :param values: A dictionary mapping from field name to value.
        :param should_validate: If true, run the .validate() method after applying overrides.
        :param keys_to_ignore: keys to ignore in reporting failed overrides. If None, do not report.
        :return: A dictionary with all the fields that were modified.
        """

        def _apply(_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            applied: Dict[str, Any] = {}
            if _overrides is not None:
                overridable_parameters = self.get_overridable_parameters().keys()
                for k, v in _overrides.items():
                    if k in overridable_parameters:
                        applied[k] = v
                        setattr(self, k, v)

            return applied

        actual_overrides = _apply(values)
        if keys_to_ignore is not None:
            self.report_on_overrides(values, keys_to_ignore)  # type: ignore
        if should_validate:
            self.validate()
        return actual_overrides

    def report_on_overrides(self, values: Dict[str, Any], keys_to_ignore: Set[str]) -> None:
        """
        Logs a warning for every parameter whose value is not as given in "values", other than those
        in keys_to_ignore.

        :param values: override dictionary, parameter names to values
        :param keys_to_ignore: set of dictionary keys not to report on
        :return: None
        """
        for key, desired in values.items():
            # If this isn't an AzureConfig instance, we don't want to warn on keys intended for it.
            if key in keys_to_ignore:
                continue
            actual = getattr(self, key, None)
            if actual == desired:
                continue
            if key not in self.params():
                reason = "parameter is undefined"
            else:
                val = self.params()[key]
                reason = self.reason_not_overridable(val)  # type: ignore
                if reason is None:
                    reason = "for UNKNOWN REASONS"
                else:
                    reason = f"parameter is {reason}"
            # We could raise an error here instead - to be discussed.
            logging.warning(f"Override {key}={desired} failed: {reason} in class {self.__class__.name}")


def create_from_matching_params(from_object: param.Parameterized, cls_: Type[T]) -> T:
    """
    Creates an object of the given target class, and then copies all attributes from the `from_object` to
    the newly created object, if there is a matching attribute. The target class must be a subclass of
    param.Parameterized.

    :param from_object: The object to read attributes from.
    :param cls_: The name of the class for the newly created object.
    :return: An instance of cls_
    """
    c = cls_()
    if not isinstance(c, param.Parameterized):
        raise ValueError(f"The created object must be a subclass of param.Parameterized, but got {type(c)}")
    for param_name, p in c.params().items():
        if not p.constant and not p.readonly:
            setattr(c, param_name, getattr(from_object, param_name))
    return c
