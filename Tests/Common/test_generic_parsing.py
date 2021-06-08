#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from typing import Any, List, Optional, Tuple

import param
import pytest

from InnerEye.Common.generic_parsing import GenericConfig, IntTuple, create_from_matching_params


class ParamEnum(Enum):
    EnumValue1 = "1",
    EnumValue2 = "2"


class ParamClass(GenericConfig):
    name: str = param.String(None, doc="Name")
    seed: int = param.Integer(42, doc="Seed")
    flag: bool = param.Boolean(False, doc="Flag")
    not_flag: bool = param.Boolean(True, doc="Not Flag")
    number: float = param.Number(3.14)
    integers: List[int] = param.List(None, class_=int)
    optional_int: Optional[int] = param.Integer(None, doc="Optional int")
    optional_float: Optional[float] = param.Number(None, doc="Optional float")
    floats: List[float] = param.List(None, class_=float)
    tuple1: Tuple[int, float] = param.NumericTuple((1, 2.3), length=2, doc="Tuple")
    int_tuple: Tuple[int, int, int] = IntTuple((1, 1, 1), length=3, doc="Integer Tuple")
    enum: ParamEnum = param.ClassSelector(default=ParamEnum.EnumValue1, class_=ParamEnum, instantiate=False)
    readonly: str = param.String("Nope", readonly=True)
    _non_override: str = param.String("Nope")
    constant: str = param.String("Nope", constant=True)


def test_overridable_parameter() -> None:
    """
    Test to check overridable parameters are correctly identified.
    """
    param_dict = ParamClass.get_overridable_parameters()
    assert "name" in param_dict
    assert "flag" in param_dict
    assert "not_flag" in param_dict
    assert "seed" in param_dict
    assert "number" in param_dict
    assert "integers" in param_dict
    assert "optional_int" in param_dict
    assert "optional_float" in param_dict
    assert "tuple1" in param_dict
    assert "int_tuple" in param_dict
    assert "enum" in param_dict
    assert "readonly" not in param_dict
    assert "_non_override" not in param_dict
    assert "constant" not in param_dict


def test_create_parser() -> None:
    """
    Check that parse_args works as expected, with both non default and default values.
    """

    def check(arg: List[str], expected_key: str, expected_value: Any) -> None:
        parsed = ParamClass.parse_args(arg)
        assert getattr(parsed, expected_key) == expected_value

    def check_fails(arg: List[str]) -> None:
        with pytest.raises(SystemExit) as e:
            ParamClass.parse_args(arg)
        assert e.type == SystemExit
        assert e.value.code == 2

    check(["--name=foo"], "name", "foo")
    check(["--seed", "42"], "seed", 42)
    check(["--seed", ""], "seed", 42)
    check(["--number", "2.17"], "number", 2.17)
    check(["--number", ""], "number", 3.14)
    check(["--integers", "1,2,3"], "integers", [1, 2, 3])
    check(["--optional_int", ""], "optional_int", None)
    check(["--optional_int", "2"], "optional_int", 2)
    check(["--optional_float", ""], "optional_float", None)
    check(["--optional_float", "3.14"], "optional_float", 3.14)
    check(["--tuple1", "1,2"], "tuple1", (1, 2.0))
    check(["--int_tuple", "1,2,3"], "int_tuple", (1, 2, 3))
    check(["--enum=2"], "enum", ParamEnum.EnumValue2)
    check(["--floats=1,2,3.14"], "floats", [1., 2., 3.14])
    check(["--integers=1,2,3"], "integers", [1, 2, 3])
    check(["--flag=on"], "flag", True)
    check(["--flag=On"], "flag", True)
    check(["--flag=t"], "flag", True)
    check(["--flag=T"], "flag", True)
    check(["--flag=true"], "flag", True)
    check(["--flag=True"], "flag", True)
    check(["--flag=y"], "flag", True)
    check(["--flag=Y"], "flag", True)
    check(["--flag=yes"], "flag", True)
    check(["--flag=Yes"], "flag", True)
    check(["--flag=1"], "flag", True)
    check(["--flag=off"], "flag", False)
    check(["--flag=Off"], "flag", False)
    check(["--flag=f"], "flag", False)
    check(["--flag=F"], "flag", False)
    check(["--flag=false"], "flag", False)
    check(["--flag=False"], "flag", False)
    check(["--flag=n"], "flag", False)
    check(["--flag=N"], "flag", False)
    check(["--flag=no"], "flag", False)
    check(["--flag=No"], "flag", False)
    check(["--flag=0"], "flag", False)
    check(["--not_flag=false"], "not_flag", False)
    check(["--not_flag=true"], "not_flag", True)
    check(["--flag"], "flag", True)
    check_fails(["--not_flag"])
    check_fails(["--flag=Falsf"])
    check_fails(["--flag=Truf"])
    # Check that default values are created as expected, and that the non-overridable parameters
    # are omitted.
    defaults = vars(ParamClass.create_argparser().parse_args([]))
    assert defaults["seed"] == 42
    assert defaults["tuple1"] == (1, 2.3)
    assert defaults["int_tuple"] == (1, 1, 1)
    assert defaults["enum"] == ParamEnum.EnumValue1
    assert not defaults["flag"]
    assert defaults["not_flag"]
    assert "readonly" not in defaults
    assert "constant" not in defaults
    assert "_non_override" not in defaults
    # We can't test if all invalid cases are handled because argparse call sys.exit
    # upon errors.


def test_apply_overrides() -> None:
    """
    Test that overrides are applied correctly, ond only to overridable parameters,
    """
    m = ParamClass()
    overrides = {"name": "newName", "int_tuple": (0, 1, 2)}
    actual_overrides = m.apply_overrides(overrides)
    assert actual_overrides == overrides
    assert all([x == i and isinstance(x, int) for i, x in enumerate(m.int_tuple)])
    assert m.name == "newName"
    # Attempt to change seed and constant, but the latter should be ignored.
    change_seed = {"seed": 123}
    old_constant = m.constant
    changes2 = m.apply_overrides({**change_seed, "constant": "Nothing"})
    assert changes2 == change_seed
    assert m.seed == 123
    assert m.constant == old_constant


@pytest.mark.parametrize("value_idx_0", [1.0, 1])
@pytest.mark.parametrize("value_idx_1", [2.0, 2])
@pytest.mark.parametrize("value_idx_2", [3.0, 3])
def test_int_tuple_validation(value_idx_0: Any, value_idx_1: Any, value_idx_2: Any) -> None:
    """
    Test integer tuple parameter is validated correctly.
    """
    m = ParamClass()
    val = (value_idx_0, value_idx_1, value_idx_2)
    if not all([isinstance(x, int) for x in val]):
        with pytest.raises(ValueError):
            m.int_tuple = (value_idx_0, value_idx_1, value_idx_2)
    else:
        m.int_tuple = (value_idx_0, value_idx_1, value_idx_2)


class ClassFrom(param.Parameterized):
    foo = param.String("foo")
    bar = param.Integer(1)
    baz = param.String("baz")
    _private = param.String("private")
    constant = param.String("constant", constant=True)


class ClassTo(param.Parameterized):
    foo = param.String("foo2")
    bar = param.Integer(2)
    _private = param.String("private2")
    constant = param.String("constant2", constant=True)


class NotParameterized:
    foo = 1


def test_create_from_matching_params() -> None:
    """
    Test if Parameterized objects can be cloned by looking at matching fields.
    """
    class_from = ClassFrom()
    class_to = create_from_matching_params(class_from, cls_=ClassTo)
    assert isinstance(class_to, ClassTo)
    assert class_to.foo == "foo"
    assert class_to.bar == 1
    # Constant fields should not be touched
    assert class_to.constant == "constant2"
    # Private fields must be copied over.
    assert class_to._private == "private"
    # Baz is only present in the "from" object, and should not be copied to the new object
    assert not hasattr(class_to, "baz")

    with pytest.raises(ValueError) as ex:
        create_from_matching_params(class_from, NotParameterized)
    assert "subclass of param.Parameterized" in str(ex)
    assert "NotParameterized" in str(ex)
