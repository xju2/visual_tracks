# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import pickle
import types
from argparse import Namespace
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union
from acctrack import utils

log = utils.get_pylogger(__name__)

class AttributeDict(Dict):
    """Extended dictionary accessible with dot notation.
    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42
    """

    def __getattr__(self, key: str) -> Optional[Any]:
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = "\n".join(rows)
        return out

ALLOWED_CONFIG_TYPES = (AttributeDict, MutableMapping, Namespace)
PRIMITIVE_TYPES = (bool, int, float, str)

def is_picklable(obj: object) -> bool:
    """Tests if an object can be pickled."""

    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, RuntimeError):
        return False


def clean_namespace(hparams: MutableMapping) -> None:
    """Removes all unpicklable entries from hparams."""
    del_attrs = [k for k, v in hparams.items() if not is_picklable(v)]

    for k in del_attrs:
        log.warning(f"attribute '{k}' removed from hparams because it cannot be pickled")
        del hparams[k]


def parse_class_init_keys(cls: Any) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse key words for standard ``self``, ``*args`` and ``**kwargs``.
    Examples:
        >>> class Model():
        ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
        ...         pass
        >>> parse_class_init_keys(Model)
        ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: List[inspect.Parameter],
        param_type: Literal[inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD],
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def get_init_args(frame: types.FrameType) -> Tuple[Optional[Any], Dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return None, {}
    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters.keys()}
    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    self_arg = local_vars.get(self_var, None)
    return self_arg, local_args


def collect_init_args(
    frame: types.FrameType,
    path_args: List[Dict[str, Any]],
    inside: bool = False,
    classes: Tuple[Type, ...] = (),
) -> List[Dict[str, Any]]:
    """Recursively collects the arguments passed to the child constructors in the inheritance tree.
    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon
        classes: the classes in which to inspect the frames
    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)
    # frame.f_back must be of a type types.FrameType for get_init_args/collect_init_args due to mypy
    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    local_self, local_args = get_init_args(frame)
    if "__class__" in local_vars and (not classes or isinstance(local_self, classes)):
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)
    if not inside:
        return collect_init_args(frame.f_back, path_args, inside=False, classes=classes)
    return path_args


def save_hyperparameters(
    obj: Any, *args: Any, ignore: Optional[Union[Sequence[str], str]] = None, frame: Optional[types.FrameType] = None
) -> None:
    """See :meth:`~lightning.pytorch.LightningModule.save_hyperparameters`"""

    if len(args) == 1 and not isinstance(args, str) and not args[0]:
        # args[0] is an empty container
        return

    if not frame:
        current_frame = inspect.currentframe()
        # inspect.currentframe() return type is Optional[types.FrameType]: current_frame.f_back called only if available
        if current_frame:
            frame = current_frame.f_back
    if not isinstance(frame, types.FrameType):
        raise AttributeError("There is no `frame` available while being required.")

    if is_dataclass(obj):
        init_args = {f.name: getattr(obj, f.name) for f in fields(obj)}
    else:
        init_args = {}

        for local_args in collect_init_args(frame, [], classes=(HyperparametersMixin,)):
            init_args.update(local_args)

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    elif isinstance(ignore, (list, tuple)):
        ignore = [arg for arg in ignore if isinstance(arg, str)]

    ignore = list(set(ignore))
    init_args = {k: v for k, v in init_args.items() if k not in ignore}

    if not args:
        # take all arguments
        hp = init_args
        obj._hparams_name = "kwargs" if hp else None
    else:
        # take only listed arguments in `save_hparams`
        isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
        if len(isx_non_str) == 1:
            hp = args[isx_non_str[0]]
            cand_names = [k for k, v in init_args.items() if v == hp]
            obj._hparams_name = cand_names[0] if cand_names else None
        else:
            hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
            obj._hparams_name = "kwargs"

    # `hparams` are expected here
    obj._set_hparams(hp)

    # for k, v in obj._hparams.items():
    #     if isinstance(v, nn.Module):
    #         log.warn(
    #             f"Attribute {k!r} is an instance of `nn.Module` and is already saved during checkpointing."
    #             f" It is recommended to ignore them using `self.save_hyperparameters(ignore=[{k!r}])`."
    #         )

    # make a deep copy so there are no other runtime changes reflected
    obj._hparams_initial = copy.deepcopy(obj._hparams)


class HyperparametersMixin:

    __jit_unused_properties__: List[str] = ["hparams", "hparams_initial"]

    def __init__(self) -> None:
        super().__init__()
        self._log_hyperparams = False

    def save_hyperparameters(
        self,
        *args: Any,
        ignore: Optional[Union[Sequence[str], str]] = None,
        frame: Optional[types.FrameType] = None,
        logger: bool = True,
    ) -> None:
        """Save arguments to ``hparams`` attribute.
        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
                or string names or arguments from class ``__init__``
            ignore: an argument name or a list of argument names from
                class ``__init__`` to be ignored
            frame: a frame object. Default is None
            logger: Whether to send the hyperparameters to the logger. Default: True
        Example::
            >>> from lightning.pytorch.core.mixins import HyperparametersMixin
            >>> class ManuallyArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # manually assign arguments
            ...         self.save_hyperparameters('arg1', 'arg3')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14
            >>> from lightning.pytorch.core.mixins import HyperparametersMixin
            >>> class AutomaticArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # equivalent automatic
            ...         self.save_hyperparameters()
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = AutomaticArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg2": abc
            "arg3": 3.14
            >>> from lightning.pytorch.core.mixins import HyperparametersMixin
            >>> class SingleArgModel(HyperparametersMixin):
            ...     def __init__(self, params):
            ...         super().__init__()
            ...         # manually assign single argument
            ...         self.save_hyperparameters(params)
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
            >>> model.hparams
            "p1": 1
            "p2": abc
            "p3": 3.14
            >>> from lightning.pytorch.core.mixins import HyperparametersMixin
            >>> class ManuallyArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # pass argument(s) to ignore as a string or in a list
            ...         self.save_hyperparameters(ignore='arg2')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14
        """
        self._log_hyperparams = logger
        # the frame needs to be created in this file.
        if not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back
        save_hyperparameters(self, *args, ignore=ignore, frame=frame)

    def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
        hp = self._to_hparams_dict(hp)

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    @staticmethod
    def _to_hparams_dict(hp: Union[MutableMapping, Namespace, str]) -> Union[MutableMapping, AttributeDict]:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")
        return hp

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user.
        For the frozen set of initial hyperparameters, use :attr:`hparams_initial`.
        Returns:
            Mutable hyperparameters dictionary
        """
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
        Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.
        Returns:
            AttributeDict: immutable initial hyperparameters
        """
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)
