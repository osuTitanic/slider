from __future__ import annotations

import datetime
from functools import lru_cache
from typing import Any, Callable, Generic, Iterator, MutableSequence, TypeVar, overload


_OwnerT = TypeVar("_OwnerT")
_ValueT = TypeVar("_ValueT")


class lazyval(Generic[_OwnerT, _ValueT]):
    """Decorator to lazily compute and cache a value."""

    def __init__(self, fget: Callable[[_OwnerT], _ValueT]):
        self._fget = fget
        self._name: str | None = None

    def __set_name__(self, owner: type[_OwnerT], name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type[_OwnerT]) -> "lazyval[_OwnerT, _ValueT]":
        ...

    @overload
    def __get__(self, instance: _OwnerT, owner: type[_OwnerT]) -> _ValueT:
        ...

    def __get__(
        self,
        instance: _OwnerT | None,
        owner: type[_OwnerT],
    ) -> _ValueT | "lazyval[_OwnerT, _ValueT]":
        if instance is None:
            return self

        if self._name is None:
            raise AttributeError("lazyval descriptor is missing attribute name")

        value = self._fget(instance)
        vars(instance)[self._name] = value
        return value

    def __set__(self, instance: _OwnerT, value: _ValueT) -> None:
        if self._name is None:
            raise AttributeError("lazyval descriptor is missing attribute name")

        vars(instance)[self._name] = value


class no_default:
    """Sentinel type; this should not be instantiated.

    This type is used so functions can tell the difference between no argument
    passed and an explicit value passed even if ``None`` is a valid value.

    Notes
    -----
    This is implemented as a type to make functions which use this as a default
    argument serializable.
    """

    def __new__(cls) -> "no_default":  # pragma: no cover - construction forbidden
        raise TypeError("cannot create instances of sentinel type")


memoize = lru_cache(None)


def accuracy(count_300: int, count_100: int, count_50: int, count_miss: int) -> float:
    """Calculate osu! standard accuracy from discrete hit counts."""
    points_of_hits = count_300 * 300 + count_100 * 100 + count_50 * 50
    total_hits = count_300 + count_100 + count_50 + count_miss
    return points_of_hits / (total_hits * 300)


def orange(_start_or_stop: Any, *args: Any) -> Iterator[Any]:
    """Range for arbitrary objects."""
    if not args:
        start = 0
        stop = _start_or_stop
        step = 1
    elif len(args) == 1:
        start = _start_or_stop
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start = _start_or_stop
        stop, step = args
    else:
        raise TypeError(
            "orange takes from 1 to 3 positional arguments but"
            f" {len(args) + 1} were given",
        )

    while start < stop:
        yield start
        start += step


ByteBuffer = MutableSequence[int]


def consume_byte(buffer: ByteBuffer) -> int:
    result = buffer[0]
    del buffer[0]
    return result


def consume_short(buffer: ByteBuffer) -> int:
    result = int.from_bytes(buffer[:2], "little")
    del buffer[:2]
    return result


def consume_int(buffer: ByteBuffer) -> int:
    result = int.from_bytes(buffer[:4], "little")
    del buffer[:4]
    return result


def consume_long(buffer: ByteBuffer) -> int:
    result = int.from_bytes(buffer[:8], "little")
    del buffer[:8]
    return result


def consume_uleb128(buffer: ByteBuffer) -> int:
    result = 0
    shift = 0
    while True:
        byte = consume_byte(buffer)
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7

    return result


def consume_string(buffer: ByteBuffer) -> str | None:
    mode = consume_byte(buffer)
    if mode == 0:
        return None
    if mode != 0x0B:
        raise ValueError(
            f"unknown string start byte: {hex(mode)}, expected 0 or 0x0b",
        )
    byte_length = consume_uleb128(buffer)
    data = buffer[:byte_length]
    del buffer[:byte_length]
    return bytes(data).decode("utf-8")


_windows_epoch = datetime.datetime(1, 1, 1)


def consume_datetime(buffer: ByteBuffer) -> datetime.datetime:
    windows_ticks = consume_long(buffer)
    return _windows_epoch + datetime.timedelta(microseconds=windows_ticks / 10)
