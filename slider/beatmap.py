from __future__ import annotations

import re
import math
import inspect
import numpy as np
import operator as op
from zipfile import ZipFile
from functools import partial
from datetime import timedelta
from bisect import bisect_left
from enum import IntEnum, unique
from itertools import chain, cycle, islice
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    Sequence
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    IO,
    List,
    Literal,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from .abc import abstractmethod
from .curve import Curve
from .game_mode import GameMode
from .position import Point, Position, distance
from .mod import ar_to_ms, circle_radius, ms_300_to_od, ms_to_ar, od_to_ms_300
from .events import Event, EventType, EventCollection, Background, Video, Break
from .utils import (
    lazyval,
    memoize,
    no_default,
    orange,
)


T = TypeVar("T")
NoDefaultType = type[no_default]
GroupValue = Union[List[str], Dict[str, str]]
GroupsMapping = Dict[str, GroupValue]
ColorTuple = Tuple[int, int, int]


def _get(cs: Sequence[str], ix: int, default: str | NoDefaultType = no_default) -> str:
    try:
        return cs[ix]
    except IndexError:
        if default is no_default:
            raise
        return cast(str, default)


class TimingPoint:
    """A timing point assigns properties to an offset into a beatmap.

    Parameters
    ----------
    offset : timedelta
        When this ``TimingPoint`` takes effect.
    ms_per_beat : float
        The milliseconds per beat, this is another representation of BPM.
    meter : int
        The number of beats per measure.
    sample_type : int
        The type of hit sound samples that are used.
    sample_set : int
        The set of hit sound samples that are used.
    volume : int
        The volume of hit sounds in the range [0, 100]. This value will be
        clipped if outside the range.
    parent : TimingPoint or None
        The parent of an inherited timing point. An inherited timing point
        differs from a normal timing point in that the ``ms_per_beat`` value is
        negative, and defines a new ``ms_per_beat`` based on the parent
        timing point. This can be used to change volume without affecting
        offset timing, or changing slider speeds. If this is not an inherited
        timing point the parent should be ``None``.
    kiai_mode : bool
        Whether or not kiai time effects are active.
    """

    def __init__(
        self,
        offset: timedelta,
        ms_per_beat: float,
        meter: int,
        sample_type: int,
        sample_set: int,
        volume: int,
        parent: "TimingPoint | None",
        kiai_mode: bool,
    ) -> None:
        self.offset = offset
        self.ms_per_beat = ms_per_beat
        self.meter = meter
        self.sample_type = sample_type
        self.sample_set = sample_set
        self.volume = np.clip(volume, 0, 100)
        self.parent = parent
        self.inherited = parent is not None
        self.kiai_mode = kiai_mode

    @lazyval
    def half_time(self) -> "TimingPoint":
        """The ``TimingPoint`` as it would appear with
        :data:`~slider.mod.Mod.half_time` enabled.
        """
        return type(self)(
            4 * self.offset / 3,
            self.ms_per_beat if self.inherited else (4 * self.ms_per_beat / 3),
            self.meter,
            self.sample_type,
            self.sample_set,
            self.volume,
            getattr(self.parent, "half_time", None),
            self.kiai_mode,
        )

    def double_time(self) -> "TimingPoint":
        """The ``TimingPoint`` as it would appear with
        :data:`~slider.mod.Mod.double_time` enabled.
        """
        return type(self)(
            2 * self.offset / 3,
            self.ms_per_beat if self.inherited else (2 * self.ms_per_beat / 3),
            self.meter,
            self.sample_type,
            self.sample_set,
            self.volume,
            getattr(self.parent, "double_time", None),
            self.kiai_mode,
        )

    @lazyval
    def bpm(self) -> int | None:
        """The bpm of this timing point.

        If this is an inherited timing point this value will be None.
        """
        ms_per_beat = self.ms_per_beat
        if ms_per_beat < 0:
            return None
        return round(60000 / ms_per_beat)

    def __repr__(self) -> str:
        if self.parent is None:
            inherited = "inherited "
        else:
            inherited = ""
        return (
            f"<{type(self).__qualname__}:"
            f" {inherited}{self.offset.total_seconds() * 1000:g}ms>"
        )

    @classmethod
    def parse(cls, data: str, parent: "TimingPoint | None") -> "TimingPoint":
        """Parse a TimingPoint object from a line in a ``.osu`` file.

        Parameters
        ----------
        data : str
            The line to parse.
        parent : TimingPoint
            The last non-inherited timing point.

        Returns
        -------
        timing_point : TimingPoint
            The parsed timing point.

        Raises
        ------
        ValueError
            Raised when ``data`` does not describe a ``TimingPoint`` object.
        """
        try:
            offset_raw, ms_per_beat_raw, *rest = data.split(",")
        except ValueError:
            raise ValueError(
                f"failed to parse {cls.__qualname__} from {data!r}",
            )

        try:
            offset_float = float(offset_raw)
        except ValueError:
            raise ValueError(f"offset should be a float, got {offset_raw!r}")

        offset_td = timedelta(milliseconds=offset_float)

        try:
            ms_per_beat_value = float(ms_per_beat_raw)
        except ValueError:
            raise ValueError(
                f"ms_per_beat should be a float, got {ms_per_beat_raw!r}",
            )

        try:
            meter = int(_get(rest, 0, "4"))
        except ValueError:
            raise ValueError(f"meter should be an int, got {meter!r}")

        try:
            sample_type = int(_get(rest, 1, "0"))
        except ValueError:
            raise ValueError(
                f"sample_type should be an int, got {sample_type!r}",
            )

        try:
            sample_set = int(_get(rest, 2, "0"))
        except ValueError:
            raise ValueError(
                f"sample_set should be an int, got {sample_set!r}",
            )

        try:
            volume = int(_get(rest, 3, "1"))
        except ValueError:
            raise ValueError(f"volume should be an int, got {volume!r}")

        try:
            inherited = not bool(int(_get(rest, 4, "1")))
        except ValueError:
            raise ValueError(f"inherited should be a bool, got {inherited!r}")

        try:
            kiai_mode = bool(int(_get(rest, 5, "0")))
        except ValueError:
            raise ValueError(f"kiai_mode should be a bool, got {kiai_mode!r}")

        return cls(
            offset=offset_td,
            ms_per_beat=ms_per_beat_value,
            meter=meter,
            sample_type=sample_type,
            sample_set=sample_set,
            volume=volume,
            parent=parent if inherited else None,
            kiai_mode=kiai_mode,
        )

    def pack(self) -> str:
        """The string representing this timing point used in ``.osu`` file,
        without trailing ``\\n``.

        Returns
        -------
        packed_str : str
            The packed str of this timing point.

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """
        return ",".join(
            [
                _pack_timedelta("time", self.offset),
                _pack_float("beatLength", self.ms_per_beat),
                _pack_int("meter", self.meter),
                _pack_int("sampleSet", self.sample_type),
                _pack_int("sampleIndex", self.sample_set),
                _pack_int("volume", int(self.volume)),
                "1" if self.parent is None else "0",
                _pack_bool("effects", self.kiai_mode),
            ]
        )


class HitObject:
    """An abstract hit element for osu! standard.

    Parameters
    ----------
    position : Position
        Where this element appears on the screen.
    time : timedelta
        When this element appears in the map.
    hitsound : int
        The hitsound to play when this object is hit.
    addition : str, optional
        Unknown currently.
    new_combo : bool
        Whether this element is the start of a new combo.
    combo_skip : int
        How many combo colors to skip if this element is the start of a new
        combo.
    """

    time_related_attributes: ClassVar[frozenset[str]] = frozenset({"time"})
    # must be set by subclasses
    type_code: ClassVar[int] = 0

    # TODO slider v1.x.x: reconsider argument order and default parameters
    # (defaults only exist right now for backwards compat). similarly for all
    # hitobject subclasses.
    def __init__(
        self,
        position: Position,
        time: timedelta,
        hitsound: int,
        addition: str = "0:0:0:0:",
        new_combo: bool = False,
        combo_skip: int = 0,
    ) -> None:
        self.position = position
        self.time = time
        self.hitsound = hitsound
        self.addition = addition
        self.new_combo = new_combo
        self.combo_skip = combo_skip

        self.ht_enabled = False
        self.dt_enabled = False
        self.hr_enabled = False

    def __repr__(self):
        return (
            f"<{type(self).__qualname__}: {self.position},"
            f" {self.time.total_seconds() * 1000:g}ms>"
        )

    def _time_modify(self, coefficient: float) -> "HitObject":
        """Modify this ``HitObject`` by multiplying time related attributes
        by the ``coefficient``.

        Parameters
        ----------
        coefficient : float
            The coefficient to multiply the ``time_related`` values by.

        Returns
        -------
        modified : HitObject
            The modified hit object.
        """
        time_related_attributes = self.time_related_attributes
        kwargs: dict[str, Any] = {}
        for name in inspect.signature(type(self)).parameters:
            value = getattr(self, name)
            if name in time_related_attributes:
                value *= coefficient
            kwargs[name] = value

        return cast(HitObject, type(self)(**kwargs))

    def _get_type_bits(self) -> int:
        # bit numbers below are zero indexed.

        # type code (bits number 0, 1, 3, and 7)
        bits = self.type_code
        # new combo (bit 2)
        bits |= self.new_combo << 2
        # combo_skip (bits 4, 5, and 6).
        bits |= self.combo_skip << 4
        return bits

    @lazyval
    def half_time(self):
        """The ``HitObject`` as it would appear with
        :data:`~slider.mod.Mod.half_time` enabled.
        """
        if self.ht_enabled:
            return self

        obj = self._time_modify(4 / 3)
        obj.ht_enabled = True
        return obj

    @lazyval
    def double_time(self):
        """The ``HitObject`` as it would appear with
        :data:`~slider.mod.Mod.double_time` enabled.
        """
        if self.dt_enabled:
            return self

        obj = self._time_modify(2 / 3)
        obj.dt_enabled = True
        return obj

    @lazyval
    def hard_rock(self):
        """The ``HitObject`` as it would appear with
        :data:`~slider.mod.Mod.hard_rock` enabled.
        """
        if self.hr_enabled:
            return self

        kwargs: dict[str, Any] = {}
        for name in inspect.signature(type(self)).parameters:
            value = getattr(self, name)
            if name == "position":
                value = Position(value.x, 384 - value.y)
            kwargs[name] = value

        obj = cast(HitObject, type(self)(**kwargs))
        obj.hr_enabled = True
        return obj

    @classmethod
    def parse(
        cls,
        data: str,
        timing_points: Sequence[TimingPoint],
        slider_multiplier: float,
        slider_tick_rate: float,
    ) -> "HitObject":
        """Parse a HitObject object from a line in a ``.osu`` file.

        Parameters
        ----------
        data : str
            The line to parse.
        timing_points : list[TimingPoint]
            The timing points in the map.
        slider_multiplier : float
            The slider multiplier for computing slider end_time and ticks.
        slider_tick_rate : float
            The slider tick rate for computing slider end_time and ticks.

        Returns
        -------
        hit_objects : HitObject
            The parsed hit object. This will be the concrete subclass given
            the type.

        Raises
        ------
        ValueError
            Raised when ``data`` does not describe a ``HitObject`` object.
        """
        try:
            x_raw, y_raw, time_raw, type_raw, hitsound_raw, *rest = data.split(",")
        except ValueError:
            raise ValueError(f"not enough elements in line, got {data!r}")

        try:
            # in old beatmaps (and potentially newer ones which were manually
            # edited?), x and y can be floats (see b/128). Without the
            # secondary float cast, parsing these maps would fail.
            # Lazer casts these to integers
            # (https://github.com/ppy/osu/blob/d4ea57c6607a77abb8a5e2fe55b220d8
            # dfeeb456/osu.Game/Rulesets/Objects/Legacy/ConvertHitObjectParser.
            # cs#L49), so we're still matching in-game positions even though we
            # technically lose precision from the .osu file by casting.
            x = int(float(x_raw))
        except ValueError:
            raise ValueError(f"x should be an int, got {x_raw!r}")

        try:
            y = int(float(y_raw))
        except ValueError:
            raise ValueError(f"y should be an int, got {y_raw!r}")

        try:
            time_ms = int(time_raw)
        except ValueError:
            raise ValueError(f"time should be an int, got {time_raw!r}")
        time = timedelta(milliseconds=time_ms)

        try:
            type_code = int(type_raw)
        except ValueError:
            raise ValueError(f"type should be an int, got {type_raw!r}")

        try:
            hitsound = int(hitsound_raw)
        except ValueError:
            raise ValueError(f"hitsound should be an int, got {hitsound_raw!r}")

        parser: Callable[
            [Position, timedelta, int, bool, int, Sequence[str]],
            HitObject,
        ]

        if type_code & Circle.type_code:
            parser = Circle._parse
        elif type_code & Slider.type_code:
            parser = cast(
                Callable[
                    [Position, timedelta, int, bool, int, Sequence[str]],
                    HitObject,
                ],
                partial(
                    Slider._parse,
                    timing_points=timing_points,
                    slider_multiplier=slider_multiplier,
                    slider_tick_rate=slider_tick_rate,
                ),
            )
        elif type_code & Spinner.type_code:
            parser = Spinner._parse
        elif type_code & HoldNote.type_code:
            parser = HoldNote._parse
        else:
            raise ValueError(f"unknown type code {type_code!r}")

        # new combo info is in second bit (0-indexed)
        new_combo = bool(type_code & 0b00000100)
        # 3 bit int for combo skip is held in 4th, 5th, and 6th bits
        combo_skip = (type_code & 0b01110000) >> 4
        return parser(Position(x, y), time, hitsound, new_combo, combo_skip, rest)

    @abstractmethod
    def pack(self) -> str:
        """The string representing this hit element used in .osu file,
        without trailing ``\\n``.

        Returns
        -------
        packed_str : str
            The packed str of this hit element.

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """
        raise NotImplementedError("pack")


class Circle(HitObject):
    """A circle hit element.

    Parameters
    ----------
    position : Position
        Where this circle appears on the screen.
    time : timedelta
        When this circle appears in the map.
    hitsound : int
        The hitsound to play when this object is hit.
    new_combo : bool
        Whether this circle is the start of a new combo.
    combo_skip : int
        How many combo colors to skip if this circle is the start of a new
        combo.
    """

    type_code = 1

    @classmethod
    def _parse(
        cls,
        position: Position,
        time: timedelta,
        hitsound: int,
        new_combo: bool,
        combo_skip: int,
        rest: Sequence[str],
    ) -> "Circle":
        rest_values = list(rest)
        if len(rest_values) > 1:
            raise ValueError(f"extra data: {rest_values!r}")

        addition = rest_values[0] if rest_values else "0:0:0:0:"
        return cls(position, time, hitsound, addition, new_combo, combo_skip)

    def pack(self) -> str:
        """The string representing this circle hit element used in ``.osu`` file,
        without trailing ``\\n``.

        Returns
        -------
        packed_str : str
            The packed str of this circle hit element.

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """
        # Circles do not have objectParams
        return ",".join(
            [
                _pack_float("x", self.position.x),
                _pack_float("y", self.position.y),
                _pack_timedelta("time", self.time),
                _pack_int("type", self._get_type_bits()),
                _pack_int("hitSound", self.hitsound),
                _pack_str("hitSample", self.addition),
            ]
        )


class Spinner(HitObject):
    """A spinner hit element

    Parameters
    ----------
    position : Position
        Where this spinner appears on the screen.
    time : timedelta
        When this spinner appears in the map.
    hitsound : int
        The hitsound to play when this object is hit.
    end_time : timedelta
        When this spinner ends in the map.
    addition : str
        Hitsound additions.
    new_combo : bool
        Whether this spinner is the start of a new combo.
    combo_skip : int
        How many combo colors to skip if this spinner is the start of a new
        combo.
    """

    type_code = 8
    time_related_attributes = frozenset({"time", "end_time"})

    def __init__(
        self,
        position: Position,
        time: timedelta,
        hitsound: int,
        end_time: timedelta,
        addition: str = "0:0:0:0:",
        new_combo: bool = False,
        combo_skip: int = 0,
    ) -> None:
        super().__init__(position, time, hitsound, addition, new_combo, combo_skip)
        self.end_time = end_time

    @classmethod
    def _parse(
        cls,
        position: Position,
        time: timedelta,
        hitsound: int,
        new_combo: bool,
        combo_skip: int,
        rest: Sequence[str],
    ) -> "Spinner":
        rest_values = list(rest)
        try:
            end_time_raw, *rest_values = rest_values
        except ValueError:
            raise ValueError("missing end_time")

        try:
            end_time_ms = int(end_time_raw)
        except ValueError:
            raise ValueError(f"end_time should be an int, got {end_time_raw!r}")

        end_time = timedelta(milliseconds=end_time_ms)

        if len(rest_values) > 1:
            raise ValueError(f"extra data: {rest_values!r}")

        addition = rest_values[0] if rest_values else "0:0:0:0:"
        return cls(position, time, hitsound, end_time, addition, new_combo, combo_skip)

    def pack(self) -> str:
        """The string representing this spinner hit element used in ``.osu`` file,
        without trailing ``\\n``.

        Returns
        -------
        packed_str : str
            The packed str of this spinner hit element.

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """
        return ",".join(
            [
                _pack_float("x", self.position.x),
                _pack_float("y", self.position.y),
                _pack_timedelta("time", self.time),
                _pack_int("type", self._get_type_bits()),
                _pack_int("hitSound", self.hitsound),
                _pack_timedelta("endTime", self.end_time),
                _pack_str("hitSample", self.addition),
            ]
        )


class Slider(HitObject):
    """A slider hit element.

    Parameters
    ----------
    position : Position
        Where this slider appears on the screen.
    time : datetime.timedelta
        When this slider appears in the map.
    end_time : datetime.timedelta
        When this slider ends in the map
    hitsound : int
        The sound played on the ticks of the slider.
    curve : Curve
        The slider's curve function.
    repeat : int
        The number of times the slider will repeat.
    length : float
        The length of this slider in osu! pixels.
    ticks : int
        The number of slider ticks including the head and tail of the slider.
    num_beats : int
        The number of beats that this slider spans.
    tick_rate : float
        The rate at which ticks appear along sliders.
    ms_per_beat : int
        The milliseconds per beat during the segment of the beatmap that this
        slider appears in.
    edge_sounds : list[int]
        A list of hitsounds for each edge.
    edge_additions : list[str]
        A list of additions for each edge.
    addition : str
        Hitsound additions.
    new_combo : bool
        Whether this slider is the start of a new combo.
    combo_skip : int
        How many combo colors to skip if this slider is the start of a new
        combo.
    """

    type_code = 2
    time_related_attributes = frozenset({"time", "end_time", "ms_per_beat"})
    LEGACY_LAST_TICK_OFFSET = timedelta(milliseconds=36)

    def __init__(
        self,
        position: Position,
        time: timedelta,
        end_time: timedelta,
        hitsound: int,
        curve: Curve,
        repeat: int,
        length: float,
        ticks: int,
        num_beats: float,
        tick_rate: float,
        ms_per_beat: float,
        edge_sounds: Sequence[int],
        edge_additions: Sequence[str],
        addition: str = "0:0:0:0:",
        new_combo: bool = False,
        combo_skip: int = 0,
    ) -> None:
        super().__init__(position, time, hitsound, addition, new_combo, combo_skip)
        self.end_time = end_time
        self.curve = curve
        self.repeat = repeat
        self.length = length
        self.ticks = ticks
        self.num_beats = num_beats
        self.tick_rate = tick_rate
        self.ms_per_beat = ms_per_beat
        self.edge_sounds: List[int] = list(edge_sounds)
        self.edge_additions: List[str] = list(edge_additions)

    @lazyval
    def tick_points(self) -> List[Point]:
        """The position and time of each slider tick."""
        repeat = self.repeat

        time = self.time
        repeat_duration = (self.end_time - time) / repeat

        curve = self.curve

        pre_repeat_ticks: List[Point] = []
        append_tick = pre_repeat_ticks.append

        beats_per_repeat = self.num_beats / repeat
        for t in orange(self.tick_rate, beats_per_repeat, self.tick_rate):
            pos = curve(t / beats_per_repeat)
            timediff = timedelta(milliseconds=t * self.ms_per_beat)
            append_tick(Point(pos.x, pos.y, time + timediff))

        pos = curve(1)
        timediff = repeat_duration
        append_tick(Point(pos.x, pos.y, time + timediff))

        mirrored_points: Sequence[Union[Point, Position]] = cast(
            Sequence[Union[Point, Position]],
            list(chain(pre_repeat_ticks[-2::-1], [self.position])),
        )
        mirrored_positions: List[Position] = [
            Position(point.x, point.y) for point in mirrored_points
        ]
        repeat_ticks: List[Point] = [
            Point(pos.x, pos.y, pre_repeat_tick.offset)
            for pre_repeat_tick, pos in zip(pre_repeat_ticks, mirrored_positions)
        ]

        tick_sequences = islice(
            cycle([pre_repeat_ticks, repeat_ticks]),
            repeat,
        )
        return list(
            chain.from_iterable(
                (Point(p.x, p.y, p.offset + n * repeat_duration) for p in tick_sequence)
                for n, tick_sequence in enumerate(tick_sequences)
            ),
        )

    @lazyval
    def true_tick_points(self) -> List[Point]:
        """The position and time of each slider tick. This accounts for
        the legacy last tick offset.

        See also https://github.com/ppy/osu/blob/5a1940facf2649bc6b2892965c27b8
        c62a41988f/osu.Game/Rulesets/Objects/SliderEventGenerator.cs#L71-L93"""
        tick_points = self.tick_points
        # curve() takes in a percentage of how far along we want the point.
        # Take away the offset from the total length of the slider to get
        # the percentage of the slider we want the point at.
        true_end_time = self.end_time - self.LEGACY_LAST_TICK_OFFSET

        # keep in mind to check for negative values here, e.g. at least
        # have the duration be 0
        legacy_duration = max(true_end_time - self.time, timedelta(milliseconds=0))
        real_duration = self.end_time - self.time

        ratio = legacy_duration / real_duration
        curve_point = int(self.length * ratio)
        pos = self.curve(curve_point / self.length)

        tick_points[-1] = Point(pos.x, pos.y, true_end_time)

        return tick_points

    @lazyval
    def hard_rock(self) -> "Slider":
        """The ``HitObject`` as it would appear with
        :data:`~slider.mod.Mod.hard_rock` enabled.
        """
        if self.hr_enabled:
            return self

        kwargs: dict[str, Any] = {}
        for name in inspect.signature(type(self)).parameters:
            value = getattr(self, name)
            if name == "position":
                value = Position(value.x, 384 - value.y)
            elif name == "curve":
                value = value.hard_rock
            kwargs[name] = value
        obj = cast(Slider, type(self)(**kwargs))
        obj.hr_enabled = True
        return obj

    @classmethod
    def _parse(
        cls,
        position: Position,
        time: timedelta,
        hitsound: int,
        new_combo: bool,
        combo_skip: int,
        rest: Sequence[str],
        *,
        timing_points: Sequence[TimingPoint],
        slider_multiplier: float,
        slider_tick_rate: float,
    ) -> "Slider":
        rest_list = list(rest)
        try:
            group_1, *rest_list = rest_list
        except ValueError:
            raise ValueError(f"missing required slider data in {rest_list!r}")

        try:
            slider_type, *raw_points = group_1.split("|")
        except ValueError:
            raise ValueError(
                "expected slider type and points in the first"
                f" element of rest, {rest!r}",
            )

        points = [position]
        for point in raw_points:
            try:
                x_str, y_str = point.split(":")
            except ValueError:
                raise ValueError(
                    f"expected points in the form x:y, got {point!r}",
                )

            try:
                x_coord = int(float(x_str))
            except ValueError:
                raise ValueError(f"x should be an int, got {x_str!r}")

            try:
                y_coord = int(float(y_str))
            except ValueError:
                raise ValueError(f"y should be an int, got {y_str!r}")

            points.append(Position(x_coord, y_coord))

        try:
            repeat_raw, *rest_list = rest_list
        except ValueError:
            raise ValueError(f"missing repeat in {rest_list!r}")

        try:
            repeat = int(repeat_raw)
        except ValueError:
            raise ValueError(f"repeat should be an int, got {repeat_raw!r}")

        try:
            pixel_length_raw, *rest_list = rest_list
        except ValueError:
            raise ValueError(f"missing pixel_length in {rest_list!r}")

        try:
            pixel_length = float(pixel_length_raw)
        except ValueError:
            raise ValueError(
                f"pixel_length should be a float, got {pixel_length_raw!r}",
            )

        try:
            raw_edge_sounds_grouped, *rest_list = rest_list
        except ValueError:
            raw_edge_sounds_grouped = ""

        raw_edge_sounds = raw_edge_sounds_grouped.split("|")
        edge_sounds: List[int] = []
        if raw_edge_sounds != [""]:
            for edge_sound in raw_edge_sounds:
                try:
                    edge_sound_value = int(edge_sound)
                except ValueError:
                    raise ValueError(
                        f"edge_sound should be an int, got {edge_sound!r}",
                    )
                edge_sounds.append(edge_sound_value)

        try:
            edge_additions_grouped, *rest_list = rest_list
        except ValueError:
            edge_additions_grouped = ""

        if edge_additions_grouped:
            edge_additions_list: List[str] = edge_additions_grouped.split("|")
        else:
            edge_additions_list = []

        if len(rest_list) > 1:
            raise ValueError(f"extra data: {rest_list!r}")

        addition = rest_list[0] if rest_list else "0:0:0:0:"

        for tp in reversed(timing_points):
            if tp.offset <= time:
                break
        else:
            tp = timing_points[0]

        if tp.parent is not None:
            velocity_multiplier = np.clip(-100 / tp.ms_per_beat, 0.1, 10)
            ms_per_beat = tp.parent.ms_per_beat
        else:
            velocity_multiplier = 1
            ms_per_beat = tp.ms_per_beat

        pixels_per_beat = slider_multiplier * 100 * velocity_multiplier
        num_beats = (pixel_length * repeat) / pixels_per_beat

        if math.isnan(num_beats):
            # In really rare cases, this value can be "NaN"
            # I'm not sure if this is the correct way to handle this
            num_beats = 0

        duration = timedelta(milliseconds=int(num_beats * ms_per_beat))

        ticks = int(
            ((np.ceil((num_beats - 0.1) / repeat * slider_tick_rate) - 1)) * repeat
            + repeat
            + 1
        )

        return cls(
            position,
            time,
            time + duration,
            hitsound,
            Curve.from_kind_and_points(slider_type, points, pixel_length),
            repeat,
            pixel_length,
            ticks,
            num_beats,
            slider_tick_rate,
            ms_per_beat,
            edge_sounds,
            edge_additions_list,
            addition,
            new_combo=new_combo,
            combo_skip=combo_skip,
        )

    def pack(self) -> str:
        """The string representing this slider hit element used in ``.osu`` file,
        without trailing ``\\n``.

        Returns
        -------
        packed_str : str
            The packed str of this slider hit element.

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """
        return ",".join(
            [
                _pack_float("x", self.position.x),
                _pack_float("y", self.position.y),
                _pack_timedelta("time", self.time),
                _pack_int("type", self._get_type_bits()),
                _pack_int("hitSound", self.hitsound),
                self.curve.pack(),
                _pack_int("slides", self.repeat),
                _pack_float("length", self.length),
                "|".join(
                    _pack_int("edgeSound", edge_sound)
                    for edge_sound in self.edge_sounds
                ),
                "|".join(
                    _pack_str("edgeSet", edge_addition)
                    for edge_addition in self.edge_additions
                ),
                _pack_str("hitSample", self.addition),
            ]
        )


class HoldNote(HitObject):
    """A HoldNote hit element.

    Parameters
    ----------
    position : Position
        Where this HoldNote appears on the screen.
    time : timedelta
        When this HoldNote appears in the map.
    new_combo : bool
        Whether this HoldNote is the start of a new combo.
    combo_skip : int
        How many combo colors to skip if this HoldNote is the start of a new
        combo.

    Notes
    -----
    A ``HoldNote`` can only appear in an osu!mania map.
    """

    type_code = 128
    time_related_attributes = frozenset({"time", "end_time"})

    def __init__(
        self,
        position: Position,
        time: timedelta,
        hitsound: int,
        end_time: timedelta,
        addition: str = "0:0:0:0:",
        new_combo: bool = False,
        combo_skip: int = 0,
    ) -> None:
        super().__init__(position, time, hitsound, addition, new_combo, combo_skip)
        self.end_time = end_time

    @classmethod
    def _parse(
        cls,
        position: Position,
        time: timedelta,
        hitsound: int,
        new_combo: bool,
        combo_skip: int,
        rest: Sequence[str],
    ) -> "HoldNote":
        rest_values = list(rest)
        try:
            end_time_raw, *rest_values = rest_values
        except ValueError:
            raise ValueError("missing end_time")

        if ":" in end_time_raw:
            # Some maps seem to have extra data, e.g. "15495:0:0:0:0:"
            end_time_raw, *_ = end_time_raw.split(":")

        try:
            end_time_ms = int(end_time_raw)
        except ValueError:
            raise ValueError(f"end_time should be an int, got {end_time_raw!r}")
        end_time = timedelta(milliseconds=end_time_ms)
        if len(rest_values) > 1:
            raise ValueError(f"extra data: {rest_values!r}")

        addition = rest_values[0] if rest_values else "0:0:0:0:"
        return cls(position, time, hitsound, end_time, addition, new_combo, combo_skip)

    def pack(self):
        """The string representing this HoldNote hit element used in ``.osu`` file,
        without trailing ``\\n``.

        Returns
        -------
        packed_str : str
            The packed str of this HoldNote hit element.

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """
        # HoldNotes differ with Sliders in that their endTime is
        # joined with hitSample with ':' rather than with ','
        return ",".join(
            [
                _pack_int("x", self.position.x),
                _pack_int("y", self.position.y),
                _pack_timedelta("time", self.time),
                _pack_int("type", self._get_type_bits()),
                _pack_int("hitSound", self.hitsound),
                ":".join(
                    [
                        _pack_timedelta("endTime", self.end_time),
                        _pack_str("hitSample", self.addition),
                    ]
                ),
            ]
        )


@overload
def _get_as_str(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
) -> str: ...


@overload
def _get_as_str(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T,
) -> str | T: ...


def _get_as_str(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T | NoDefaultType = no_default,
) -> str | T:
    """Lookup a field from a given section.

    Parameters
    ----------
    groups : dict[str, dict[str, str]]
        The grouped osu! file.
    section : str
        The section to read from.
    field : str
        The field to read.
    default : int, optional
        A value to return if ``field`` is not in ``groups[section]``.

    Returns
    -------
    cs : str
        ``groups[section][field]`` or default if ``field` is not in
         ``groups[section]``.
    """
    try:
        mapping = groups[section]
    except KeyError:
        if default is no_default:
            raise ValueError(f"missing section {section!r}")
        return cast(T, default)

    try:
        return mapping[field]
    except KeyError:
        if default is no_default:
            raise ValueError(f"missing field {field!r} in section {section!r}")
        return cast(T, default)


@overload
def _get_as_int(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
) -> int: ...


@overload
def _get_as_int(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T,
) -> int | T: ...


def _get_as_int(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T | NoDefaultType = no_default,
) -> int | T:
    """Lookup a field from a given section and parse it as an integer.

    Parameters
    ----------
    groups : dict[str, dict[str, str]]
        The grouped osu! file.
    section : str
        The section to read from.
    field : str
        The field to read and parse.
    default : int, optional
        A value to return if ``field`` is not in ``groups[section]``.

    Returns
    -------
    integer : int
        ``int(groups[section][field])`` or default if ``field` is not in
        ``groups[section]``.
    """
    v = _get_as_str(groups, section, field, default)

    if v is default:
        return cast(T, v)

    str_v = cast(str, v)

    try:
        return int(str_v)
    except ValueError:
        raise ValueError(
            f"field {field!r} in section {section!r} should be an int," f" got {v!r}",
        )


@overload
def _get_as_int_list(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
) -> List[int]: ...


@overload
def _get_as_int_list(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T,
) -> List[int] | T: ...


def _get_as_int_list(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T | NoDefaultType = no_default,
) -> List[int] | T:
    """Lookup a field from a given section and parse it as an integer list.

    Parameters
    ----------
    groups : dict[str, dict[str, str]]
        The grouped osu! file.
    section : str
        The section to read from.
    field : str
        The field to read and parse.
    default : int, optional
        A value to return if ``field`` is not in ``groups[section]``.

    Returns
    -------
    ints : list[int]
        ``int(groups[section][field])`` or default if ``field` is not in
        ``groups[section]``.
    """
    v = _get_as_str(groups, section, field, default)

    if v is default:
        return cast(T, v)

    str_v = cast(str, v)

    if not str_v:
        return []

    try:
        return [int(e.strip()) for e in str_v.split(",")]
    except ValueError:
        raise ValueError(
            f"field {field!r} in section {section!r} should be an int list,"
            f" got {v!r}",
        )


@overload
def _get_as_float(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
) -> float: ...


@overload
def _get_as_float(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T,
) -> float | T: ...


def _get_as_float(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T | NoDefaultType = no_default,
) -> float | T:
    """Lookup a field from a given section and parse it as an float

    Parameters
    ----------
    groups : dict[str, dict[str, str]]
        The grouped osu! file.
    section : str
        The section to read from.
    field : str
        The field to read and parse.
    default : float, optional
        A value to return if ``field`` is not in ``groups[section]``.

    Returns
    -------
    f : float
        ``float(groups[section][field])`` or default if ``field` is not in
        ``groups[section]``.
    """
    v = _get_as_str(groups, section, field, default)

    if v is default:
        return cast(T, v)

    str_v = cast(str, v)

    try:
        return float(str_v)
    except ValueError:
        raise ValueError(
            f"field {field!r} in section {section!r} should be a float," f" got {v!r}",
        )


@overload
def _get_as_bool(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
) -> bool: ...


@overload
def _get_as_bool(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T,
) -> bool | T: ...


def _get_as_bool(
    groups: Mapping[str, Mapping[str, str]],
    section: str,
    field: str,
    default: T | NoDefaultType = no_default,
) -> bool | T:
    """Lookup a field from a given section and parse it as an float

    Parameters
    ----------
    groups : dict[str, dict[str, str]]
        The grouped osu! file.
    section : str
        The section to read from.
    field : str
        The field to read and parse.
    default : float, optional
        A value to return if ``field`` is not in ``groups[section]``.

    Returns
    -------
    f : float
        ``float(groups[section][field])`` or default if ``field` is not in
        ``groups[section]``.
    """
    v = _get_as_str(groups, section, field, default)

    if v is default:
        return cast(T, v)

    str_v = cast(str, v)

    try:
        # cast to int then to bool because '0' is still True; bools are written
        # to the file as '0' and '1' so this is safe.
        return bool(int(str_v))
    except ValueError:
        raise ValueError(
            f"field {field!r} in section {section!r} should be a bool," f" got {v!r}",
        )


def _invalid_to_default(
    field: str,
    field_value: Any,
    expected_type: type[Any] | tuple[type[Any], ...],
    default: Any | NoDefaultType = no_default,
) -> Any:
    """
    Replaces the field_value with default value if it is invalid
    (missing or of incorrect type).

    Parameters
    ----------
    field : str
        The name of the field.
    field_value : Any
        Field value.
    expected_type : Any
        The expected type of ``field_value``.
    default : float, optional
        A value to return if ``field`` is invalid.

    Returns
    -------
    field_value : Any
        Valid ``field_value``

    Raises
    ------
    ValueError
        Raised when ``field_value`` is missing or is of incorrect type,
        but no default value is available.
    """
    if isinstance(field_value, expected_type):
        return field_value

    if default is no_default:
        if isinstance(expected_type, tuple):
            expected_name = ", ".join(t.__name__ for t in expected_type)
        else:
            expected_name = expected_type.__name__
        raise ValueError(
            f"field {field!r} should be a {expected_name!r},"
            f" got {field_value.__class__.__name__!r}",
        )

    return default


def _pack_timedelta(
    field: str,
    td: timedelta,
    default: timedelta | NoDefaultType = no_default,
) -> str:
    """Pack timedelta to a string.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    td : timedelta
        The value to be packed to string.
    default : timedelta, optional
        A value to return if ``td`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``td`` is not a timedelta and default is not available.
    """
    td = _invalid_to_default(field, td, timedelta, default)
    return str(td // timedelta(milliseconds=1))


def _pack_bool(
    field: str,
    bool_in: bool,
    default: bool | NoDefaultType = no_default,
) -> str:
    """Pack bool to a string.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    bool_in : bool
        The value to be packed to string.
    default : bool, optional
        A value to return if ``bool_in`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``bool_in`` is not a bool and default is not available.
    """
    bool_in = _invalid_to_default(field, bool_in, bool, default)
    return "1" if bool_in else "0"


def _pack_int(
    field: str,
    int_in: int,
    default: int | NoDefaultType = no_default,
) -> str:
    """Pack int to a string.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    int_in : int
        The value to be packed to string.
    default : int, optional
        A value to return if ``int_in`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``int_in`` is not a int and default is not available.
    """
    int_in = _invalid_to_default(field, int_in, int, default)
    return str(int(int_in))


def _pack_float(
    field: str,
    float_in: float | int,
    default: float | int | NoDefaultType = no_default,
) -> str:
    """Pack float to a string. If the float number can be converted to
    int without loss, return the packed string of the converted int.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    float_in : float
        The value to be packed to string.
    default : float, optional
        A value to return if ``float_in`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``float_in`` is not a float and default is not available.
    """
    float_in = _invalid_to_default(field, float_in, (int, float), default)
    # try to give out an int-like string when packing float fields,
    # as osu! client does
    int_ = int(float_in)
    return str(int_) if int_ == float_in else str(float_in)


def _pack_str(
    field: str,
    str_in: str,
    default: str | NoDefaultType = no_default,
) -> str:
    """Pack string to a string, with validity check.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    str_in : str
        The value to be packed to string.
    default : str, optional
        A value to return if ``str_in`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``str_in`` is not a str and default is not available.
    """
    str_in = _invalid_to_default(field, str_in, str, default)
    return str_in


def _pack_int_enum(
    field: str,
    enum_in: IntEnum,
    default: IntEnum | NoDefaultType = no_default,
) -> str:
    """Pack IntEnum to a string.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    enum_in : IntEnum
        The value to be packed to string.
    default : IntEnum, optional
        A value to return if ``enum_in`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``enum_in`` is not a IntEnum and default is not available.
    """
    enum_in = _invalid_to_default(field, enum_in, IntEnum, default)
    return str(int(enum_in))


def _pack_str_list(
    field: str,
    list_str: Sequence[str],
    sep: str = " ",
    default: Sequence[str] | NoDefaultType = no_default,
) -> str:
    """Pack a list of string to a string, with `sep` as separator
    between elements.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    list_str : list
        The value to be packed to string.
    sep : str
        separator to join packed strings of elements
    default : list, optional
        A value to return if ``list_str`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``list_str`` is not a list of str
        and default is not available.
    """
    list_str = cast(List[str], _invalid_to_default(field, list_str, list, default))
    return sep.join(list_str)


def _pack_timedelta_list(
    field: str,
    list_td: Sequence[timedelta],
    sep: str = ",",
    default: Sequence[timedelta] | NoDefaultType = no_default,
) -> str:
    """Pack a list of timedelta to a string, with `sep` as separator
    between elements.

    Parameters
    ----------
    field : str
        The name  of the field to be packed.
    list_td : list
        The value to be packed to string.
    sep : str
        separator to join packed strings of elements
    default : list, optional
        A value to return if ``list_td`` is not valid.

    Returns
    -------
    packed_str : str
        The packed str.

    Raises
    ------
    ValueError
        Raised when ``list_td`` is not a list of timedelta
        and default is not available.
    """
    list_td = cast(List[timedelta], _invalid_to_default(field, list_td, list, default))
    return sep.join(str(td // timedelta(milliseconds=1)) for td in list_td)


def _moving_average_by_time(
    times: np.ndarray,
    data: np.ndarray,
    delta: float | int,
    num: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take the moving average of some values and sample it at regular
    frequencies.

    Parameters
    ----------
    times : np.ndarray
        The array of times to use in the average.
    data : np.ndarray
        The array of values to take the average of. Each column is averaged
        independently.
    delta : int or float
        The length of the leading and trailing window in seconds
    num : int
        The number of samples to take.

    Returns
    -------
    times : np.ndarray
        A column vector of the times sampled at.
    averages : np.ndarray
        A column array of the averages. 1 column per column in the input
    """

    # take an even sample from 0 to the end time
    out_times = np.linspace(
        times[0].item(),
        times[-1].item(),
        num,
        dtype="timedelta64[ns]",
    )
    delta_ns = np.timedelta64(int(delta * 1e9), "ns")

    # compute the start and stop indices for each sampled window
    window_start_ixs = np.searchsorted(times[:, 0], out_times - delta_ns)
    window_stop_ixs = np.searchsorted(times[:, 0], out_times + delta_ns)

    # a 2d array of shape ``(num, 2)`` where each row holds the start and stop
    # index for the window
    window_ixs = np.stack([window_start_ixs, window_stop_ixs], axis=1)

    # append a nan to the end of the values so that we can do many slices all
    # the way to the end in reduceat
    values = np.vstack([data, [np.nan] * data.shape[1]])

    # sum the values in the ranges ``[window_start_ixs, window_stop_ixs)``
    window_sums = np.add.reduceat(values, window_ixs.ravel())[::2]
    window_sizes = np.diff(window_ixs, axis=1).ravel()
    # convert window_sizes of 0 to 1 (inplace) to prevent division by zero
    np.clip(window_sizes, 1, None, out=window_sizes)

    out_values = np.stack(window_sums / window_sizes.reshape((-1, 1)))

    return out_times.reshape((-1, 1)), out_values


def _calculate_accuracy_array(
    count_300: np.ndarray,
    count_100: np.ndarray,
    count_50: np.ndarray,
    count_miss: np.ndarray,
) -> np.ndarray:
    """Vectorized variant of :func:`slider.utils.accuracy`."""
    total_hits = count_300 + count_100 + count_50 + count_miss
    points_of_hits = (count_300 * 300) + (count_100 * 100) + (count_50 * 50)
    return points_of_hits / (total_hits * 300)


class _DifficultyHitObject:
    """An object used to accumulate the strain information for calculating
    stars.

    Parameters
    ----------
    hit_object : HitObject
        The hit object to wrap.
    radius : int
        The circle radius
    previous : _DifficultyHitObject, optional
        The previous difficulty hit object.
    """

    decay_base: Tuple[float, float] = (0.3, 0.15)

    almost_diameter = 90

    stream_spacing = 110
    single_spacing = 125

    weight_scaling: Tuple[float, float] = (1400, 26.25)

    circle_size_buffer_threshold = 30

    @unique
    class Strain(IntEnum):
        """Indices for the strain specific values."""

        speed = 0
        aim = 1

    def __init__(
        self,
        hit_object: HitObject,
        radius: float,
        previous: "_DifficultyHitObject | None" = None,
    ) -> None:
        self.hit_object = hit_object

        scaling_factor = 52 / radius
        if radius < self.circle_size_buffer_threshold:
            scaling_factor *= (
                1
                + min(
                    self.circle_size_buffer_threshold - radius,
                    5,
                )
                / 50
            )

        # this currently ignores slider length
        self.normalized_start = self.normalized_end = Position(
            hit_object.position.x * scaling_factor,
            hit_object.position.y * scaling_factor,
        )

        if previous is None:
            self.strains = (0.0, 0.0)
        else:
            self.strains = (
                self._calculate_strain(previous, self.Strain.speed),
                self._calculate_strain(previous, self.Strain.aim),
            )

    def _calculate_strain(
        self,
        previous: "_DifficultyHitObject",
        strain: "_DifficultyHitObject.Strain",
    ) -> float:
        result = 0.0
        scaling = self.weight_scaling[strain]

        hit_object = self.hit_object
        if isinstance(hit_object, (Circle, Slider)):
            result = self._spacing_weight(self._distance(previous), strain) * scaling

        time_elapsed = (
            self.hit_object.time - previous.hit_object.time
        ).total_seconds() * 1000
        result /= max(time_elapsed, 50)
        decay = self.decay_base[strain] ** (time_elapsed / 1000)
        return previous.strains[strain] * decay + result

    def _distance(self, previous: "_DifficultyHitObject") -> float:
        """The magnitude of distance between the current object and the
        previous.

        Parameters
        ----------
        previous : _DifficultyHitObject
            The previous difficulty hit object.

        Returns
        -------
        distance : float
            The absolute difference between the two hit objects.
        """
        start = self.normalized_start
        end = previous.normalized_end
        return float(np.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2))

    def _spacing_weight(
        self,
        distance: float,
        strain: "_DifficultyHitObject.Strain",
    ) -> float:
        if strain == self.Strain.speed:
            if distance > self.single_spacing:
                return 2.5
            elif distance > self.stream_spacing:
                return 1.6 + 0.9 * (distance - self.stream_spacing) / (
                    self.single_spacing - self.stream_spacing
                )
            elif distance > self.almost_diameter:
                return 1.2 + 0.4 * (distance - self.almost_diameter) / (
                    self.stream_spacing - self.almost_diameter
                )
            elif distance > self.almost_diameter / 2:
                return 0.95 + 0.25 * (distance - self.almost_diameter / 2) / (
                    self.almost_diameter / 2.0
                )
            return 0.95

        return distance**0.99


class Beatmap:
    """A beatmap for osu! standard.

    Parameters
    ----------
    format_version : int
        The version of the beatmap file.
    audio_filename : str
        The location of the audio file relative to the unpacked ``.osz``
        directory.
    audio_lead_in : timedelta
        The amount of time added before the audio file begins playing. Useful
        selection menu.
    preview_time : timedelta
        When the audio file should begin playing when selected in the song for
        audio files that begin immediately.
    countdown : bool
        Should the countdown be displayed before the first hit object.
    sample_set : str
        The set of hit sounds to use through the beatmap.
    stack_leniency : float
        How often closely placed hit objects will be placed together.
    mode : GameMode
        The game mode.
    letterbox_in_breaks : bool
        Should the letterbox appear during breaks.
    widescreen_storyboard : bool
        Should the storyboard be widescreen?
    bookmarks : list[timedelta]
        The time for all of the bookmarks.
    distance_spacing : float
        A multiplier for the 'distance snap' feature.
    beat_divisor : int
        The beat division for placing objects.
    grid_size : int
        The size of the grid for the 'grid snap' feature.
    timeline_zoom : float
        The zoom in the editor timeline.
    title : str
        The title of the song limited to ascii characters.
    title_unicode : str
        The title of the song with unicode support.
    artist : str
        The name of the song artist limited to ascii characters.
    artist_unicode : str
        The name of the song artist with unicode support.
    creator : str
        The username of the mapper.
    version : str
        The name of the beatmap's difficulty.
    source : str
        The origin of the song.
    tags : list[str]
        A collection of words describing the song. This is searchable on the
        osu! website.
    beatmap_id : int or None
        The id of this single beatmap. Old beatmaps did not store this in the
        file.
    beatmap_set_id : int or None
        The id of this beatmap set. Old beatmaps did not store this in the
        file.
    hp_drain_rate : float
        The ``HP`` attribute of the beatmap.
    circle_size, : float
        The ``CS`` attribute of the beatmap.
    overall_difficulty : float
        The ``OD`` attribute of the beatmap.
    approach_rate : float
        The ``AR`` attribute of the beatmap.
    slider_multiplier : float
        The multiplier for slider velocity.
    slider_tick_rate : float
        How often slider ticks appear.
    combo_colors : list[tuple[int, int, int]]
        The combo colors from the ``[Colours]`` section.
    timing_points : list[TimingPoint]
        The timing points the the map.
    hit_objects : list[HitObject]
        The hit objects in the map.
    events : list[Event]
        The events in the map.

    Notes
    -----
    This is currently missing the storyboard data.
    """

    _version_regex = re.compile(r"^osu file format v(\d+)$")

    def __init__(
        self,
        *,
        format_version: int,
        audio_filename: str,
        audio_lead_in: timedelta,
        preview_time: timedelta,
        countdown: bool,
        sample_set: str,
        stack_leniency: float,
        mode: GameMode,
        letterbox_in_breaks: bool,
        widescreen_storyboard: bool,
        bookmarks: Sequence[timedelta],
        distance_spacing: float,
        beat_divisor: int,
        grid_size: int,
        timeline_zoom: float,
        title: str,
        title_unicode: str,
        artist: str,
        artist_unicode: str,
        creator: str,
        version: str,
        source: str | None,
        tags: Sequence[str],
        beatmap_id: int | None,
        beatmap_set_id: int | None,
        hp_drain_rate: float,
        circle_size: float,
        overall_difficulty: float,
        approach_rate: float,
        slider_multiplier: float,
        slider_tick_rate: float,
        combo_colors: Sequence[ColorTuple],
        timing_points: Sequence[TimingPoint],
        hit_objects: Sequence[HitObject],
        events: EventCollection
    ) -> None:
        self.format_version = format_version
        self.audio_filename = audio_filename
        self.audio_lead_in = audio_lead_in
        self.preview_time = preview_time
        self.countdown = countdown
        self.sample_set = sample_set
        self.stack_leniency = stack_leniency
        self.mode = mode
        self.letterbox_in_breaks = letterbox_in_breaks
        self.widescreen_storyboard = widescreen_storyboard
        self.bookmarks = list(bookmarks)
        self.distance_spacing = distance_spacing
        self.beat_divisor = beat_divisor
        self.grid_size = grid_size
        self.timeline_zoom = timeline_zoom
        self.title = title
        self.title_unicode = title_unicode
        self.artist = artist
        self.artist_unicode = artist_unicode
        self.creator = creator
        self.version = version
        self.source = source
        self.tags = list(tags)
        self.beatmap_id = beatmap_id
        self.beatmap_set_id = beatmap_set_id
        self.hp_drain_rate = hp_drain_rate
        self.circle_size = circle_size
        self.overall_difficulty = overall_difficulty
        self.approach_rate = approach_rate
        self.slider_multiplier = slider_multiplier
        self.slider_tick_rate = slider_tick_rate
        self.combo_colors = list(combo_colors)
        self.timing_points = list(timing_points)
        self.events = events
        
        self._hit_objects: List[HitObject] = list(hit_objects)
        # cache hit object stacking at different ar and cs values
        self._hit_objects_with_stacking: Dict[
            Tuple[float, float], Tuple[HitObject, ...]
        ] = {}

        # cache the stars with different mod combinations
        self._stars_cache: Dict[Tuple[bool, bool, bool, bool], float] = {}
        self._aim_stars_cache: Dict[Tuple[bool, bool, bool, bool], float] = {}
        self._speed_stars_cache: Dict[Tuple[bool, bool, bool, bool], float] = {}
        self._rhythm_awkwardness_cache: Dict[Tuple[bool, bool, bool, bool], float] = {}

    @property
    def display_name(self) -> str:
        """The name of the map as it appears in game."""
        return f"{self.artist} - {self.title} [{self.version}]"

    @memoize
    def bpm_min(
        self,
        *,
        half_time: bool = False,
        double_time: bool = False,
    ) -> float:
        """The minimum BPM in this beatmap.

        Parameters
        ----------
        half_time : bool
            The BPM with half time enabled.
        double_time : bool
            The BPM with double time enabled.

        Returns
        -------
        bpm : float
            The minimum BPM in this beatmap.
        """
        bpm = float(min(p.bpm for p in self.timing_points if p.bpm))
        if double_time:
            bpm *= 1.5
        elif half_time:
            bpm *= 0.75
        return bpm

    @memoize
    def bpm_max(
        self,
        *,
        half_time: bool = False,
        double_time: bool = False,
    ) -> float:
        """The maximum BPM in this beatmap.

        Parameters
        ----------
        half_time : bool
            The BPM with half time enabled.
        double_time : bool
            The BPM with double time enabled.

        Returns
        -------
        bpm : float
            The maximum BPM in this beatmap.
        """
        bpm = float(max(p.bpm for p in self.timing_points if p.bpm))
        if double_time:
            bpm *= 1.5
        elif half_time:
            bpm *= 0.75
        return bpm

    def hp(self, *, easy: bool = False, hard_rock: bool = False) -> float:
        """Compute the Health Drain (HP) value for different mods.

        Parameters
        ----------
        easy : bool, optional
            HP with the easy mod enabled.
        hard_rock : bool, optional
            HP with the hard rock mod enabled.

        Returns
        -------
        hp : float
            The HP value.
        """
        hp = self.hp_drain_rate
        if hard_rock:
            hp = min(1.4 * hp, 10)
        elif easy:
            hp /= 2
        return hp

    def cs(self, *, easy: bool = False, hard_rock: bool = False) -> float:
        """Compute the Circle Size (CS) value for different mods.

        Parameters
        ----------
        easy : bool, optional
            CS with the easy mod enabled.
        hard_rock : bool, optional
            CS with the hard rock mod enabled.

        Returns
        -------
        cs : float
            The CS value.
        """
        cs = self.circle_size
        if hard_rock:
            cs = min(1.3 * cs, 10)
        elif easy:
            cs /= 2
        return cs

    def od(
        self,
        *,
        easy: bool = False,
        hard_rock: bool = False,
        half_time: bool = False,
        double_time: bool = False,
    ) -> float:
        """Compute the Overall Difficulty (OD) value for different mods.

        Parameters
        ----------
        easy : bool, optional
            OD with the easy mod enabled.
        hard_rock : bool, optional
            OD with the hard rock mod enabled.
        half_time : bool, optional
            Effective OD with the half time mod enabled.
        double_time : bool, optional
            Effective OD with the double time mod enabled.

        Returns
        -------
        od : float
            The OD value.
        """
        od = self.overall_difficulty
        if hard_rock:
            od = min(1.4 * od, 10)
        elif easy:
            od /= 2

        if double_time:
            od = ms_300_to_od(2 * od_to_ms_300(od) / 3)
        elif half_time:
            od = ms_300_to_od(4 * od_to_ms_300(od) / 3)

        return od

    def ar(
        self,
        *,
        easy: bool = False,
        hard_rock: bool = False,
        half_time: bool = False,
        double_time: bool = False,
    ) -> float:
        """Compute the Approach Rate (AR) value for different mods.

        Parameters
        ----------
        easy : bool, optional
            AR with the easy mod enabled.
        hard_rock : bool, optional
            AR with the hard rock mod enabled.
        half_time : bool, optional
            Effective AR with the half time mod enabled.
        double_time : bool, optional
            Effective AR with the double time mod enabled.

        Returns
        -------
        ar : float
            The effective AR value.

        Notes
        -----
        ``double_time`` and ``half_time`` do not actually affect the in game
        AR; however, because the map is sped up or slowed down, the effective
        approach rate is changed.
        """
        ar = self.approach_rate
        if easy:
            ar /= 2
        elif hard_rock:
            ar = min(1.4 * ar, 10)

        if double_time:
            ar = ms_to_ar(2 * ar_to_ms(ar) / 3)
        elif half_time:
            ar = ms_to_ar(4 * ar_to_ms(ar) / 3)

        return ar

    @lazyval
    def breaks(self) -> list[Break]:
        """The breaks of this beatmap."""
        return [e for e in self.events if e.event_type is EventType.Break]

    @lazyval
    def backgrounds(self) -> list[Background]:
        """The backgrounds of this beatmap."""
        return [e for e in self.events if e.event_type is EventType.Background]

    @lazyval
    def videos(self) -> list[Video]:
        """The videos of this beatmap."""
        return [e for e in self.events if e.event_type is EventType.Video]

    def hit_objects(
        self,
        *,
        circles: bool = True,
        sliders: bool = True,
        spinners: bool = True,
        stacking: bool = True,
        easy: bool = False,
        hard_rock: bool = False,
        double_time: bool = False,
        half_time: bool = False,
    ) -> Tuple[HitObject, ...]:
        """Retrieve hit_objects.

        Parameters
        ----------
        circles : bool, optional
            If circles should be included.
        sliders : bool, optional
            If sliders should be included.
        spinners : bool, optional
            If spinners should be included.
        stacking : bool, optional
            If stacking should be calculated.
        easy : bool, optional
            Get the effective position of the hit objects with easy enabled.
        hard_rock : bool, optional
            Get the effective position of the hit objects with hard rock
            enabled.
        half_time : bool, optional
            Get the effective position of the hit objects with half time
            enabled.
        double_time : bool, optional
            Get the effective position of the hit objects with double time
            enabled.

        Returns
        -------
        hit_objects : list[HitObject]
            The objects with their effective positions and timings given the
            parameter set.
        """

        hit_objects: Sequence[HitObject] = self._hit_objects

        if hard_rock:
            hit_objects = [ob.hard_rock for ob in hit_objects]

        if stacking:
            ar = self.ar(easy=easy, hard_rock=hard_rock)
            cs = self.cs(easy=easy, hard_rock=hard_rock)
            # stacking changes with ar and cs (or equivalently EZ/HR), so only
            # cache up to ar and cs
            stacking_key = (ar, cs)

            if self.format_version >= 6:
                resolve_stacking_method = self._resolve_stacking
            else:
                resolve_stacking_method = self._resolve_stacking_old

            # use cache if available
            if stacking_key in self._hit_objects_with_stacking:
                hit_objects = self._hit_objects_with_stacking[stacking_key]
            else:
                hit_objects = resolve_stacking_method(hit_objects, ar, cs)
                # cache stacking calculation
                self._hit_objects_with_stacking[stacking_key] = hit_objects

        if double_time:
            hit_objects = [ob.double_time for ob in hit_objects]
        elif half_time:
            hit_objects = [ob.half_time for ob in hit_objects]

        keep_classes: List[type[HitObject]] = []
        if spinners:
            keep_classes.append(Spinner)
        if circles:
            keep_classes.append(Circle)
        if sliders:
            keep_classes.append(Slider)

        keep_classes_tuple = tuple(keep_classes)
        return tuple(ob for ob in hit_objects if isinstance(ob, keep_classes_tuple))

    def _resolve_stacking(
        self,
        hit_objects: Sequence[HitObject],
        ar: float,
        cs: float,
    ) -> Tuple[HitObject, ...]:
        """
        Adjusts the hit objects to account for stacking in beatmap versions 6
        and up.

        Parameters
        ----------
        hit_objects : list[HitObject]
            The objects to resolve stacking for.
        ar : float
            The approach rate to resolve stacking for.
        cs : float
            The circle size to resolve stacking for.

        Returns
        -------
        hit_objects : list[HitObject]
            The objects with their new positions, as adjusted by account for
            stacking.
        """
        stack_threshold = ar_to_ms(ar) * self.stack_leniency
        stack_threshold_td = timedelta(milliseconds=stack_threshold)
        stack_dist = 3
        hit_object_list = list(hit_objects)
        stack_height: Dict[HitObject, int] = {ob: 0 for ob in hit_object_list}
        hit_objects_reversed = list(reversed(hit_object_list))

        for i, ob_i in enumerate(hit_objects_reversed):

            if stack_height[ob_i] != 0 or isinstance(ob_i, Spinner):
                continue

            if isinstance(ob_i, Circle):
                for n, ob_n in enumerate(hit_objects_reversed[i + 1 :], start=i + 1):

                    if isinstance(ob_n, Spinner):
                        continue

                    if hasattr(ob_n, "end_time"):
                        end_time = ob_n.end_time
                    else:
                        end_time = ob_n.time

                    if (ob_i.time - end_time) > stack_threshold_td:
                        break

                    if (
                        isinstance(ob_n, Slider)
                        and distance(ob_n.curve(1), ob_i.position) < stack_dist
                    ):
                        stack_delta = stack_height[ob_i] - stack_height[ob_n] + 1

                        for hj in hit_objects_reversed[i:n]:
                            # For each object which was declared under this
                            # slider, we will offset it to appear *below*
                            # the slider end (rather than above).
                            dist = distance(ob_n.curve(1), hj.position)
                            if dist < stack_dist:
                                stack_height[hj] -= stack_delta

                        # We have hit a slider.  We should restart calculation
                        # using this as the new base.
                        # Breaking here will mean that the slider still has
                        # stack_height of 0, so it will be handled
                        # in the i-outer-loop.
                        break

                    if distance(ob_n.position, ob_i.position) < stack_dist:
                        # Keep processing as if there are no sliders.
                        # If we come across a slider, this gets cancelled out.
                        # NOTE: Sliders with start positions stacking
                        # are a special case that is also handled here.
                        stack_height[ob_n] = stack_height[ob_i] + 1
                        ob_i = ob_n

            elif isinstance(ob_i, Slider):
                # We have hit the first slider in a possible stack.
                # From this point on, we ALWAYS stack positive regardless.
                for n, ob_n in enumerate(hit_objects_reversed[i + 1 :], start=i + 1):

                    if isinstance(ob_n, Spinner):
                        continue

                    if ob_i.time - ob_n.time > stack_threshold_td:
                        break

                    if isinstance(ob_n, Slider):
                        ob_n_end_position = ob_n.curve(1)
                    else:
                        ob_n_end_position = ob_n.position

                    if distance(ob_n_end_position, ob_i.position) < stack_dist:
                        stack_height[ob_n] = stack_height[ob_i] + 1
                        ob_i = ob_n

        # apply stacking to original ordering
        radius = circle_radius(cs)
        stack_offset = radius / 10.0

        for hit_object in hit_object_list:
            offset = stack_offset * stack_height[hit_object]
            p = hit_object.position
            p_new = Position(p.x - offset, p.y - offset)
            hit_object.position = p_new

        return tuple(hit_object_list)

    def _resolve_stacking_old(
        self,
        hit_objects: Sequence[HitObject],
        ar: float,
        cs: float,
    ) -> Tuple[HitObject, ...]:
        """
        Adjusts the hit objects to account for stacking in beatmap versions 5
        and below.

        Parameters
        ----------
        hit_objects : list[HitObject]
            The objects to resolve stacking for.
        ar : float
            The approach rate to resolve stacking for.
        cs : float
            The circle size to resolve stacking for.

        Returns
        -------
        hit_objects : list[HitObject]
            The objects with their new positions, as adjusted by account for
            stacking.
        """
        stack_threshold = ar_to_ms(ar) * self.stack_leniency
        stack_threshold_td = timedelta(milliseconds=stack_threshold)
        stack_dist = 3
        hit_object_list = list(hit_objects)
        stack_height: Dict[HitObject, int] = {ob: 0 for ob in hit_object_list}
        for i, ob_i in enumerate(hit_object_list):

            if stack_height[ob_i] != 0 and not isinstance(ob_i, Slider):
                continue

            if hasattr(ob_i, "end_time"):
                start_time = ob_i.end_time
            else:
                start_time = ob_i.time
            slider_stack = 0

            for j, ob_j in enumerate(hit_object_list[i + 1 :], start=i + 1):

                if ob_j.time - stack_threshold_td > start_time:
                    break

                if distance(ob_j.position, ob_i.position) < stack_dist:
                    stack_height[ob_i] += 1

                    if hasattr(ob_j, "end_time"):
                        start_time = ob_j.end_time
                    else:
                        start_time = ob_j.time

                elif (
                    isinstance(ob_i, Slider)
                    and distance(ob_j.position, ob_i.curve(1)) < stack_dist
                ):
                    # Case for sliders - bump notes down and right,
                    # rather than up and left.
                    slider_stack += 1
                    stack_height[ob_j] -= slider_stack

                    if hasattr(ob_j, "end_time"):
                        start_time = ob_j.end_time
                    else:
                        start_time = ob_j.time

        # apply stacking
        radius = circle_radius(cs)
        stack_offset = radius / 10.0

        for hit_object in hit_object_list:
            offset = stack_offset * stack_height[hit_object]
            p = hit_object.position
            p_new = Position(p.x - offset, p.y - offset)
            hit_object.position = p_new

        return tuple(hit_object_list)

    @lazyval
    def _hit_object_times(self) -> List[timedelta]:
        """a (sorted) list of hitobject time's, so they can be searched with
        ``np.searchsorted``
        """
        return [hitobj.time for hitobj in self._hit_objects]

    def closest_hitobject(
        self,
        t: timedelta,
        side: Literal["left", "right"] = "left",
    ) -> HitObject:
        """The hitobject closest in time to ``t``.

        Parameters
        ----------
        t : datetime.timedelta
            The time to find the hitobject closest to.
        side : {"left", "right"}
            Whether to prefer the earlier (left) or later (right) hitobject
            when breaking ties.

        Returns
        -------
        hit_object : HitObject
            The closest hitobject in time to ``t``.
        None
            If the beatmap has no hitobjects.
        """
        if len(self._hit_objects) == 0:
            raise ValueError(
                f"The beatmap {self!r} must have at least one "
                "hit object to determine the closest hitobject."
            )
        if len(self._hit_objects) == 1:
            return self._hit_objects[0]

        i = bisect_left(self._hit_object_times, t)
        # if ``t`` is after the last hitobject, an index of
        # len(self._hit_objects) will be returned. The last hitobject will
        # always be the closest hitobject in this case.
        if i == len(self._hit_objects):
            return self._hit_objects[-1]
        # similar logic follows for the first hitobject.
        if i == 0:
            return self._hit_objects[0]

        # searchsorted tells us the two closest hitobjects, but not which is
        # closer. Check both candidates.
        hitobj1 = self._hit_objects[i - 1]
        hitobj2 = self._hit_objects[i]
        dist1 = abs(hitobj1.time - t)
        dist2 = abs(hitobj2.time - t)

        hitobj1_closer = dist1 <= dist2 if side == "left" else dist1 < dist2

        if hitobj1_closer:
            return hitobj1
        return hitobj2

    @lazyval
    def max_combo(self) -> int:
        """The highest combo that can be achieved on this beatmap."""
        max_combo = 0

        for hit_object in self._hit_objects:
            if isinstance(hit_object, Slider):
                max_combo += hit_object.ticks
            else:
                max_combo += 1

        return max_combo

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__}: {self.display_name}>"

    @classmethod
    def from_osz_path(cls, path: str) -> Dict[str, "Beatmap"]:
        """Read a beatmap collection from an ``.osz`` file on disk.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path to read from.

        Returns
        -------
        beatmaps : dict[str, Beatmap]
            A mapping from difficulty name to the parsed Beatmap.

        Raises
        ------
        ValueError
            Raised when the file cannot be parsed as a ``.osz`` file.
        """
        with ZipFile(path) as zf:
            return cls.from_osz_file(zf)

    @classmethod
    def from_path(cls, path: str) -> "Beatmap":
        """Read in a ``Beatmap`` object from a file on disk.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the file to read from.

        Returns
        -------
        beatmap : Beatmap
            The parsed beatmap object.

        Raises
        ------
        ValueError
            Raised when the file cannot be parsed as a ``.osu`` file.
        """
        with open(path, encoding="utf-8-sig") as file:
            return cls.from_file(file)

    @classmethod
    def from_osz_file(cls, file: ZipFile) -> Dict[str, "Beatmap"]:
        """Read a beatmap collection from a ``.osz`` file on disk.

        Parameters
        ----------
        file : zipfile.ZipFile
            The zipfile to read from.

        Returns
        -------
        beatmaps : dict[str, Beatmap]
            A mapping from difficulty name to the parsed Beatmap.

        Raises
        ------
        ValueError
            Raised when the file cannot be parsed as a ``.osz`` file.
        """
        return {
            beatmap.version: beatmap
            for beatmap in (
                Beatmap.parse(file.read(name).decode("utf-8-sig"))
                for name in file.namelist()
                if name.endswith(".osu")
            )
        }

    @classmethod
    def from_file(cls, file: IO[str]) -> "Beatmap":
        """Read in a ``Beatmap`` object from an open file object.

        Parameters
        ----------
        file : file-like
            The file object to read from.

        Returns
        -------
        beatmap : Beatmap
            The parsed beatmap object.

        Raises
        ------
        ValueError
            Raised when the file cannot be parsed as a ``.osu`` file.
        """
        return cls.parse(file.read())

    _mapping_groups = frozenset(
        {
            "General",
            "Editor",
            "Metadata",
            "Difficulty",
        }
    )

    def write_path(self, path: str) -> None:
        """Write a ``Beatmap`` object to a file on disk.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the file to write to.

        Raises
        ------
        ValueError
            Raised when the ``Beatmap`` object is invalid to be
            written to a ``.osu`` file.
        """
        with open(path, mode="w", encoding="utf-8-sig") as file:
            self.write_file(file)

    def write_file(self, file: IO[str]) -> None:
        """Write a ``Beatmap`` object to an open file object.

        Parameters
        ----------
        file : file-like
            The file object to write to.

        Raises
        ------
        ValueError
            Raised when the ``Beatmap`` object is invalid to be
            written to a ``.osu`` file.
        """
        file.write(self.pack())

    @classmethod
    def _find_groups(cls, lines: Iterable[str]) -> GroupsMapping:
        """Split the input data into the named groups.

        Parameters
        ----------
        lines : iterator[str]
            The raw lines from the file.

        Returns
        -------
        groups : dict[str, list[str] or dict[str, str]]
            The lines in the section. If the section is a mapping section
            the the value will be a dict from key to value.
        """
        groups: GroupsMapping = {}

        current_group: str | None = None
        group_buffer: List[str] = []

        def commit_group():
            nonlocal group_buffer

            if current_group is None:
                # we are not building a group, just return
                return

            # we are currently building a group
            if current_group in cls._mapping_groups:
                # build a dict from the ``Key: Value`` line format.
                mapping: Dict[str, str] = {}
                for line in group_buffer:
                    split = line.split(":", 1)
                    try:
                        key, value = split
                    except ValueError:
                        key = split[0]
                        value = ""

                    # throw away whitespace
                    mapping[key.strip()] = value.strip()
                groups[current_group] = mapping
            else:
                groups[current_group] = list(group_buffer)
            group_buffer = []

        for line in lines:
            # some (presmuably manually edited) beatmaps have whitespace at the
            # beginning or end of lines. This can cause logic relying on tokens
            # occurring at specific indices to fail, so we get rid of it.
            line = line.strip()
            if not line or line.startswith("//"):
                # filter out empty lines and comments
                continue

            if line[0] == "[" and line[-1] == "]":
                # we found a section header, commit the current buffered group
                # and start the new group
                commit_group()
                current_group = line[1:-1]
            else:
                group_buffer.append(line)

        # commit the final group
        commit_group()
        return groups

    @staticmethod
    def _parse_combo_colors(lines: Sequence[str]) -> List[ColorTuple]:
        """Parse the combo colors from the ``[Colours]`` section.
        
        Parameters
        ----------
        lines : list[str]
            The lines in the ``[Colours]`` section.
            
        Returns
        -------
        combo_colors : list[tuple[int, int, int]]
            The combo colors from the ``[Colours]`` section.
            
        Raises
        ------
        ValueError
            Raised when the combo colors cannot be parsed.
        """
        combo_color_map: Dict[int, ColorTuple] = {}
        for line in lines:
            key, value = line.split(":", 1)
            key = key.strip()

            if not key.startswith("Combo"):
                continue

            try:
                combo_index = int(key[5:])
            except ValueError:
                continue

            rgb = [part.strip() for part in value.split(",")]
            if len(rgb) != 3:
                raise ValueError(
                    f"Invalid color value for {key!r}: expected 3 channels, got {value!r}",
                )

            try:
                color = cast(ColorTuple, tuple(map(int, rgb)))
            except ValueError as exc:
                raise ValueError(f"Invalid color value for {key!r}: {value!r}") from exc

            combo_color_map[combo_index] = color

        return [combo_color_map[index] for index in sorted(combo_color_map)]

    @classmethod
    def parse(cls, data: str) -> "Beatmap":
        """Parse a ``Beatmap`` from text in the ``.osu`` format.

        Parameters
        ----------
        data : str
            The data to parse.

        Returns
        -------
        beatmap : Beatmap
            The parsed beatmap object.

        Raises
        ------
        ValueError
            Raised when the data cannot be parsed in the ``.osu`` format.
        """
        data = data.lstrip()
        lines = iter(data.splitlines())
        line = next(lines)

        # Remove BOM if present
        line = line.removeprefix("\ufeff")

        match = cls._version_regex.match(line)
        if match is None:
            raise ValueError(f"missing osu file format specifier in: {line!r}")

        format_version = int(match.group(1))
        groups = cls._find_groups(lines)
        groups_mapping = cast(Mapping[str, Mapping[str, str]], groups)

        artist = _get_as_str(groups_mapping, "Metadata", "Artist")
        title = _get_as_str(groups_mapping, "Metadata", "Title")
        od = _get_as_float(
            groups_mapping,
            "Difficulty",
            "OverallDifficulty",
        )

        if "TimingPoints" not in groups:
            groups["TimingPoints"] = {}

        if "HitObjects" not in groups:
            groups["HitObjects"] = {}

        timing_points: List[TimingPoint] = []
        # the parent starts as None because the first timing point should
        # not be inherited
        parent: TimingPoint | None = None
        for raw_timing_point in cast(List[str], groups["TimingPoints"]):
            timing_point = TimingPoint.parse(raw_timing_point, parent)
            if timing_point.parent is None:
                # we have a new parent node, pass that along to the new
                # timing points
                parent = timing_point
            timing_points.append(timing_point)

        event_data = cast(List[str], groups.get("Events", []))
        events = EventCollection.parse(event_data)

        slider_multiplier = _get_as_float(
            groups_mapping,
            "Difficulty",
            "SliderMultiplier",
            default=1.4,  # taken from wiki
        )
        slider_tick_rate = _get_as_float(
            groups_mapping,
            "Difficulty",
            "SliderTickRate",
            default=1.0,  # taken from wiki
        )
        combo_colors = cls._parse_combo_colors(
            cast(List[str], groups.get("Colours", []))
        )
        return cls(
            format_version=format_version,
            audio_filename=_get_as_str(groups_mapping, "General", "AudioFilename"),
            audio_lead_in=timedelta(
                milliseconds=_get_as_int(groups_mapping, "General", "AudioLeadIn", 0),
            ),
            preview_time=timedelta(
                milliseconds=_get_as_int(groups_mapping, "General", "PreviewTime", -1),
            ),
            countdown=_get_as_bool(groups_mapping, "General", "Countdown", False),
            sample_set=_get_as_str(
                groups_mapping,
                "General",
                "SampleSet",
                "Normal",
            ),
            stack_leniency=_get_as_float(
                groups_mapping,
                "General",
                "StackLeniency",
                0,
            ),
            mode=GameMode(_get_as_int(groups_mapping, "General", "Mode", 0)),
            letterbox_in_breaks=_get_as_bool(
                groups_mapping,
                "General",
                "LetterboxInBreaks",
                False,
            ),
            widescreen_storyboard=_get_as_bool(
                groups_mapping,
                "General",
                "WidescreenStoryboard",
                False,
            ),
            bookmarks=[
                timedelta(milliseconds=ms)
                for ms in _get_as_int_list(
                    groups_mapping,
                    "Editor",
                    "Bookmarks",
                    [],
                )
            ],
            distance_spacing=_get_as_float(
                groups_mapping,
                "Editor",
                "DistanceSpacing",
                1,
            ),
            beat_divisor=_get_as_int(groups_mapping, "Editor", "BeatDivisor", 4),
            grid_size=_get_as_int(groups_mapping, "Editor", "GridSize", 4),
            timeline_zoom=_get_as_float(
                groups_mapping,
                "Editor",
                "TimelineZoom",
                1.0,
            ),
            title=title,
            title_unicode=_get_as_str(
                groups_mapping,
                "Metadata",
                "TitleUnicode",
                title,
            ),
            artist=artist,
            artist_unicode=_get_as_str(
                groups_mapping,
                "Metadata",
                "ArtistUnicode",
                artist,
            ),
            creator=_get_as_str(groups_mapping, "Metadata", "Creator"),
            version=_get_as_str(groups_mapping, "Metadata", "Version"),
            source=_get_as_str(groups_mapping, "Metadata", "Source", None),
            # space delimited list
            tags=_get_as_str(groups_mapping, "Metadata", "Tags", "").split(),
            beatmap_id=_get_as_int(groups_mapping, "Metadata", "BeatmapID", None),
            beatmap_set_id=_get_as_int(
                groups_mapping,
                "Metadata",
                "BeatmapSetID",
                None,
            ),
            hp_drain_rate=_get_as_float(groups_mapping, "Difficulty", "HPDrainRate"),
            circle_size=_get_as_float(groups_mapping, "Difficulty", "CircleSize"),
            overall_difficulty=_get_as_float(
                groups_mapping,
                "Difficulty",
                "OverallDifficulty",
            ),
            approach_rate=_get_as_float(
                groups_mapping,
                "Difficulty",
                "ApproachRate",
                # old maps didn't have an AR so the OD is used as a default
                default=od,
            ),
            slider_multiplier=slider_multiplier,
            slider_tick_rate=slider_tick_rate,
            combo_colors=combo_colors,
            timing_points=timing_points,
            hit_objects=list(
                map(
                    partial(
                        HitObject.parse,
                        timing_points=timing_points,
                        slider_multiplier=slider_multiplier,
                        slider_tick_rate=slider_tick_rate,
                    ),
                    cast(List[str], groups["HitObjects"]),
                )
            ),
            events=events,
        )

    def pack(self):
        """The string content in ``.osu`` file of this beatmap.
        Default values assumed by osu! client Beatmap editor are used to
        replace member values which are missing or are of incorrect type.

        Returns
        -------
        packed_str : str
            The packed str of this beatmap

        Raises
        ------
        ValueError
            Raised when essential member values are missing
            or are of incorrect type.
        """

        def pack_field(
            field, field_value, pack_func, default=no_default, skip_empty=False
        ):
            packed_field_str = pack_func(field, field_value, default=default)
            
            # if ``skip_empty`` is True, empty string will be
            # returned for empty fields
            if skip_empty and packed_field_str == "":
                return ""
            
            fields_with_spaces = (
                "DistanceSpacing",
                "BeatDivisor",
                "GridSize",
                "TimelineZoom",
                "AudioFilename",
                "AudioLeadIn",
                "PreviewTime",
                "Countdown",
                "SampleSet",
                "StackLeniency",
                "Mode",
                "LetterboxInBreaks",
                "WidescreenStoryboard",
                "Bookmarks"
            )

            # Some fields have a space between the colon and the value
            if field in fields_with_spaces:
                packed_field_str = " " + packed_field_str

            return field + ":" + packed_field_str + "\n"

        # we'll pin ourselves to file format v14 for packing for now. We'll
        # need to update this if we ever update the format in which we output
        # packed beatmaps in.
        packed_str = "osu file format v14\n\n"

        # pack General section
        packed_str += "[General]\n"
        packed_str += pack_field(
            "AudioFilename", self.audio_filename, _pack_str, no_default
        )
        packed_str += pack_field(
            "AudioLeadIn",
            self.audio_lead_in,
            _pack_timedelta,
            timedelta(milliseconds=0),
        )
        packed_str += pack_field(
            "PreviewTime",
            self.preview_time,
            _pack_timedelta,
            timedelta(milliseconds=-1),
        )
        packed_str += pack_field("Countdown", self.countdown, _pack_bool, False)
        packed_str += pack_field("SampleSet", self.sample_set, _pack_str, "None")
        packed_str += pack_field("StackLeniency", self.stack_leniency, _pack_float, 0)
        packed_str += pack_field("Mode", self.mode, _pack_int_enum, GameMode.standard)
        packed_str += pack_field(
            "LetterboxInBreaks", self.letterbox_in_breaks, _pack_bool, False
        )
        packed_str += pack_field(
            "WidescreenStoryboard", self.widescreen_storyboard, _pack_bool, False
        )
        packed_str += "\n"

        # pack Editor section
        packed_str += "[Editor]\n"
        # Bookmarks field actually does not even exist in .osu file
        # if there's no bookmark at all.
        packed_str += pack_field(
            "Bookmarks",
            self.bookmarks,
            partial(_pack_timedelta_list, sep=","),
            [],
            skip_empty=True,
        )
        packed_str += pack_field(
            "DistanceSpacing", self.distance_spacing, _pack_float, 1.0
        )
        packed_str += pack_field("BeatDivisor", self.beat_divisor, _pack_int, 4)
        packed_str += pack_field("GridSize", self.grid_size, _pack_int, 4)
        packed_str += pack_field("TimelineZoom", self.timeline_zoom, _pack_float, 1.0)
        packed_str += "\n"

        # pack Metadata section
        packed_str += "[Metadata]\n"
        # osu! beatmap editor forces mappers to enter Title, Artist,
        # Creator, Version fields when creating a new beatmap from
        # an audio file, so these fields are considered indispensable
        # for a valid Beatmap. When packing a Beatmap, ValueError will
        # be raised if these fields do not have sensible values.
        packed_str += pack_field("Title", self.title, _pack_str, no_default)
        packed_str += pack_field(
            "TitleUnicode", self.title_unicode, _pack_str, self.title
        )
        packed_str += pack_field("Artist", self.artist, _pack_str, no_default)
        packed_str += pack_field(
            "ArtistUnicode", self.artist_unicode, _pack_str, self.artist
        )
        packed_str += pack_field("Creator", self.creator, _pack_str, no_default)
        packed_str += pack_field("Version", self.version, _pack_str, no_default)
        packed_str += pack_field("Source", self.source, _pack_str, "")
        packed_str += pack_field(
            "Tags", self.tags, partial(_pack_str_list, sep=" "), ""
        )
        # old beatmaps did not store beatmap_id or beatmap_set_id.
        if self.beatmap_id is not None:
            packed_str += pack_field(
                "BeatmapID",
                self.beatmap_id,
                _pack_int,
            )

        if self.beatmap_set_id is not None:
            packed_str += pack_field("BeatmapSetID", self.beatmap_set_id, _pack_int)
        packed_str += "\n"

        # pack Difficulty section
        packed_str += "[Difficulty]\n"
        packed_str += pack_field("HPDrainRate", self.hp_drain_rate, _pack_float, 5.0)
        packed_str += pack_field("CircleSize", self.circle_size, _pack_float, 5.0)
        packed_str += pack_field(
            "OverallDifficulty", self.overall_difficulty, _pack_float, 5.0
        )
        packed_str += pack_field("ApproachRate", self.approach_rate, _pack_float, 5.0)
        packed_str += pack_field(
            "SliderMultiplier", self.slider_multiplier, _pack_float, 1.4
        )
        packed_str += pack_field(
            "SliderTickRate", self.slider_tick_rate, _pack_float, 1.0
        )
        packed_str += "\n"

        # pack Events section
        packed_str += "[Events]\n"
        packed_str += self.events.pack() + "\n"
        packed_str += "\n"

        # pack TimingPoints section
        packed_str += "[TimingPoints]\n"
        for timing_point in self.timing_points:
            # each timing point occupies a line
            packed_str += timing_point.pack() + "\n"
        packed_str += "\n"

        # pack Colours section
        if self.combo_colors:
            packed_str += "[Colours]\n"
            for index, (r, g, b) in enumerate(self.combo_colors, start=1):
                packed_str += f"Combo{index} : {r},{g},{b}\n"
            packed_str += "\n"

        # pack HitObjects section
        packed_str += "[HitObjects]\n"
        for hit_object in self._hit_objects:
            # each hit object occupies a line
            packed_str += hit_object.pack() + "\n"

        return packed_str

    def timing_point_at(self, time: timedelta) -> TimingPoint:
        """Get the :class:`slider.beatmap.TimingPoint` at the given time.

        Parameters
        ----------
        time : datetime.timedelta
            The time to lookup the :class:`slider.beatmap.TimingPoint` for.

        Returns
        -------
        timing_point : TimingPoint
            The :class:`slider.beatmap.TimingPoint` at the given time.
        """
        for tp in reversed(self.timing_points):
            if tp.offset <= time:
                return tp

        return self.timing_points[0]

    @staticmethod
    def _base_strain(strain: float | np.ndarray) -> float | np.ndarray:
        """Scale up the base attribute"""
        return ((5 * np.maximum(1, strain / 0.0675) - 4) ** 3) / 100000

    @staticmethod
    def _handle_group(group: Sequence[timedelta]) -> Iterator[float]:
        inner = range(1, len(group))
        for n in range(len(group)):
            for m in inner:
                if n == m:
                    continue

                a = group[n]
                b = group[m]

                ratio = a / b if a > b else b / a

                closest_power_of_two = 2 ** round(np.log2(ratio))
                offset = abs(closest_power_of_two - ratio) / closest_power_of_two
                yield offset**2

    _strain_step = timedelta(milliseconds=400)
    _decay_weight = 0.9

    def _calculate_difficulty(
        self,
        strain: _DifficultyHitObject.Strain,
        difficulty_hit_objects: Sequence[_DifficultyHitObject],
    ) -> float:
        highest_strains: List[float] = []
        append_highest_strain = highest_strains.append

        strain_step = self._strain_step
        interval_end = strain_step
        max_strain = 0.0

        previous: _DifficultyHitObject | None = None
        for difficulty_hit_object in difficulty_hit_objects:
            while difficulty_hit_object.hit_object.time > interval_end:
                append_highest_strain(max_strain)

                if previous is None:
                    max_strain = 0.0
                else:
                    decay = (
                        _DifficultyHitObject.decay_base[strain]
                        ** (interval_end - previous.hit_object.time).total_seconds()
                    )
                    max_strain = previous.strains[strain] * decay

                interval_end += strain_step

            max_strain = max(max_strain, difficulty_hit_object.strains[strain])
            previous = difficulty_hit_object

        difficulty = 0.0
        weight = 1.0

        decay_weight = self._decay_weight
        for strain_value in sorted(highest_strains, reverse=True):
            difficulty += weight * strain_value
            weight *= decay_weight

        return difficulty

    _star_scaling_factor = 0.0675
    _extreme_scaling_factor = 0.5

    @staticmethod
    def _product_no_diagonal(sequence: Sequence[T]) -> Iterator[Tuple[T, T]]:
        """An iterator of the Cartesian product of ``sequence`` with itself
        with the diagonals removed.

        Parameters
        ----------
        sequence : sequence
            The sequence to take the product with.

        Yields
        ------
        pair : (any, any)
            The element of the product which is not on the diagonal.
        """
        inner = range(1, len(sequence))
        for n in range(len(sequence)):
            for m in inner:
                if n == m:
                    continue

                yield sequence[n], sequence[m]

    def hit_object_difficulty(
        self,
        *,
        easy: bool = False,
        hard_rock: bool = False,
        double_time: bool = False,
        half_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the difficulty of each hit object.

        Parameters
        ----------
        easy : bool
            Compute difficulty with easy.
        hard_rock : bool
            Compute difficulty with hard rock.
        double_time : bool
            Compute difficulty with double time.
        half_time : bool
            Compute difficulty with half time.

        Returns
        ----------
        times : np.ndarray
            Single column array of times as ``timedelta64[ns]``
        difficulties : np.ndarray
            Array of difficulties as ``float64``. Speed in the first column,
            aim in the second.
        """
        cs = self.cs(easy=easy, hard_rock=hard_rock)
        radius = circle_radius(cs)

        modify: Callable[[HitObject], HitObject]
        if double_time:
            modify = cast(
                Callable[[HitObject], HitObject], op.attrgetter("double_time")
            )
        elif half_time:
            modify = cast(Callable[[HitObject], HitObject], op.attrgetter("half_time"))
        else:

            def modify(e: HitObject) -> HitObject:
                return e

        times = np.empty(
            (len(self._hit_objects) - 1, 1),
            dtype="timedelta64[ns]",
        )
        strains = np.empty((len(self._hit_objects) - 1, 2), dtype=np.float64)

        hit_objects = map(modify, self._hit_objects)
        previous = _DifficultyHitObject(next(hit_objects), radius)
        for i, hit_object in enumerate(hit_objects):
            new = _DifficultyHitObject(
                hit_object,
                radius,
                previous,
            )
            times[i] = hit_object.time
            strains[i] = new.strains
            previous = new

        return times, strains

    def smoothed_difficulty(
        self,
        smoothing_window: float | int,
        num_points: int,
        *,
        easy: bool = False,
        hard_rock: bool = False,
        double_time: bool = False,
        half_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a smoothed difficulty at evenly spaced points in time
        between the beginning of the song and the last hit object of the map.

        Done by taking an average of difficulties of hit objects within a
        certain time window of each point

        Useful if you want to calculate a difficulty curve for the map
        since the unsmoothed values vary locally a lot.

        Parameters
        ----------
        smoothing_window : int or float
            Time window (in seconds) for the moving average.
            Bigger will make a smoother curve.
        num_points : int
            Number of points to calculate the average at.
        easy : bool
            Compute difficulty with easy.
        hard_rock : bool
            Compute difficulty with hard rock.
        double_time : bool
            Compute difficulty with double time.
        half_time : bool
            Compute difficulty with half time.

        Returns
        ----------
        difficulties : array
            2D array containing smoothed time, difficulty pairs.
        """
        times, values = self.hit_object_difficulty(
            easy=easy, hard_rock=hard_rock, double_time=double_time, half_time=half_time
        )

        return _moving_average_by_time(
            times,
            values,
            smoothing_window,
            num_points,
        )

    def _calculate_stars(
        self,
        easy: bool,
        hard_rock: bool,
        double_time: bool,
        half_time: bool,
    ) -> None:
        """Compute the stars and star components for this map.

        Parameters
        ----------
        easy : bool
            Compute stars with easy.
        hard_rock : bool
            Compute stars with hard rock.
        double_time : bool
            Compute stars with double time.
        half_time : bool
            Compute stars with half time.
        """
        cs = self.cs(easy=easy, hard_rock=hard_rock)
        radius = circle_radius(cs)

        difficulty_hit_objects: List[_DifficultyHitObject] = []
        append_difficulty_hit_object = difficulty_hit_objects.append

        intervals: List[timedelta] = []
        append_interval = intervals.append

        modify: Callable[[HitObject], HitObject]
        if double_time:
            modify = cast(
                Callable[[HitObject], HitObject], op.attrgetter("double_time")
            )
        elif half_time:
            modify = cast(Callable[[HitObject], HitObject], op.attrgetter("half_time"))
        else:

            def modify(e: HitObject) -> HitObject:
                return e

        hit_objects = map(modify, self._hit_objects)
        previous = _DifficultyHitObject(next(hit_objects), radius)
        append_difficulty_hit_object(previous)
        for hit_object in hit_objects:
            new = _DifficultyHitObject(
                hit_object,
                radius,
                previous,
            )
            append_interval(new.hit_object.time - previous.hit_object.time)
            append_difficulty_hit_object(new)
            previous = new

        group: List[timedelta] = []
        append_group_member = group.append
        clear_group = group.clear

        # todo: compute break time from ar
        break_threshhold = timedelta(milliseconds=1200)

        count_offsets = 0
        rhythm_awkwardness = 0.0

        for interval in intervals:
            is_break = interval >= break_threshhold

            if not is_break:
                append_group_member(interval)

            if is_break or len(group) == 5:
                for awk in self._handle_group(group):
                    count_offsets += 1
                    rhythm_awkwardness += awk

                clear_group()

        for awk in self._handle_group(group):
            count_offsets += 1
            rhythm_awkwardness += awk

        rhythm_awkwardness /= count_offsets or 1
        rhythm_awkwardness *= 82

        aim = self._calculate_difficulty(
            _DifficultyHitObject.Strain.aim,
            difficulty_hit_objects,
        )
        speed = self._calculate_difficulty(
            _DifficultyHitObject.Strain.speed,
            difficulty_hit_objects,
        )

        key = easy, hard_rock, double_time, half_time
        self._aim_stars_cache[key] = aim = np.sqrt(aim) * self._star_scaling_factor
        self._speed_stars_cache[key] = speed = (
            np.sqrt(speed) * self._star_scaling_factor
        )
        self._stars_cache[key] = (
            aim + speed + abs(speed - aim) * self._extreme_scaling_factor
        )
        self._rhythm_awkwardness_cache[key] = rhythm_awkwardness

    @staticmethod
    def _stars_cache_value(name: str, doc: str) -> Callable[..., float]:
        """Create a cached function from pulling from the values generated
        in ``_calculate_stars``.

        Parameters
        ----------
        name : str
            The name of the attribute.
        doc : str
            The docstring for the attribute.

        Returns
        -------
        getter : function
            The getter function.
        """
        cache_name = f"_{name}_cache"

        def get(
            self: "Beatmap",
            *,
            easy: bool = False,
            hard_rock: bool = False,
            double_time: bool = False,
            half_time: bool = False,
        ) -> float:
            key = (
                bool(easy),
                bool(hard_rock),
                bool(double_time),
                bool(half_time),
            )
            try:
                return getattr(self, cache_name)[key]
            except KeyError:
                self._calculate_stars(*key)

            return getattr(self, cache_name)[key]

        get.__name__ = name
        get.__doc__ = doc
        return get

    speed_stars = _stars_cache_value(
        "speed_stars",
        """The speed part of the stars.

        Parameters
        ----------
        easy : bool, optional
            Stars with the easy mod applied.
        hard_rock : bool, optional
            Stars with the hard rock mod applied.
        double_time : bool, optional
            Stars with the double time mod applied.
        half_time : bool, optional
            Stars with the half time mod applied.

        Returns
        -------
        speed_stars : float
            The aim component of the stars.
        """,
    )
    aim_stars = _stars_cache_value(
        "aim_stars",
        """The aim part of the stars.

        Parameters
        ----------
        easy : bool, optional
            Stars with the easy mod applied.
        hard_rock : bool, optional
            Stars with the hard rock mod applied.
        double_time : bool, optional
            Stars with the double time mod applied.
        half_time : bool, optional
            Stars with the half time mod applied.

        Returns
        -------
        aim_stars : float
            The aim component of the stars.
        """,
    )
    stars = _stars_cache_value(
        "stars",
        """The stars as seen in osu!.

        Parameters
        ----------
        easy : bool, optional
            Stars with the easy mod applied.
        hard_rock : bool, optional
            Stars with the hard rock mod applied.
        double_time : bool, optional
            Stars with the double time mod applied.
        half_time : bool, optional
            Stars with the half time mod applied.

        Returns
        -------
        stars : float
            The total stars for the map.
        """,
    )
    rhythm_awkwardness = _stars_cache_value(
        "rhythm_awkwardness",
        """The rhythm awkwardness component of the song.

        Parameters
        ----------
        easy : bool, optional
            Rhythm awkwardness with the easy mod applied.
        hard_rock : bool, optional
            Rhythm awkwardness with the hard rock mod applied.
        double_time : bool, optional
            Rhythm awkwardness with the double time mod applied.
        half_time : bool, optional
            Rhythm awkwardness with the half time mod applied.

        Returns
        -------
        rhythm_awkwardness : float
            The rhythm awkwardness score.
        """,
    )

    del _stars_cache_value

    def _round_hitcounts(
        self,
        accuracy: np.ndarray,
        count_miss: np.ndarray | int | Sequence[int] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Round the accuracy to the nearest hit counts.

        Parameters
        ----------
        accuracy : np.ndarray[float]
            The accuracy to round in the range [0, 1]
        count_miss : np.ndarray[int]int, optional
            The number of misses to fix.

        Returns
        -------
        count_300 : np.ndarray[int]
            The number of 300s.
        count_100 : np.ndarray[int]
            The number of 100s.
        count_50 : np.ndarray[int]
            The number of 50s.
        count_miss : np.ndarray[int]
            The number of misses.
        """
        accuracy_array = np.array(accuracy, ndmin=1, copy=False, dtype=np.float64)

        if count_miss is None:
            count_miss_array = np.zeros_like(accuracy_array, dtype=np.int64)
        else:
            count_miss_array = np.array(
                count_miss,
                ndmin=1,
                copy=False,
                dtype=np.int64,
            )

        max_300 = np.asarray(
            len(self._hit_objects) - count_miss_array,
            dtype=np.int64,
        )

        zero_hits = np.zeros_like(max_300, dtype=np.int64)

        clamped_accuracy = np.maximum(
            0.0,
            np.minimum(
                _calculate_accuracy_array(
                    max_300,
                    zero_hits,
                    zero_hits,
                    count_miss_array,
                )
                * 100.0,
                accuracy_array * 100.0,
            ),
        )

        count_50 = np.zeros_like(max_300, dtype=np.int64)
        count_100 = np.round(
            -3.0
            * (
                (clamped_accuracy * 0.01 - 1.0) * len(self._hit_objects)
                + count_miss_array
            )
            * 0.5,
        ).astype(np.int64)

        mask = count_100 > len(self._hit_objects) - count_miss_array
        count_100[mask] = 0
        count_50[mask] = np.round(
            -6.0
            * (
                (clamped_accuracy[mask] * 0.01 - 1.0) * len(self._hit_objects)
                + count_miss_array[mask]
            )
            * 0.2,
        ).astype(np.int64)
        count_50[mask] = np.minimum(max_300[mask], count_50[mask])

        count_100[~mask] = np.minimum(max_300[~mask], count_100[~mask])

        count_300 = len(self._hit_objects) - count_100 - count_50 - count_miss_array

        return count_300, count_100, count_50, count_miss_array

    def performance_points(
        self,
        *,
        combo: int | None = None,
        accuracy: float | Sequence[float] | np.ndarray | None = None,
        count_300: int | np.ndarray | None = None,
        count_100: int | np.ndarray | None = None,
        count_50: int | np.ndarray | None = None,
        count_miss: int | np.ndarray | None = None,
        no_fail: bool = False,
        easy: bool = False,
        hidden: bool = False,
        hard_rock: bool = False,
        double_time: bool = False,
        half_time: bool = False,
        flashlight: bool = False,
        spun_out: bool = False,
        version: int = 1,
    ) -> float | np.ndarray:
        """Compute the performance points for the given map.

        Parameters
        ----------
        combo : int, optional
            The combo achieved on the map. Defaults to max combo.
        accuracy : float, optional
            The accuracy achieved in the range [0, 1]. If not provided
            and none of ``count_300``, ``count_100``, or ``count_50``
            provided then the this defaults to 100%
        count_300 : int, optional
            The number of 300s hit.
        count_100 : int, optional
            The number of 100s hit.
        count_50 : int, optional
            The number of 50s hit.
        count_miss : int, optional
            The number of misses.
        no_fail : bool, optional
            Account for no fail mod.
        easy : bool, optional
            Account for the easy mod.
        hidden : bool, optional
            Account for the hidden mod.
        hard_rock : bool, optional
            Account for the hard rock mod.
        double_time : bool, optional
            Account for the double time mod.
        half_time : bool, optional
            Account for the half time mod.
        flashlight : bool, optional
            Account for the flashlight mod.
        spun_out : bool, optional
            Account for the spun out mod.
        version : int, optional
            The version of the performance points calculation to use.

        Returns
        -------
        pp : float
            The performance points awarded for the specified play.

        Notes
        -----
        ``accuracy`` or hit counts may be passed as array-likes in which case
        the resulting ``pp`` will be a sequence of the same length. This is
        more efficient for computing PP with different results on the same
        beatmap.

        Examples
        --------
        >>> from slider.example_data.beatmaps import sendan_life
        >>> beatmap = sendan_life("Crystal's Garakowa")
        >>> # compute for 100%
        >>> beatmap.performance_points(accuracy=1.0)
        274.487178791355
        >>> # compute for 95%  accuracy
        >>> beatmap.performance_points(accuracy=0.95)
        219.09554433691147
        >>> # compute with explicit hit counts
        >>> beatmap.performance_points(
        ...     count_300=330,
        ...     count_100=2,
        ...     count_50=0,
        ...     count_miss=0,
        ... )
        ... )
        array([ 219.09554434,  223.67413382,  230.20890527,  239.72525216, 253.74272587,  274.48717879])
        >>> # with mods
        >>> beatmap.performance_points(
        ...     accuracy=[0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
        ...     hidden=True,
        ... )
        array([ 245.0240618 ,  249.77318802,  256.50049755,  266.24423831, 280.54452189,  301.66016166])
        """  # noqa
        if version not in {1, 2}:
            raise ValueError(f"unknown PP version: {version}")

        if combo is None:
            combo = self.max_combo

        if accuracy is not None:
            if count_300 is not None or count_100 is not None or count_50 is not None:
                raise ValueError("cannot pass accuracy and hit counts")
            # compute the closest hit counts for the accuracy
            accuracy_array = np.array(accuracy, ndmin=1, copy=False, dtype=np.float64)
            count_miss_array = (
                np.zeros_like(accuracy_array, dtype=np.int64)
                if count_miss is None
                else np.array(count_miss, ndmin=1, copy=False, dtype=np.int64)
            )
            count_300, count_100, count_50, count_miss_array = self._round_hitcounts(
                accuracy_array,
                count_miss_array,
            )
            accuracy = accuracy_array
            count_miss = count_miss_array

        elif count_300 is None and count_100 is None and count_50 is None:
            accuracy_array = np.array(1.0, ndmin=1, copy=False, dtype=np.float64)
            count_miss_array = (
                np.zeros_like(accuracy_array, dtype=np.int64)
                if count_miss is None
                else np.array(count_miss, ndmin=1, copy=False, dtype=np.int64)
            )
            count_300, count_100, count_50, count_miss_array = self._round_hitcounts(
                accuracy_array,
                count_miss_array,
            )
            accuracy = accuracy_array
            count_miss = count_miss_array
        else:
            if count_300 is None or count_100 is None or count_50 is None:
                raise ValueError("must provide all hit counts")

            count_300 = np.array(count_300, ndmin=1, copy=False, dtype=np.int64)
            count_100 = np.array(count_100, ndmin=1, copy=False, dtype=np.int64)
            count_50 = np.array(count_50, ndmin=1, copy=False, dtype=np.int64)

            if count_miss is None:
                count_miss = np.zeros_like(count_300, dtype=np.int64)
            else:
                count_miss = np.array(count_miss, ndmin=1, copy=False, dtype=np.int64)

            if np.any(
                count_300 + count_100 + count_50 + count_miss != len(self._hit_objects)
            ):
                s = count_300 + count_100 + count_50 + count_miss
                os = len(self._hit_objects)
                raise ValueError(
                    f"hit counts don't sum to the total for the map, {s} != {os}"
                )

        od = self.od(
            easy=easy,
            hard_rock=hard_rock,
            half_time=half_time,
            double_time=double_time,
        )
        ar = self.ar(
            easy=easy,
            hard_rock=hard_rock,
            half_time=half_time,
            double_time=double_time,
        )

        accuracy_scaled = _calculate_accuracy_array(
            count_300,
            count_100,
            count_50,
            count_miss,
        )
        accuracy = accuracy_scaled * 100
        accuracy_bonus = 0.5 + accuracy_scaled / 2

        count_hit_objects = len(self._hit_objects)
        count_hit_objects_over_2000 = count_hit_objects / 2000
        length_bonus = (
            0.95
            + 0.4 * np.minimum(1.0, count_hit_objects_over_2000)
            + (
                np.log10(count_hit_objects_over_2000) * 0.5
                if count_hit_objects > 2000
                else 0
            )
        )

        miss_penalty = 0.97**count_miss

        combo_break_penalty = combo**0.8 / self.max_combo**0.8

        ar_bonus = 1.0
        if ar > 10.33:
            # high ar bonus
            ar_bonus += 0.45 * (ar - 10.33)
        elif ar < 8:
            # low ar bonus
            low_ar_bonus = 0.01 * (8.0 - ar)
            if hidden:
                low_ar_bonus *= 2
            ar_bonus += low_ar_bonus

        hidden_bonus = 1.18 if hidden else 1.0
        flashlight_bonus = (1.45 * length_bonus) if flashlight else 1.0
        od_bonus = 0.98 + od**2 / 2500

        mods: Dict[str, bool] = {
            "easy": easy,
            "hard_rock": hard_rock,
            "half_time": half_time,
            "double_time": double_time,
        }
        aim_score = (
            self._base_strain(self.aim_stars(**mods))
            * length_bonus
            * miss_penalty
            * combo_break_penalty
            * ar_bonus
            * accuracy_bonus
            * hidden_bonus
            * flashlight_bonus
            * od_bonus
        )

        speed_score = (
            self._base_strain(self.speed_stars(**mods))
            * length_bonus
            * miss_penalty
            * combo_break_penalty
            * accuracy_bonus
            * od_bonus
        )

        if version == 2:
            count_circles = count_hit_objects
            real_accuracy = accuracy
        else:
            count_circles = len(
                self.hit_objects(circles=True, sliders=False, spinners=False)
            )
            if count_circles:
                real_accuracy = (
                    (count_300 - (count_hit_objects - count_circles)) * 300.0
                    + count_100 * 100.0
                    + count_50 * 50.0
                ) / (count_circles * 300)
                real_accuracy = np.maximum(real_accuracy, 0)
            else:
                real_accuracy = np.array(0.0, dtype=np.float64)

        accuracy_length_bonus = min(1.5, (count_circles / 1000) ** 0.3)
        accuracy_hidden_bonus = 1.02 if hidden else 1.0
        accuracy_flashlight_bonus = 1.02 if flashlight else 1.0
        accuracy_score = (
            1.52163**od
            * real_accuracy**24.0
            * 2.83
            * accuracy_length_bonus
            * accuracy_hidden_bonus
            * accuracy_flashlight_bonus
        )

        final_multiplier = 1.12
        if no_fail:
            final_multiplier *= 0.9
        if spun_out:
            final_multiplier *= 0.95

        out = ((aim_score**1.1) + (speed_score**1.1) + (accuracy_score**1.1)) ** (
            1 / 1.1
        ) * final_multiplier

        if out.shape == (1,):
            return float(out.item())

        return out
