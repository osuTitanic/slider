from datetime import timedelta
from enum import IntEnum
from typing import Sequence
from collections import defaultdict

class EventType(IntEnum):
    Background = 0
    Video = 1
    Break = 2
    ColourTransformation = 3
    Sprite = 4
    Sample = 5
    Animation = 6

    @classmethod
    def _missing_(cls, value):
        return {
            "Background": EventType.Background,
            "Video": EventType.Video,
            "Break": EventType.Break,
            "Colour": EventType.ColourTransformation,
            "Color": EventType.ColourTransformation,
            "ColourTransformation": EventType.ColourTransformation,
            "ColorTransformation": EventType.ColourTransformation,
            "Sprite": EventType.Sprite,
            "Animation": EventType.Animation,
            "Sample": EventType.Sample,
        }[value]


class LayerType(IntEnum):
    Background = 0
    Fail = 1
    Pass = 2
    Foreground = 3
    Overlay = 4

    @classmethod
    def _missing_(cls, value):
        return {
            "Background": LayerType.Background,
            "Fail": LayerType.Fail,
            "Failing": LayerType.Fail,
            "Pass": LayerType.Pass,
            "Passing": LayerType.Pass,
            "Foreground": LayerType.Foreground,
            "Overlay": LayerType.Overlay,
        }[value]


class Event:
    """Base class for all storyboard events."""

    def __init__(
        self,
        event_type: EventType | None,
        start_time: int,
        raw_data: str | None = None,
    ):
        self.event_type = event_type
        self.start_time = timedelta(milliseconds=start_time)
        self.raw_data = raw_data

    @staticmethod
    def delta_to_ms(delta: timedelta) -> int:
        return int(delta.total_seconds() * 1000)

    def pack(self) -> str:
        if self.raw_data is not None:
            return self.raw_data

        raise NotImplementedError(f"pack is not implemented for {type(self).__name__}")

    @classmethod
    def parse(cls, data: str) -> 'Event':
        event_type, *event_params = data.split(',')
        event_type = event_type.strip()

        # Event types are allowed to be specified as either integers or
        # strings. try parsing as an int first, and just leave it alone
        # otherwise (our enum instantiation will take care of validation).
        if event_type.isdigit():
            event_type = int(event_type)

        try:
            event_type = EventType(event_type)
        except (ValueError, KeyError):
            return GenericEvent(data)

        parser = {
            EventType.Background: Background.parse,
            EventType.Video: Video.parse,
            EventType.Break: Break.parse,
            EventType.ColourTransformation: ColourTransformation.parse,
            EventType.Sprite: Sprite.parse,
            EventType.Animation: Animation.parse,
            EventType.Sample: Sample.parse,
        }
        return parser[event_type](event_params)


class StoryboardCommand:
    BLOCK_COMMANDS = frozenset(
        {"L", "T"}
    )
    COMMAND_TYPES = frozenset(
        {"F", "M", "MX", "MY", "S", "V", "R", "C", "P", "L", "T"}
    )

    def __init__(
        self,
        command_type: str,
        parameters: list[str],
        subcommands: list['StoryboardCommand'] | None = None,
    ) -> None:
        self.command_type = command_type
        self.parameters = parameters
        self.subcommands = [] if subcommands is None else subcommands

    @property
    def is_block(self) -> bool:
        return self.command_type in self.BLOCK_COMMANDS

    def pack(self, indent_level: int = 1) -> str:
        prefix = " " * indent_level
        line = prefix + ",".join([self.command_type, *self.parameters])
        if not self.subcommands:
            return line

        packed = [line]
        packed.extend(command.pack(indent_level + 1) for command in self.subcommands)
        return "\n".join(packed)

    @classmethod
    def parse_line(cls, data: str) -> tuple['StoryboardCommand', int]:
        indent_level = len(data) - len(data.lstrip(" "))
        stripped = data.lstrip(" ")
        command_type, *parameters = stripped.split(',')
        return cls(command_type.strip(), parameters), indent_level


class StoryboardObject(Event):
    def __init__(
        self,
        event_type: EventType,
        layer: LayerType,
        origin: str,
        filename: str,
        x_offset: int,
        y_offset: int,
        commands: list[StoryboardCommand] | None = None,
    ) -> None:
        super().__init__(event_type, 0)
        self.layer = layer
        self.origin = origin
        self.filename = filename
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.commands = commands or []

    def pack_header(self) -> str:
        return (
            f'{self.event_type.name},{self.layer.name},{self.origin},"{self.filename}",'
            f'{self.x_offset},{self.y_offset}'
        )

    def pack_body(self) -> str:
        return "\n".join(
            command.pack()
            for command in self.commands
        )

    def pack(self) -> str:
        body = self.pack_body()
        if not body:
            return self.pack_header()
        return f"{self.pack_header()}\n{body}"

    @classmethod
    def parse_base_fields(cls, event_params: Sequence[str]) -> tuple[LayerType, str, str, int, int]:
        if len(event_params) < 5:
            raise ValueError(
                f'expected at least 5 params for {cls.__name__}, got {event_params}'
            )

        layer, origin, filename, x_offset, y_offset, *_ = event_params
        filename = filename.strip('"')

        # Similar to event types, layers can be specified
        # as either integers or strings
        if layer.isdigit():
            layer = int(layer)

        try:
            layer_type = LayerType(layer)
        except (ValueError, KeyError):
            raise ValueError(f'Invalid layer provided, got {layer}')

        try:
            x_offset = int(x_offset)
        except ValueError as exc:
            raise ValueError(f'x_offset is invalid, got {x_offset}') from exc

        try:
            y_offset = int(y_offset)
        except ValueError as exc:
            raise ValueError(f'y_offset is invalid, got {y_offset}') from exc

        return layer_type, origin, filename, x_offset, y_offset


class EventCollection:
    def __init__(self, events: list[Event]):
        self.events = events

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def append(self, event: Event):
        self.events.append(event)

    def extend(self, events: list[Event]):
        self.events.extend(events)

    def clear(self):
        self.events.clear()

    @classmethod
    def parse(cls, event_data: list[str]) -> 'EventCollection':
        events: list[Event] = []
        current_storyboard_event: StoryboardObject | None = None
        command_stacks: dict[int, list[tuple[int, StoryboardCommand]]] = {}

        for line in event_data:
            if cls.is_storyboard_command(line) and current_storyboard_event is not None:
                cls.append_storyboard_command(current_storyboard_event, command_stacks, line)
                continue

            event = Event.parse(line.strip())
            events.append(event)
            current_storyboard_event = None

            if isinstance(event, StoryboardObject):
                current_storyboard_event = event

        return cls(events)

    @staticmethod
    def is_storyboard_command(data: str) -> bool:
        stripped = data.lstrip(" ")
        if not stripped:
            return False

        command_type = stripped.split(',', 1)[0]
        return command_type in StoryboardCommand.COMMAND_TYPES

    @staticmethod
    def append_storyboard_command(
        storyboard_event: 'StoryboardObject',
        command_stacks: dict[int, list[tuple[int, StoryboardCommand]]],
        raw_line: str,
    ) -> None:
        command, indent_level = StoryboardCommand.parse_line(raw_line)
        stack = command_stacks.setdefault(id(storyboard_event), [])

        while stack and indent_level <= stack[-1][0]:
            stack.pop()

        if stack:
            stack[-1][1].subcommands.append(command)
        else:
            storyboard_event.commands.append(command)

        if command.is_block:
            stack.append((indent_level, command))

    def pack(self) -> str:
        background_and_videos = [
            event for event in self.events
            if event.event_type in {EventType.Background, EventType.Video}
        ]
        break_periods = [
            event for event in self.events
            if event.event_type == EventType.Break
        ]
        colour_transformations = [
            event for event in self.events
            if event.event_type == EventType.ColourTransformation
        ]
        storyboard_layers = [
            event for event in self.events
            if event.event_type in {EventType.Sprite, EventType.Animation}
        ]
        sound_samples = [
            event for event in self.events
            if event.event_type == EventType.Sample
        ]
        unknown_events = [
            event for event in self.events
            if event.event_type is None
        ]

        packed_str = ""
        packed_str += "//Background and Video events\n"
        for event in background_and_videos:
            packed_str += event.pack() + "\n"

        packed_str += "//Break Periods\n"
        for event in break_periods:
            packed_str += event.pack() + "\n"

        storyboard_layer_descriptions = {
            0: "(Background)",
            1: "(Fail)",
            2: "(Pass)",
            3: "(Foreground)",
            4: "(Overlay)",
        }
        storyboard_layer_mapping = defaultdict(list)

        # These layers are added regardless of if they have events
        storyboard_layer_mapping[0] = []
        storyboard_layer_mapping[1] = []
        storyboard_layer_mapping[2] = []
        storyboard_layer_mapping[3] = []

        for event in storyboard_layers:
            layer_num = int(event.layer)
            storyboard_layer_mapping[layer_num].append(event)

        for layer, events in sorted(storyboard_layer_mapping.items()):
            layer_description = storyboard_layer_descriptions.get(layer, "")
            packed_str += f"//Storyboard Layer {layer} {layer_description}\n"

            for event in events:
                packed_str += event.pack() + "\n"

        packed_str += "//Storyboard Sound Samples\n"
        for event in sound_samples:
            packed_str += event.pack() + "\n"

        if colour_transformations:
            # This comment only seems to be added if there are transformations present
            packed_str += "//Background Colour Transformations\n"
        for event in colour_transformations:
            packed_str += event.pack() + "\n"

        for event in unknown_events:
            packed_str += event.pack() + "\n"

        return packed_str.strip()


class GenericEvent(Event):
    """
    A generic event that just stores the raw data, for event types we don't know how to parse.
    """

    def __init__(self, raw_data: str, event_type: EventType | None = None) -> None:
        super().__init__(event_type, 0, raw_data=raw_data)


class Background(Event):
    def __init__(self, filename: str, x_offset: int, y_offset: int) -> None:
        super().__init__(EventType.Background, 0)
        self.filename = filename
        self.x_offset = x_offset
        self.y_offset = y_offset

    def pack(self) -> str:
        return (
            f'0,{self.delta_to_ms(self.start_time)},"{self.filename}",'
            f'{self.x_offset},{self.y_offset}'
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'Background':
        if not event_params:
            raise ValueError('expected start_time parameter for Background')

        start_time = event_params[0]
        try:
            start_time_int = int(start_time)
        except ValueError as exc:
            raise ValueError(f'Invalid start_time provided, got {start_time}') from exc

        event_params = event_params[1:]
        if not event_params:
            raise ValueError('expected filename parameter for Background')

        filename = event_params[0].strip('"')
        x_offset_int = 0
        y_offset_int = 0

        if len(event_params) > 1:
            x_offset = event_params[1]
        if len(event_params) > 2:
            y_offset = event_params[2]

        if len(event_params) > 3:
            raise ValueError(
                "expected no more than 3 params for Background, "
                f"but got params {event_params}"
            )

        try:
            x_offset_int = int(x_offset)
        except ValueError as exc:
            raise ValueError(f'x_offset is invalid, got {x_offset}') from exc

        try:
            y_offset_int = int(y_offset)
        except ValueError as exc:
            raise ValueError(f'y_offset is invalid, got {y_offset}') from exc

        event = cls(filename, x_offset_int, y_offset_int)
        event.start_time = timedelta(milliseconds=start_time_int)
        return event


class Break(Event):
    def __init__(self, start_time: int, end_time: int) -> None:
        super().__init__(EventType.Break, start_time)
        self.end_time = timedelta(milliseconds=end_time)

    def pack(self) -> str:
        return (
            f'2,{self.delta_to_ms(self.start_time)},'
            f'{self.delta_to_ms(self.end_time)}'
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'Break':
        if len(event_params) < 2:
            raise ValueError('expected start_time and end_time parameters for Break')

        start_time = event_params[0]
        end_time = event_params[1]

        try:
            start_time_int = int(start_time)
        except ValueError as exc:
            raise ValueError(f'Invalid start_time provided, got {start_time}') from exc

        try:
            end_time_int = int(end_time)
        except ValueError as exc:
            raise ValueError(f'Invalid end_time provided, got {end_time}') from exc

        return cls(start_time_int, end_time_int)


class ColourTransformation(Event):
    def __init__(self, start_time: int, red: int, green: int, blue: int) -> None:
        super().__init__(EventType.ColourTransformation, start_time)
        self.red = red
        self.green = green
        self.blue = blue

    def pack(self) -> str:
        return (
            f'3,{self.delta_to_ms(self.start_time)},'
            f'{self.red},{self.green},{self.blue}'
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'ColourTransformation':
        if len(event_params) < 4:
            raise ValueError(
                'expected start_time, red, green and blue parameters '
                'for ColourTransformation'
            )

        start_time, red, green, blue = event_params[:4]
        try:
            return cls(int(start_time), int(red), int(green), int(blue))
        except ValueError as exc:
            raise ValueError(
                'ColourTransformation parameters must be integers, got '
                f'{event_params[:4]}'
            ) from exc


class Video(Event):
    def __init__(self, start_time: int, filename: str, x_offset: int, y_offset: int) -> None:
        super().__init__(EventType.Video, start_time)
        self.filename = filename
        self.x_offset = x_offset
        self.y_offset = y_offset

    def pack(self) -> str:
        return (
            f'Video,{self.delta_to_ms(self.start_time)},"{self.filename}",'
            f'{self.x_offset},{self.y_offset}'
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'Video':
        if not event_params:
            raise ValueError('expected start_time parameter for Video')

        try:
            start_time = event_params[0]
            start_time_int = int(start_time)
        except ValueError as exc:
            raise ValueError(f'Invalid start_time provided, got {start_time}') from exc

        event_params = event_params[1:]
        if not event_params:
            raise ValueError('expected filename parameter for Video')

        filename = event_params[0].strip('"')
        x_offset_int = 0
        y_offset_int = 0

        if len(event_params) > 1:
            x_offset = event_params[1]
        if len(event_params) > 2:
            y_offset = event_params[2]

        if len(event_params) > 3:
            raise ValueError(
                "expected no more than 3 params for Video, "
                f"but got params {event_params}"
            )

        try:
            x_offset_int = int(x_offset)
        except ValueError as exc:
            raise ValueError(f'x_offset is invalid, got {x_offset}') from exc

        try:
            y_offset_int = int(y_offset)
        except ValueError as exc:
            raise ValueError(f'y_offset is invalid, got {y_offset}') from exc

        return cls(start_time_int, filename, x_offset_int, y_offset_int)


class Sprite(StoryboardObject):
    def __init__(
        self,
        layer: LayerType,
        origin: str,
        filename: str,
        x_offset: int,
        y_offset: int,
        commands: list[StoryboardCommand] | None = None,
    ) -> None:
        super().__init__(
            EventType.Sprite,
            layer,
            origin,
            filename,
            x_offset,
            y_offset,
            commands,
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'Sprite':
        layer, origin, filename, x_offset, y_offset = cls.parse_base_fields(event_params)
        return cls(layer, origin, filename, x_offset, y_offset)


class Animation(StoryboardObject):
    def __init__(
        self,
        layer: LayerType,
        origin: str,
        filename: str,
        x_offset: int,
        y_offset: int,
        frame_count: int,
        frame_delay: int,
        loop_type: str = 'LoopForever',
        commands: list[StoryboardCommand] | None = None,
    ) -> None:
        super().__init__(
            EventType.Animation,
            layer,
            origin,
            filename,
            x_offset,
            y_offset,
            commands,
        )
        self.frame_count = frame_count
        self.frame_delay = frame_delay
        self.loop_type = loop_type

    def pack_header(self) -> str:
        return (
            f'{self.event_type.name},{self.layer.name},{self.origin},"{self.filename}",'
            f'{self.x_offset},{self.y_offset},{self.frame_count},'
            f'{self.frame_delay},{self.loop_type}'
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'Animation':
        layer, origin, filename, x_offset, y_offset = cls.parse_base_fields(event_params)
        if len(event_params) < 7:
            raise ValueError(
                'expected at least 7 params for Animation '
                '(layer, origin, filename, x, y, frame_count, frame_delay)'
            )

        frame_count = event_params[5]
        frame_delay = event_params[6]
        loop_type = event_params[7] if len(event_params) > 7 else 'LoopForever'

        try:
            frame_count_int = int(frame_count)
        except ValueError as exc:
            raise ValueError(f'frame_count is invalid, got {frame_count}') from exc

        try:
            frame_delay_int = int(frame_delay)
        except ValueError as exc:
            raise ValueError(f'frame_delay is invalid, got {frame_delay}') from exc

        return cls(
            layer,
            origin,
            filename,
            x_offset,
            y_offset,
            frame_count_int,
            frame_delay_int,
            loop_type,
        )


class Sample(Event):
    def __init__(
        self,
        start_time: int,
        layer: LayerType,
        filename: str,
        volume: int = 100,
    ) -> None:
        super().__init__(EventType.Sample, start_time)
        self.layer = layer
        self.filename = filename
        self.volume = volume

    def pack(self) -> str:
        return (
            f'5,{self.delta_to_ms(self.start_time)},'
            f'{int(self.layer)},"{self.filename}",{self.volume}'
        )

    @classmethod
    def parse(cls, event_params: list[str]) -> 'Sample':
        if len(event_params) < 3:
            raise ValueError(
                'expected start_time, layer, and filename parameters for Sample'
            )

        start_time, layer, filename, *rest = event_params
        filename = filename.strip('"')

        # Similar to event types, layers can be specified
        # as either integers or strings
        if layer.isdigit():
            layer = int(layer)

        try:
            layer_type = LayerType(layer)
        except (ValueError, KeyError):
            raise ValueError(f'Invalid layer provided, got {layer}')

        try:
            start_time_int = int(start_time)
        except ValueError as exc:
            raise ValueError(f'Invalid start_time provided, got {start_time}') from exc

        volume_int = 100

        if rest:
            try:
                volume = rest[0]
                volume_int = int(volume)
            except ValueError as exc:
                raise ValueError(f'Invalid volume provided, got {volume}') from exc

        return cls(start_time_int, layer_type, filename, volume_int)
