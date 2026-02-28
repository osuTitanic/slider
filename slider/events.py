
from datetime import timedelta
from enum import IntEnum


class EventType(IntEnum):
    Background = 0
    Video = 1
    Break = 2
    Sprite = 3
    Animation = 4

    # TODO I have absolutely no idea if Sample is really supposed to be 5.
    # See https://osu.ppy.sh/beatmapsets/14902#osu/54879 for a map with a sample event
    Sample = 5

    @classmethod
    def _missing_(cls, value):
        return {
            "Background": EventType.Background,
            "Video": EventType.Video,
            "Break": EventType.Break,
            "Sprite": EventType.Sprite,
            "Animation": EventType.Animation,
            "Sample": EventType.Sample,
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
        
        # By default, we don't know how to pack an event, so if we don't
        # have the data to just return, raise an error.
        raise NotImplementedError(f"pack is not implemented for {type(self).__name__}")

    @classmethod
    def parse(cls, data) -> 'Event':
        event_type, start_time_or_layer, *event_params = data.split(',')

        # Event types are allowed to be specified as either integers or
        # strings. try parsing as an int first, and just leave it alone
        # otherwise (our enum instantiation will take care of validation).
        if event_type.isdigit():
            event_type = int(event_type)

        try:
            event_type = EventType(event_type)
        except (ValueError, KeyError):
            return GenericEvent(data)

        storyboard_types = {
            EventType.Sprite: Sprite,
            EventType.Animation: Animation,
            EventType.Sample: Sample,
        }

        if event_type in storyboard_types:
            return storyboard_types[event_type](data)

        try:
            start_time = int(start_time_or_layer)
        except ValueError:
            raise ValueError(f'Invalid start_time provided, got {start_time_or_layer}')

        map_types = {
            EventType.Background: Background,
            EventType.Video: Video,
            EventType.Break: Break,
        }

        if event_type in map_types:
            event = map_types[event_type].parse(start_time, event_params)
            event.raw_data = data
            return event

        # Ensure we've handled all event types.
        raise ValueError(f'Unimplemented event type: {event_type}')


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

    def pack(self) -> str:
        return '\n'.join(event.pack() for event in self.events)

    @classmethod
    def parse(cls, event_data: list[str]) -> 'EventCollection':
        return cls([Event.parse(line) for line in event_data])


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
    def parse(cls, start_time: int, event_params: list[str]) -> 'Background':
        if len(event_params) == 0:
            raise ValueError('expected filename parameter for Background')

        filename = event_params[0].strip('"')
        x_offset = 0
        y_offset = 0

        # x_offset and y_offset are optional, default to 0
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
            x_offset = int(x_offset)
        except ValueError:
            raise ValueError(f'x_offset is invalid, got {x_offset}')

        try:
            y_offset = int(y_offset)
        except ValueError:
            raise ValueError(f'y_offset is invalid, got {y_offset}')

        return cls(filename, x_offset, y_offset)


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
    def parse(cls, start_time: int, event_params: list[str]) -> 'Break':
        if not event_params:
            raise ValueError('expected end_time paramter for Break')

        try:
            end_time = int(event_params[0])
        except ValueError:
            raise ValueError(f'Invalid end_time provided, got {end_time}')

        return cls(start_time, end_time)


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
    def parse(cls, start_time: int, event_params: list[str]) -> 'Video':
        if len(event_params) == 0:
            raise ValueError('expected filename parameter for Video')

        filename = event_params[0].strip('"')
        x_offset = 0
        y_offset = 0

        # x_offset and y_offset are optional, default to 0
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
            x_offset = int(x_offset)
        except ValueError:
            raise ValueError(f'x_offset is invalid, got {x_offset}')

        try:
            y_offset = int(y_offset)
        except ValueError:
            raise ValueError(f'y_offset is invalid, got {y_offset}')

        return cls(start_time, filename, x_offset, y_offset)


# TODO Implement storyboard event parsing & packing

class Sprite(GenericEvent):
    def __init__(self, raw_data: str) -> None:
        super().__init__(raw_data, EventType.Sprite)


class Animation(GenericEvent):
    def __init__(self, raw_data: str) -> None:
        super().__init__(raw_data, EventType.Animation)


class Sample(GenericEvent):
    def __init__(self, raw_data: str) -> None:
        super().__init__(raw_data, EventType.Sample)
