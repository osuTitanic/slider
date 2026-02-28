from slider.beatmap import Beatmap
from slider.events import (
    Animation,
    ColourTransformation,
    EventCollection,
    LayerType,
    Sample,
    Sprite,
)


def test_storyboard_sprite_parsing():
    events = EventCollection.parse([
        '4,0,1,"bg.jpg",320,264',
        ' L,4722,16',
        '  S,0,0,3740,1.5,1.9',
        '  S,0,3740,7480,1.9,1.5',
        ' F,1,4722,5470,0,1',
    ])

    sprite = events[0]
    assert isinstance(sprite, Sprite)
    assert sprite.layer == LayerType.Background
    assert len(sprite.commands) == 2

    loop = sprite.commands[0]
    assert loop.command_type == "L"
    assert len(loop.subcommands) == 2
    assert [command.command_type for command in loop.subcommands] == ["S", "S"]

    fade = sprite.commands[1]
    assert fade.command_type == "F"
    assert fade.parameters == ["1", "4722", "5470", "0", "1"]


def test_animation_and_sample_parsing():
    events = EventCollection.parse([
        '6,0,1,"sliderb.png",207,270,5,200',
        ' M,0,60089,65701,207,270,437,270',
        '5,5844,0,"dake\\Part 1 - Hit Circle.mp3"',
        '3,100,163,162,255',
    ])

    assert isinstance(events[0], Animation)
    assert events[0].layer == LayerType.Background
    assert events[0].frame_count == 5
    assert events[0].frame_delay == 200
    assert isinstance(events[1], Sample)
    assert events[1].layer == LayerType.Background
    assert events[1].volume == 100
    assert isinstance(events[2], ColourTransformation)


def test_find_groups_preserves_events_indentation():
    groups = Beatmap._find_groups([
        "[Events]",
        '4,0,1,"bg.jpg",320,264',
        ' L,4722,16',
        '  S,0,0,3740,1.5,1.9',
        "[HitObjects]",
    ])

    event_lines = groups["Events"]
    assert event_lines[1].startswith(" ")
    assert event_lines[2].startswith("  ")
