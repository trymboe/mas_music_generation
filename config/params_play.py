# General parameters
TEMPO = 120
LENGTH = 24  # Number of measures to be generated
LOOP_MEASURES = 4

# Drum parameters
STYLE = "country"

# Bass parameters
DURATION_PREFERENCES_BASS = [4]  # in number of beats (1-8)
PLAYSTYLE = "bass_drum"  # "bass_drum" or False

# Chord parameters
ARPEGIATE_CHORD = False
BOUNCE_CHORD = False
ARP_STYLE = 2  # Style of the arpegiator 0 for 16th notes, 1 for 12th note, 2 for 8th notes, 3 for full range 16th notes

# Melody parameters
NOTE_TEMPERATURE_MELODY = 0.8
DURATION_TEMPERATURE_MELODY = 0.8
NO_PAUSE = False
SCALE_MELODY = "major pentatonic"  # "major pentatonic"  # "major scale"",
DURATION_PREFERENCES_MELODY = [1, 3, 5, 7, 9, 11, 13, 15]  # number of quarterbeats


# Segments
SEGMENTS = [
    {
        "TEMPO": 120,
        "LENGTH": 4,
        "LOOP_MEASURES": 4,
        "STYLE": "country",
        "DURATION_PREFERENCES_BASS": [4],
        "PLAYSTYLE": "bass_drum",
        "ARPEGIATE_CHORD": True,
        "BOUNCE_CHORD": False,
        "ARP_STYLE": 2,
        "NOTE_TEMPERATURE_MELODY": 1.5,
        "DURATION_TEMPERATURE_MELODY": 0.8,
        "NO_PAUSE": True,
        "SCALE_MELODY": "major scale",
        "DURATION_PREFERENCES_MELODY": [11],
    },
    {
        "TEMPO": 120,
        "LENGTH": 4,
        "LOOP_MEASURES": 4,
        "STYLE": "country",
        "DURATION_PREFERENCES_BASS": [4],
        "PLAYSTYLE": "bass_drum",
        "ARPEGIATE_CHORD": True,
        "BOUNCE_CHORD": False,
        "ARP_STYLE": 2,
        "NOTE_TEMPERATURE_MELODY": 1.5,
        "DURATION_TEMPERATURE_MELODY": 0.8,
        "NO_PAUSE": True,
        "SCALE_MELODY": "major scale",
        "DURATION_PREFERENCES_MELODY": [7],
    },
    {
        "TEMPO": 120,
        "LENGTH": 12,
        "LOOP_MEASURES": 4,
        "STYLE": "country",
        "DURATION_PREFERENCES_BASS": False,
        "PLAYSTYLE": "bass_drum",
        "ARPEGIATE_CHORD": False,
        "BOUNCE_CHORD": False,
        "ARP_STYLE": 2,
        "NOTE_TEMPERATURE_MELODY": 0.8,
        "DURATION_TEMPERATURE_MELODY": 0.8,
        "NO_PAUSE": False,
        "SCALE_MELODY": "major pentatonic",
        "DURATION_PREFERENCES_MELODY": [1, 3, 5, 7, 9, 11, 13, 15],
    },
    {
        "TEMPO": 120,
        "LENGTH": 2,
        "LOOP_MEASURES": 2,
        "STYLE": "country",
        "DURATION_PREFERENCES_BASS": [4],
        "PLAYSTYLE": "bass_drum",
        "ARPEGIATE_CHORD": False,
        "BOUNCE_CHORD": False,
        "ARP_STYLE": 2,
        "NOTE_TEMPERATURE_MELODY": 1.2,
        "DURATION_TEMPERATURE_MELODY": 0.8,
        "NO_PAUSE": True,
        "SCALE_MELODY": "major pentatonic",
        "DURATION_PREFERENCES_MELODY": [11],
    },
]
