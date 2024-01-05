# Thanks to: Çağrı Erdem for the initial implementation of the MIDI broadcasting loop.

import clockblocks
import rtmidi

from agents import play_agents


from multiprocessing import Value, Process, Queue as mpQueue, Event as mpEvent


####################
## MIDI BROADCAST ##
####################

# Global control events


global_config = {}
current_bpm = 120
current_loop_count = 1
desired_loops = 0

is_drum_muted = False
is_bass_muted = False
is_chord_muted = False
is_melody_muted = False
is_harmony_muted = False


# Constants
MS_PER_SEC = 1_000_000  # microseconds per second
BARS = 2
BEATS_PER_BAR = 4  # 4/4 time signature
BEAT_DURATION = 60 / current_bpm  # in seconds

# Chanels
CHANNELS = {
    "drum": [0x90, 0x80],
    "bass": [0x91, 0x81],
    "chord": [0x92, 0x82],
    "melody": [0x93, 0x83],
    "harmony": [0x94, 0x84],
}

VOLUME_DICT = {"drum": 1, "bass": 1, "chord": 1, "melody": 1, "harmony": 1}


def midi2events(midi_obj):
    """(docstring)"""
    events = []
    tempo = None
    ticks_per_beat = midi_obj.ticks_per_beat
    for track in midi_obj.tracks:
        for msg in track:
            if msg.type == "note_on" or msg.type == "note_off":
                events.append((msg.time, msg.type, msg.note, msg.velocity))
            elif msg.type == "set_tempo":
                tempo = msg.tempo
    return events, tempo, ticks_per_beat


def pretty_midi2events(pretty_midi_obj):
    """Convert pretty_midi object to a sequence of events."""
    events = []

    tempo_changes = pretty_midi_obj.get_tempo_changes()
    tempos = tempo_changes[1]
    times = tempo_changes[0]
    ticks_per_beat = pretty_midi_obj.resolution

    # Assuming the tempo does not change during the piece, use the first tempo
    # If there are no tempo changes, default to 120 BPM
    tempo = int(tempos[0]) if len(tempos) > 0 else 120

    for instrument in pretty_midi_obj.instruments:
        for note in instrument.notes:
            # Convert start and end times to ticks
            start_tick = pretty_midi_obj.time_to_tick(note.start)
            end_tick = pretty_midi_obj.time_to_tick(note.end)
            duration_ticks = end_tick - start_tick

            events.append(
                (start_tick, "note_on", note.pitch, note.velocity, instrument.name)
            )
            events.append(
                (
                    start_tick + duration_ticks,
                    "note_off",
                    note.pitch,
                    0,
                    instrument.name,
                )
            )

    # Sort events by time
    events.sort(key=lambda x: x[0])

    return events, tempo, ticks_per_beat


def generate_midi_message(event_type, pitch, velocity, channel1, channel2):
    event_map = {"note_on": channel1, "note_off": channel2}

    return [event_map.get(event_type, event_type), pitch, velocity]


def set_new_channels(muted):
    if not muted[0]:
        CHANNELS["drum"] = [0x90, 0x80]
    else:
        CHANNELS["drum"] = [0x80, 0x80]
    if not muted[1]:
        CHANNELS["bass"] = [0x91, 0x81]
    else:
        CHANNELS["bass"] = [0x81, 0x81]
    if not muted[2]:
        CHANNELS["chord"] = [0x92, 0x82]
    else:
        CHANNELS["chord"] = [0x82, 0x82]
    if not muted[3]:
        CHANNELS["melody"] = [0x93, 0x83]
    else:
        CHANNELS["melody"] = [0x83, 0x83]
    if not muted[4]:
        CHANNELS["harmony"] = [0x94, 0x84]
    else:
        CHANNELS["harmony"] = [0x84, 0x84]


def set_volume(instrument, volume):
    VOLUME_DICT[instrument] = volume


def broadcasting_loop(
    generation_queue,
    stop_event,
    start_event,
    change_groove_event,
    muted,
    virtual_port=True,
    verbose=False,
):
    """This is a MIDI broadcasting loop implementation in terms of synchronization & time.
    It uses the clockblocks clock to synchronize the groove loops."""
    global desired_loops, current_bpm, global_config

    while not change_groove_event.is_set():
        pass

    set_new_channels(muted)

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    if virtual_port:
        midiout.open_virtual_port("dB virtual output")
        if verbose:
            print("Using dB virtual MIDI output")
    else:
        midiport = input("Enter the MIDI port")
        midiout.open_port(midiport)
        if verbose:
            print(f"Using {midiport} as the MIDI port")
    current_midi_events = []

    def compute_groove_duration(current_tempo, ticks_per_beat, total_ticks):
        """Computes the total duration of the groove in seconds."""
        tempo_in_seconds_per_beat = current_tempo / MS_PER_SEC
        total_duration = tempo_in_seconds_per_beat * (total_ticks / ticks_per_beat)
        return total_duration

    current_tempo = int(
        60_000_000 / current_bpm
    )  # Convert BPM to microseconds per beat
    current_loop_count = 0
    new_groove_queued = (
        False  # This flag is set to True when a new groove enters the queue
    )

    midi_obj = generation_queue.get()
    current_midi_events, current_tempo, ticks_per_beat = pretty_midi2events(midi_obj)
    microseconds_per_beat = 60_000_000 / current_bpm
    tempo_in_seconds_per_tick = microseconds_per_beat / MS_PER_SEC / ticks_per_beat

    # Initialize master clock
    master_clock = clockblocks.Clock(
        timing_policy=0, initial_tempo=current_bpm
    ).run_as_server()  # 0 is equivalent to absolute timing, 1 is equivalent to relative timing.
    reference_start_time = master_clock.time()

    try:
        current_loop_count = 0  # Initialize loop count
        while not stop_event.is_set():
            total_ticks = sum(event[0] for event in current_midi_events)

            # If there's a new groove queued up, don't process it immediately.
            # Just mark that a new groove is waiting. Wait for the current groove to loop for the desired number of times.
            if change_groove_event.is_set() and not generation_queue.empty():
                new_groove_queued = True
                change_groove_event.clear()  # Reset the event
                current_loop_count = 0  # Reset the loop count
                print(
                    f"Detected a new groove queued – waiting for the current groove to loop {desired_loops} times"
                )
            # First loop of the groove for the desired number of times, then switch to the new groove
            if new_groove_queued and current_loop_count >= desired_loops:
                midi_obj = generation_queue.get_nowait()
                current_midi_events, current_tempo, ticks_per_beat = pretty_midi2events(
                    midi_obj
                )

                microseconds_per_beat = 60_000_000 / current_bpm
                tempo_in_seconds_per_tick = (
                    microseconds_per_beat / MS_PER_SEC / ticks_per_beat
                )
                print("Switched to the new groove")
                new_groove_queued = False  # Reset the flag
                current_loop_count = 0  # Reset the loop count

            master_clock.tempo = current_bpm  # Update the tempo
            if verbose:
                print(f"Master clock tempo: {master_clock.absolute_tempo()} BPM")
            groove_duration = compute_groove_duration(
                current_tempo, ticks_per_beat, total_ticks
            )
            # Compute the expected start time for this loop based on the reference
            expected_start_time = reference_start_time + (
                current_loop_count * groove_duration
            )

            # If we're ahead of the expected start time, wait
            while master_clock.time() < expected_start_time:
                master_clock.wait(
                    0.01, units="time"
                )  # Wait in small increments to be ready #TODO: Check the efficiency of this

            # Broadcast the current MIDI events.

            previous_timestamp = 0
            wait_time_in_seconds = 0
            supposed_clock_time = 0

            for idx, event in enumerate(current_midi_events):
                while start_event.is_set() is False:
                    pass
                if stop_event.is_set():
                    break
                timestamp, event_type, pitch, velocity, instrument_name = event

                velocity = int(velocity * VOLUME_DICT[instrument_name])

                message = generate_midi_message(
                    event_type,
                    pitch,
                    velocity,
                    CHANNELS[instrument_name][0],
                    CHANNELS[instrument_name][1],
                )
                midiout.send_message(message)

                if idx == len(current_midi_events) - 1:
                    continue
                duration = current_midi_events[idx + 1][0] - previous_timestamp
                wait_time_in_seconds = duration * tempo_in_seconds_per_tick

                supposed_clock_time += wait_time_in_seconds
                if wait_time_in_seconds > 0:
                    # print(f"Waiting for {wait_time_in_seconds} seconds")
                    master_clock.wait(wait_time_in_seconds, units="time")
                previous_timestamp = current_midi_events[idx + 1][0]

            current_loop_count += 1
            print(f"Current groove looped {current_loop_count} times")

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        del midiout


# In the music_generation_process function
def music_generation_process(
    config_queue, generation_queue, change_groove_event, generation_is_complete
):
    # TODO, not use global config, get config from queue
    global global_config
    while True:
        global_config = config_queue.get()  # Blocking call
        pm = play_agents(global_config)
        generation_queue.put(pm)
        change_groove_event.set()
        generation_is_complete.set()
        print("Generation complete")
        print(generation_is_complete.is_set())
