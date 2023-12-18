# Author: Çağrı Erdem, 2023
# Description: MIDI broadcasting script for 2groove web app.

import os
import queue
import threading

import clockblocks
import rtmidi

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import signal
import threading

from agents import play_agents

####################
## MIDI BROADCAST ##
####################

# Global control events
generation_queue = queue.Queue(maxsize=10)
pause_event = threading.Event()
stop_event = threading.Event()
change_groove_event = threading.Event()

config_update_event = threading.Event()

global_config = {}
current_bpm = 120
current_loop_count = 1


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


def set_new_channels(config):
    if config["PLAY_DRUM"]:
        CHANNELS["drum"] = [0x90, 0x80]
    else:
        CHANNELS["drum"] = [0x80, 0x80]
    if config["PLAY_BASS"]:
        CHANNELS["bass"] = [0x91, 0x81]
    else:
        CHANNELS["bass"] = [0x81, 0x81]
    if config["PLAY_CHORD"]:
        CHANNELS["chord"] = [0x92, 0x82]
    else:
        CHANNELS["chord"] = [0x82, 0x82]
    if config["PLAY_MELODY"]:
        CHANNELS["melody"] = [0x93, 0x83]
    else:
        CHANNELS["melody"] = [0x83, 0x83]
    if config["PLAY_HARMONY"]:
        CHANNELS["harmony"] = [0x94, 0x84]
    else:
        CHANNELS["harmony"] = [0x84, 0x84]


def broadcasting_loop(
    generation_queue,
    stop_event,
    config,
    virtual_port=True,
    introduce_delay=False,
    verbose=False,
):
    """This is a MIDI broadcasting loop implementation in terms of synchronization & time.
    It uses the clockblocks clock to synchronize the groove loops."""
    global desired_loops, current_bpm

    set_new_channels(config)

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

                set_new_channels(config)
                tempo_in_seconds_per_tick = current_tempo / MS_PER_SEC / ticks_per_beat
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
                if stop_event.is_set():
                    break
                while pause_event.is_set():
                    master_clock.wait(0.1, units="time")
                timestamp, event_type, pitch, velocity, instrument_name = event

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


def music_generation_thread():
    global global_config
    while True:
        config_update_event.wait()  # Wait for a signal to generate new music
        config_update_event.clear()  # Reset the event
        print("Generating music...")
        # Generate music with the latest configuration
        print(global_config)
        pm = play_agents(global_config)
        add_to_queue(pm)
        generation_queue.put(pm)
        change_groove_event.set()


# Initialization of global events and queues
midi_app = Flask(__name__)  # Connect to the browser interface
CORS(midi_app)


@midi_app.route("/set_params", methods=["POST"])
def set_params():
    global current_loop_count, global_config

    data = request.json

    global_config = {
        "TEMPO": int(data.get("tempo", 120)),
        "LENGTH": int(data.get("length", 12)),
        "PLAY_DRUM": data.get("play_drum", True),
        "LOOP_MEASURES": int(data.get("loop_measures", 4)),
        "STYLE": data.get("style", "country"),
        "PLAY_BASS": data.get("play_bass", True),
        "DURATION_PREFERENCES_BASS": data.get("duration_preferences_bass", False),
        "PLAYSTYLE": data.get("playstyle", "bass_drum"),
        "PLAY_CHORD": data.get("play_chord", True),
        "ARPEGIATE_CHORD": data.get("arpegiate_chord", False),
        "BOUNCE_CHORD": data.get("bounce_chord", False),
        "ARP_STYLE": int(data.get("arp_style", 2)),
        "PLAY_MELODY": data.get("play_melody", True),
        "NOTE_TEMPERATURE_MELODY": float(data.get("note_temperature_melody", 0.8)),
        "DURATION_TEMPERATURE_MELODY": float(
            data.get("duration_temperature_melody", 0.8)
        ),
        "NO_PAUSE": data.get("no_pause", False),
        "SCALE_MELODY": data.get("scale_melody", "major pentatonic"),
        "DURATION_PREFERENCES_MELODY": data.get(
            "duration_preferences_melody", [1, 3, 5, 7, 9, 11, 13, 15]
        ),
        "PLAY_HARMONY": data.get("play_harmony", True),
        "INTERVAL_HARMONY": int(data.get("interval_harmony", 5)),
    }

    pm = play_agents(global_config)
    add_to_queue(pm)
    generation_queue.put(pm)
    change_groove_event.set()

    # config_update_event.set()

    current_loop_count = 0
    return jsonify({"message": "Processing MIDI file..."})


@midi_app.route("/control", methods=["POST"])
def control():
    action = request.json.get("action", "")
    if action == "pause":
        pause_event.set()
    elif action == "resume":
        pause_event.clear()

    elif action == "stop":
        os.kill(os.getpid(), signal.SIGINT)  # similar to cmd+C
        return jsonify({"message": f"Action {action} processed, server stopped"})

    return jsonify({"message": f"Action {action} processed"})


@midi_app.route("/shutdown", methods=["POST"])
def shutdown():
    return "Server shutting down..."


@midi_app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response


def start_broadcaster(config):
    print("---Starting the MIDI broadcaster---")
    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(
        target=lambda: midi_app.run(threaded=True, port=5005)
    )
    flask_thread.daemon = True
    flask_thread.start()

    # Start the broadcasting loop in a separate thread
    broadcasting_thread = threading.Thread(
        target=broadcasting_loop, args=(generation_queue, stop_event, config)
    )
    broadcasting_thread.daemon = True
    broadcasting_thread.start()

    # generation_thread = threading.Thread(target=music_generation_thread)
    # generation_thread.daemon = True
    # generation_thread.start()
    print("MIDI broadcaster started")


def add_to_queue(obj):
    generation_queue.put(obj)
