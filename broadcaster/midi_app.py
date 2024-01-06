from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Process, Queue as mpQueue, Event as mpEvent

import os
import threading

import signal

from .utils import (
    get_duration_preferences_bass,
    get_duration_temperature_melody,
    get_note_temperature_melody,
    get_duration_preferences_bass_from_advanced,
    get_duration_preferences_melody_from_advanced,
)
from .broadcaster import (
    set_new_channels,
    broadcasting_loop,
    music_generation_process,
    set_volume,
)

start_event = mpEvent()
pause_event = mpEvent()
stop_event = mpEvent()
generation_queue = mpQueue(maxsize=10)
generation_is_complete = mpEvent()
change_groove_event = mpEvent()

is_playing = False

is_drum_muted, is_bass_muted, is_chord_muted, is_melody_muted, is_harmony_muted = (
    False,
    False,
    False,
    False,
    False,
)

is_drum_kept, is_bass_kept, is_chord_kept, is_melody_kept, is_harmony_kept = (
    False,
    False,
    False,
    False,
    False,
)

# Initialization of global events and queues
midi_app = Flask(__name__)  # Connect to the browser interface
CORS(midi_app)


@midi_app.route("/set_params", methods=["POST"])
def set_params():
    global current_loop_count, global_config, config_queue

    data = request.json
    advanced_option = data.get("advanced_option", False)
    if not advanced_option:
        duration_slider_bass = int(data.get("bass_creativity"))
        duration_preferences_bass = get_duration_preferences_bass(duration_slider_bass)

        creative_slider_pitch_melody = int(data.get("pitch_creativity_melody"))
        creative_slider_duration_melody = int(data.get("duration_creativity_melody"))

        note_temperature_melody, scale_melody = get_note_temperature_melody(
            creative_slider_pitch_melody
        )
        (
            duration_temperature_melody,
            duration_preferences_melody,
        ) = get_duration_temperature_melody(creative_slider_duration_melody)

    else:
        note_temperature_melody = int(data.get("note_temperature_melody"))
        duration_temperature_melody = int(data.get("duration_temperature_melody"))
        scale_melody = data.get("scale_melody")
        if scale_melody == "None":
            scale_melody = False

        duration_preferences_bass = get_duration_preferences_bass_from_advanced(
            data.get("checkbox1"),
            data.get("checkbox2"),
            data.get("checkbox3"),
            data.get("checkbox4"),
            data.get("checkbox5"),
            data.get("checkbox6"),
            data.get("checkbox7"),
            data.get("checkbox8"),
        )

        duration_preferences_melody = get_duration_preferences_melody_from_advanced(
            data.get("checkbox1_melody"),
            data.get("checkbox2_melody"),
            data.get("checkbox3_melody"),
            data.get("checkbox4_melody"),
            data.get("checkbox5_melody"),
            data.get("checkbox6_melody"),
        )
    global_config = {
        # General parameters
        "TEMPO": int(data.get("tempo", 120)),
        "LENGTH": int(data.get("length", 4)),
        # Drum parameters
        "KEEP_DRUM": data.get("keep_drum", False),
        "LOOP_MEASURES": int(data.get("loop_measures", 4)),
        "STYLE": data.get("style", "country"),
        # Bass parameters
        "KEEP_BASS": data.get("keep_bass", False),
        "DURATION_PREFERENCES_BASS": duration_preferences_bass,
        "PLAYSTYLE": data.get("playstyle", "bass_drum"),
        # Chord parameters
        "KEEP_CHORD": data.get("keep_chord", False),
        "ARPEGIATE_CHORD": data.get("arpegiate_chord", False),
        "BOUNCE_CHORD": data.get("bounce_chord", False),
        "ARP_STYLE": int(data.get("arp_style", 2)),
        # Melody parameters
        "KEEP_MELODY": data.get("keep_melody", False),
        "NOTE_TEMPERATURE_MELODY": note_temperature_melody,
        "DURATION_TEMPERATURE_MELODY": duration_temperature_melody,
        "NO_PAUSE": data.get("no_pause", False),
        "SCALE_MELODY": scale_melody,
        "DURATION_PREFERENCES_MELODY": duration_preferences_melody,
        # Harmony parameters
        "INTERVAL_HARMONY": int(data.get("interval_harmony", 5)),
    }
    config_queue.put(global_config)
    current_loop_count = 0
    return jsonify({"message": "Processing MIDI file..."})


@midi_app.route("/mute", methods=["POST"])
def mute():
    global is_drum_muted, is_bass_muted, is_chord_muted, is_melody_muted, is_harmony_muted
    data = request.get_json()
    instrument = data.get("instrument")
    mute_state = data.get("mute")

    if instrument == "drum":
        is_drum_muted = mute_state
    elif instrument == "bass":
        is_bass_muted = mute_state
    elif instrument == "chord":
        is_chord_muted = mute_state
    elif instrument == "melody":
        is_melody_muted = mute_state
    elif instrument == "harmony":
        is_harmony_muted = mute_state
    set_new_channels(
        [
            is_drum_muted,
            is_bass_muted,
            is_chord_muted,
            is_melody_muted,
            is_harmony_muted,
        ]
    )

    return (
        jsonify({"status": "success", "instrument": instrument, "muted": mute_state}),
        200,
    )


@midi_app.route("/update_volume", methods=["POST"])
def update_volume():
    data = request.get_json()
    instrument = data.get("instrument").split("-")[1]
    volume = float(data.get("volume"))

    set_volume(instrument, volume)

    return (
        jsonify({"status": "success", "instrument": instrument, "new_volume": volume}),
        200,
    )


@midi_app.route("/check_status", methods=["GET"])
def check_status():
    global generation_is_complete
    is_complete = generation_is_complete.is_set()
    return jsonify({"isComplete": is_complete})


@midi_app.route("/shutdown", methods=["POST"])
def shutdown():
    global gen_process
    gen_process.terminate()
    gen_process.join()
    return "Server shutting down..."


@midi_app.route("/acknowledge_complete", methods=["POST"])
def acknowledge_complete():
    global generation_is_complete
    generation_is_complete.clear()  # Reset the event
    return jsonify({"acknowledged": True})


@midi_app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response


# In midi_app.py
@midi_app.route("/control/play_pause", methods=["POST"])
def play_pause():
    global is_playing

    is_playing = not is_playing

    if is_playing:
        start_event.set()
    else:
        start_event.clear()

    return jsonify({"status": "success", "isPlaying": is_playing})


def start_broadcaster():
    global config_queue, gen_process, generation_queue

    print("---Starting the MIDI broadcaster---")
    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(
        target=lambda: midi_app.run(threaded=True, port=5005)
    )
    flask_thread.daemon = True
    flask_thread.start()

    # Start the broadcasting loop in a separate thread
    broadcasting_thread = threading.Thread(
        target=broadcasting_loop,
        args=(
            generation_queue,
            stop_event,
            start_event,
            change_groove_event,
            [
                is_drum_muted,
                is_bass_muted,
                is_chord_muted,
                is_melody_muted,
                is_harmony_muted,
            ],
        ),
    )
    broadcasting_thread.daemon = True
    broadcasting_thread.start()

    config_queue = mpQueue()
    gen_process = Process(
        target=music_generation_process,
        args=(
            config_queue,
            generation_queue,
            change_groove_event,
            generation_is_complete,
        ),
    )
    gen_process.start()

    print("MIDI broadcaster started")
