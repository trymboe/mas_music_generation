from .utils import (
    tokens_to_note_sequence,
    load_model,
    generate_sequences,
    note_sequence_to_midi_file,
    continue_sequence,
)

from data_processing import Drum_Dataset, load_yaml, get_drum_dataset

from config import DRUM_STYLES, TEMPO, MODEL_PATH_DRUM, DEVICE

import random
import pretty_midi

from .drum_network import Drum_Network

import note_seq as ns


def play_drum(
    measures: int, loops: int, style: str = "highlife"
) -> pretty_midi.PrettyMIDI:
    drum_dataset: Drum_Dataset = get_drum_dataset()
    if style:
        return play_drum_from_style(measures, loops, drum_dataset, style)

    conf = load_yaml("config/bumblebeat/params.yaml")

    time_vocab = load_yaml("config/bumblebeat/time_steps_vocab.yaml")

    model_conf = conf["model"]

    path = MODEL_PATH_DRUM

    model = load_model(path, DEVICE)

    pitch_vocab = drum_dataset.reverse_vocab
    velocity_vocab = {v: k for k, v in drum_dataset.vel_vocab.items()}

    mem_len = model_conf["mem_len"]
    gen_len = 220

    simplified_pitches = [[36], [38], [42], [46], [45], [48], [50], [49], [51]]

    seqs = generate_sequences(
        model,
        num=1,
        gen_len=gen_len,
        mem_len=mem_len,
        device=DEVICE,
        temp=1,
        topk=5,
    )
    for i, s in enumerate(seqs):
        note_sequence = tokens_to_note_sequence(
            s[1:],
            pitch_vocab,
            simplified_pitches,
            velocity_vocab,
            time_vocab,
            120,
        )
        note_sequence = ns.quantize_note_sequence(
            note_sequence, conf["processing"]["steps_per_quarter"]
        )

        pm = ns.note_sequence_to_pretty_midi(note_sequence)
        # pm.write("results/drum/loop_gen.midi")

        note_sequence_to_midi_file(note_sequence, f"results/drum/full_gen.midi")

    return pm


def play_drum_from_style(measures, loops, drum_dataset, style):
    conf: dict = load_yaml("config/bumblebeat/params.yaml")

    time_vocab: dict[int, int] = load_yaml("config/bumblebeat/time_steps_vocab.yaml")

    model: Drum_Network = load_model(MODEL_PATH_DRUM, DEVICE)

    pitch_vocab: dict[int, int] = drum_dataset.reverse_vocab
    velocity_vocab: dict[int, int] = {v: k for k, v in drum_dataset.vel_vocab.items()}

    primer_length: int = 256
    gen_len: int = 128

    simplified_pitches: list[list[int]] = [
        [36],
        [38],
        [42],
        [46],
        [45],
        [48],
        [50],
        [49],
        [51],
    ]

    attempt: int = 0
    while True:
        attempt += 1
        if attempt == 100:
            print(
                f"Could not find a sequence long enough for {style}, default to {get_key(DRUM_STYLES, 7)}"
            )
            style: str = get_key(DRUM_STYLES, 7)
        random_sequence: dict = random.choice(
            [
                x
                for x in drum_dataset.train_data
                if x["style"]["primary"] == DRUM_STYLES[style]
            ]
        )

        dev_sequence: ns.NoteSequence = drum_dataset._quantize(
            ns.midi_to_note_sequence(random_sequence["midi"]), 4
        )

        quantize: bool = True
        in_tokens: list[int] = drum_dataset._tokenize(dev_sequence, 4, quantize)

        if len(in_tokens) >= primer_length:
            break

        if attempt == 100:
            break

    note_sequence = tokens_to_note_sequence(
        in_tokens,
        pitch_vocab,
        simplified_pitches,
        velocity_vocab,
        time_vocab,
        TEMPO,
    )

    out_tokens = continue_sequence(
        model,
        seq=in_tokens[-primer_length:],
        prime_len=primer_length - 1,
        gen_len=gen_len,
        mem_len=gen_len,
        device=DEVICE,
        temp=0.95,
        topk=None,
    )

    out_tokens = out_tokens[gen_len:]

    note_sequence = tokens_to_note_sequence(
        out_tokens,
        pitch_vocab,
        simplified_pitches,
        velocity_vocab,
        time_vocab,
        60,
    )

    note_sequence = ns.quantize_note_sequence(
        note_sequence, conf["processing"]["steps_per_quarter"]
    )

    pm = ns.note_sequence_to_pretty_midi(note_sequence)

    pm = loop_drum(pm, measures, loops)
    # pm.write("results/drum/loop_continue.midi")

    note_sequence_to_midi_file(note_sequence, f"results/drum/full_continue.midi")

    return pm


def loop_drum(
    pm: pretty_midi.PrettyMIDI, measures: int, loops: int
) -> pretty_midi.PrettyMIDI:
    # Assuming 4/4 time signature
    beats_per_measure = 4

    # Get the tempo (assuming a constant tempo for simplicity)
    tempo = 120  # pm.get_tempo_changes()[1][0]

    # Calculate the time duration of the specified measures
    seconds_per_beat = 60.0 / tempo
    clip_duration = seconds_per_beat * beats_per_measure * measures

    # Clip the MIDI object
    clipped_pm = pretty_midi.PrettyMIDI()
    for instrument in pm.instruments:
        clipped_instrument = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum
        )
        for note in instrument.notes:
            if note.start < clip_duration:
                # Adjust note end if it exceeds the clip_duration
                note_end = min(note.end, clip_duration)
                clipped_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note_end,
                    )
                )
        clipped_pm.instruments.append(clipped_instrument)

    # Loop the clipped section
    looped_pm = pretty_midi.PrettyMIDI()
    for instrument in clipped_pm.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum
        )
        for _ in range(loops):
            time_offset = _ * clip_duration
            for note in instrument.notes:
                new_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + time_offset,
                        end=note.end + time_offset,
                    )
                )
        looped_pm.instruments.append(new_instrument)

    return looped_pm


def get_key(my_dict: dict[str, int], value: int) -> str:
    for key, val in my_dict.items():
        if val == value:
            return key
    return "Key not found"
