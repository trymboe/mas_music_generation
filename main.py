import math

from data_processing.chord_processing import extract_chords_from_files


# Example usage:
root_directory = 'data'
number_of_chords = 0
chords = extract_chords_from_files(root_directory, number_of_chords, True)

print(len(chords))