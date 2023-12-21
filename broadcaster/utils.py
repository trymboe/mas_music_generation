def get_duration_preferences_bass(duration_slider):
    """
    Returns a list of duration preferences for the bass agent.

    Parameters
    ----------
    duration_slider : int
        The duration slider value from the GUI.

    Returns
    ----------
    duration_preferences : list
        A list of duration preferences for the bass agent.
    """
    if duration_slider >= 88:
        return [1, 2, 3, 4, 5, 6, 7, 8]
    elif duration_slider >= 66:
        return [2, 4, 6, 8]
    elif duration_slider >= 44:
        return [2, 4]
    elif duration_slider >= 22:
        return [4]
    else:
        return [4]


def get_note_temperature_melody(note_slider):
    """
    Calculate the temperature and scale type based on the note slider value.

    Parameters:
    ----------
    note_slider (int): The value of the note slider.

    Returns:
    ----------
    tuple: A tuple containing the temperature (float) and scale type (str).
    """

    if note_slider >= 90:
        return 3, False
    elif note_slider >= 80:
        return 2.5, "major scale"
    elif note_slider >= 70:
        return 2, "major scale"
    elif note_slider >= 60:
        return 1.5, "major pentatonic"
    elif note_slider >= 50:
        return 1, "major pentatonic"
    elif note_slider >= 40:
        return 0.8, "major pentatonic"
    elif note_slider >= 30:
        return 0.5, "major pentatonic"
    elif note_slider >= 20:
        return 0.3, "major pentatonic"
    elif note_slider >= 10:
        return 0.1, "major pentatonic"
    else:
        return 0.01, "major pentatonic"


def get_duration_temperature_melody(duration_slider):
    """
    Calculate the duration temperature and melody based on the given duration slider value.

    Args:
    ----------
        duration_slider (int): The value of the duration slider.

    Returns:
    ----------
        tuple: A tuple containing the duration temperature and the melody.
               The duration temperature is a float value.
               The melody is a list of integers.

    """
    if duration_slider >= 90:
        return 3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif duration_slider >= 80:
        return 2.5, [0, 1, 3, 5, 7, 9, 11, 13, 15]
    elif duration_slider >= 70:
        return 2, [1, 3, 5, 7, 9, 11, 13, 15]
    elif duration_slider >= 60:
        return 1.5, [1, 3, 5, 7, 9, 11, 13, 15]
    elif duration_slider >= 50:
        return 1, [1, 3, 5, 7, 9, 11, 13, 15]
    elif duration_slider >= 40:
        return 0.8, [1, 3, 5, 7, 9, 11, 13, 15]
    elif duration_slider >= 30:
        return 0.5, [1, 3, 7, 9, 13, 15]
    elif duration_slider >= 20:
        return 0.3, [3, 7, 15]
    elif duration_slider >= 10:
        return 0.1, [3, 7, 15]
    else:
        return 0.01, [7]


def get_duration_preferences_bass_from_advanced(c1, c2, c3, c4, c5, c6, c7, c8):
    """
    Returns a list of duration preferences for the bass agent.

    Parameters
    ----------
    c1 : bool
        The value indicating whether the duration 1 is preferred.
    c2 : bool
        The value indicating whether the duration 2 is preferred.
    c3 : bool
        The value indicating whether the duration 3 is preferred.
    c4 : bool
        The value indicating whether the duration 4 is preferred.
    c5 : bool
        The value indicating whether the duration 5 is preferred.
    c6 : bool
        The value indicating whether the duration 6 is preferred.
    c7 : bool
        The value indicating whether the duration 7 is preferred.
    c8 : bool
        The value indicating whether the duration 8 is preferred.

    Returns
    ----------
    duration_preferences : list
        A list of duration preferences for the bass agent.
    """
    duration_preferences = []
    if c1:
        duration_preferences.append(1)
    if c2:
        duration_preferences.append(2)
    if c3:
        duration_preferences.append(3)
    if c4:
        duration_preferences.append(4)
    if c5:
        duration_preferences.append(5)
    if c6:
        duration_preferences.append(6)
    if c7:
        duration_preferences.append(7)
    if c8:
        duration_preferences.append(8)
    return duration_preferences


def get_duration_preferences_melody_from_advanced(
    note16, note8, note4, note2, note1, note_double
):
    """
    Returns a list of duration preferences for the melody agent.

    Parameters
    ----------
    note16 : bool
        The value of the note16 checkbox.
    note8 : bool
        The value of the note8 checkbox.
    note4 : bool
        The value of the note4 checkbox.
    note2 : bool
        The value of the note2 checkbox.
    note1 : bool
        The value of the note1 checkbox.
    note_double : bool
        The value of the note_double checkbox.

    Returns
    ----------
    duration_preferences : list
        A list of duration preferences for the melody agent.
    """
    duration_preferences = []
    if note16:
        duration_preferences.append(0)
    if note8:
        duration_preferences.append(1, 9)
    if note4:
        duration_preferences.append(3, 11)
    if note2:
        duration_preferences.append(5, 13)
    if note1:
        duration_preferences.append(7)
    if note_double:
        duration_preferences.append(15)
    return duration_preferences
