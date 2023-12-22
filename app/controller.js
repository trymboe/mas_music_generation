function toggleMute(instrumentId) {
    var button = document.getElementById(instrumentId);
    var instrument = instrumentId.split('-')[1]; // Extracts the instrument name from the id
    var shouldBeMuted = !button.classList.contains('muted');

    // Immediately change the button's color to reflect the attempted action
    button.style.backgroundColor = shouldBeMuted ? '#CD5C08' : '#eae8e0';

    // Fetch request to toggle the mute state
    fetch('http://localhost:5005/mute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            instrument: instrument, // Pass the instrument name
            mute: shouldBeMuted, // Pass the desired mute state
        }),
    })
        .then(response => {
            if (response.ok) {
                // Toggle the 'muted' class if the operation was successful
                button.classList.toggle('muted');
                // No need to change the color again; it's already been set
            } else {
                // If the operation was not successful, revert the button color
                button.style.backgroundColor = shouldBeMuted ? '#eae8e0' : '#CD5C08';
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            // In case of an error, revert the button color
            button.style.backgroundColor = shouldBeMuted ? '#eae8e0' : '#CD5C08';
        });
}


document.querySelectorAll('.volume-container input[type=range]').forEach(slider => {
    slider.addEventListener('input', function () {
        updateVolume(this.name, this.value);
    });
});

function updateVolume(instrument, volume) {
    console.log('Updating volume for ' + instrument + ' to ' + volume);

    fetch('http://localhost:5005/update_volume', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            instrument: instrument,
            volume: volume
        }),
    })
        .then(response => {
            if (!response.ok) {
                console.error('Failed to update volume');
            }
        })
        .catch(error => console.error('Error:', error));
}
