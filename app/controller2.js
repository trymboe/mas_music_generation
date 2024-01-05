function togglePlayPause() {
    fetch('http://localhost:5005/control/play_pause', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            updatePlayPauseButton(data.isPlaying);
        })
        .catch(error => console.error('Error:', error));
}

function updatePlayPauseButton(isPlaying) {
    const button = document.getElementById('playPauseButton');
    if (isPlaying) {
        button.innerHTML = '&#10074;&#10074;'; // Pause symbol
    } else {
        button.innerHTML = '&#9658;'; // Play symbol
    }
}
