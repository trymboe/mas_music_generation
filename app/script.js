let isGeneratingMusic = false;

document.getElementById("control-form").onsubmit = function (event) {
    event.preventDefault();
    var submitButton = document.getElementById('submit-button');
    var statusMessage = document.getElementById("status-message");

    submitButton.disabled = true;
    isGeneratingMusic = true;
    statusMessage.innerHTML = "Generating Music...";

    var formData = {
        tempo: document.getElementById("tempo").value,
        length: document.getElementById("length").value,
        play_drum: document.getElementById("play_drum").checked,
        loop_measures: document.getElementById("loop_measures").value,
        style: document.getElementById("style").value,
        play_bass: document.getElementById("play_bass").checked,
        duration_preferences_bass: document.getElementById("duration_preferences_bass").checked,
        playstyle: document.getElementById("playstyle").value,
        play_chord: document.getElementById("play_chord").checked,
        arpegiate_chord: document.getElementById("arpegiate_chord").checked,
        bounce_chord: document.getElementById("bounce_chord").checked,
        arp_style: document.getElementById("arp_style").value,
        play_melody: document.getElementById("play_melody").checked,
        note_temperature_melody: document.getElementById("note_temperature_melody").value,
        duration_temperature_melody: document.getElementById("duration_temperature_melody").value,
        no_pause: document.getElementById("no_pause").checked,
        scale_melody: document.getElementById("scale_melody").value,
        duration_preferences_melody: document.getElementById("duration_preferences_melody").value.split(',').map(Number),
        play_harmony: document.getElementById("play_harmony").checked,
        interval_harmony: document.getElementById("interval_harmony").value
        // Add more parameters here
    };

    fetch('http://localhost:5005/set_params', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            pollForCompletion();
        })
        .catch((error) => {
            console.error('Error:', error);
            submitButton.disabled = false;
            statusMessage.textContent = "";
        });

    function pollForCompletion() {
        if (!isGeneratingMusic) {
            return; // Exit if no music generation is in progress
        }

        fetch('http://localhost:5005/check_status')
            .then(response => response.json())
            .then(data => {
                if (data.isComplete) {
                    document.getElementById("submit-button").disabled = false;
                    document.getElementById("status-message").textContent = "";
                    isGeneratingMusic = false;
                    acknowledgeGenerationComplete(); // Acknowledge completion
                } else {
                    setTimeout(pollForCompletion, 500); // Continue polling
                }
            })
            .catch(error => {
                console.error('Error:', error);
                isGeneratingMusic = false;
            });
    }

    function acknowledgeGenerationComplete() {
        fetch('http://localhost:5005/acknowledge_complete', { method: 'POST' })
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error('Error:', error));
    }
};