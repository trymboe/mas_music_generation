let isGeneratingMusic = false;

document.getElementById("control-form").onsubmit = function (event) {
    event.preventDefault();
    var submitButton = document.getElementById('submit-button');
    var statusMessage = document.getElementById("status-message");

    submitButton.disabled = true;
    isGeneratingMusic = true;
    statusMessage.innerHTML = "Generating Music...";

    var formData = {
        advancedSettings: document.getElementById("advanced-option").checked,

        // General parameters
        tempo: document.getElementById("tempo").value,
        length: document.getElementById("length").value,

        // Drum parameters
        play_drum: document.getElementById("play_drum").checked,
        style: document.getElementById("style").value,
        loop_measures: document.getElementById("loop_measures").value,

        // Bass parameters
        play_bass: document.getElementById("play_bass").checked,
        bass_creativity: document.getElementById("bass_creativity").value,
        checkbox1: document.getElementById("checkbox1").checked,
        checkbox2: document.getElementById("checkbox2").checked,
        checkbox3: document.getElementById("checkbox3").checked,
        checkbox4: document.getElementById("checkbox4").checked,
        checkbox5: document.getElementById("checkbox5").checked,
        checkbox6: document.getElementById("checkbox6").checked,
        checkbox7: document.getElementById("checkbox7").checked,
        checkbox8: document.getElementById("checkbox8").checked,

        // Chord parameters
        play_chord: document.getElementById("play_chord").checked,
        arpegiate_chord: document.getElementById("arpegiate_chord").checked,
        bounce_chord: document.getElementById("bounce_chord").checked,
        arp_style: document.getElementById("arp_style").value,

        // Melody parameters
        play_melody: document.getElementById("play_melody").checked,
        pitch_creativity_melody: document.getElementById("pitch_creativity_melody").value,
        duration_creativity_melody: document.getElementById("duration_creativity_melody").value,
        note_temperature_melody: document.getElementById("note_temperature_melody").value,
        duration_temperature_melody: document.getElementById("duration_temperature_melody").value,
        no_pause: document.getElementById("no_pause").checked,
        scale_melody: document.getElementById("scale_melody").value,
        checkbox1_melody: document.getElementById("16th_note_melody").checked,
        checkbox2_melody: document.getElementById("8th_note_melody").checked,
        checkbox3_melody: document.getElementById("4th_note_melody").checked,
        checkbox4_melody: document.getElementById("half_note_melody").checked,
        checkbox5_melody: document.getElementById("whole_note_melody").checked,
        checkbox6_melody: document.getElementById("double_whole_note_melody").checked,

        // Harmony parameters
        play_harmony: document.getElementById("play_harmony").checked,
        interval_harmony: document.getElementById("interval_harmony").value
    };


    // Add more parameters here if needed


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


document.getElementById('advanced-option').addEventListener('change', function () {
    var advancedSettings = document.querySelectorAll('.advanced-settings');
    var normalSettings = document.querySelectorAll('.normal-settings');

    if (this.checked) {
        advancedSettings.forEach(function (section) {
            section.style.display = 'block'; // Show advanced settings
        });
        normalSettings.forEach(function (section) {
            section.style.display = 'none'; // Hide normal settings
        });
    } else {
        advancedSettings.forEach(function (section) {
            section.style.display = 'none'; // Hide advanced settings
        });
        normalSettings.forEach(function (section) {
            section.style.display = 'block'; // Show normal settings
        });
    }
});

// Disable or enable arpegiate chord based on bounce chord checkbox
window.onload = function () {
    var arpegiateChordCheckbox = document.getElementById('arpegiate_chord');
    var bounceChordCheckbox = document.getElementById('bounce_chord');
    var arpStyleSelect = document.getElementById('arp_style');

    // Disable or enable bounce chord based on arpegiate chord checkbox
    arpegiateChordCheckbox.onclick = function () {
        bounceChordCheckbox.checked = false;
        bounceChordCheckbox.disabled = this.checked;
        arpStyleSelect.disabled = !this.checked;
    };

    // Disable or enable arpegiate chord based on bounce chord checkbox
    bounceChordCheckbox.onclick = function () {
        arpegiateChordCheckbox.checked = false;
        arpegiateChordCheckbox.disabled = this.checked;
        arpStyleSelect.disabled = true; // Always disable arp_style if bounce is selected
    };
};

