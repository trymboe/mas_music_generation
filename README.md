# Interactive Music Generation Using Machine Learning and Multi-Agent Systems

## Description
This project is developed as part of a master thesis with the aim of exploring and implementing an interactive music generation system using Multi-Agent Systems and Machine Learning. The aim is to discover the potential that lies in using a such a modular approach for both network performance and musical co-creation. The master's thesis is available [here](Thesis.pdf).

## Dependencies and Setup
Dependencies required to run this project are listed in the requirements.txt file. Install them using:
```bash
pip install -r requirements.txt
```

In addition, you will need a Digital Audio Workstation (DAW) that can handle multiple MIDI-inputs from different ports. This is described further [in the following section](#connecting-a-daw-to-get-sound).


### Models
Training models requires several hours, even on high-performance computers. To train your own models, run the following command with these arguments:
```bash
python main.py
```
With these args:
-tb: train bass model
-tc: train chord model
-td: train drum model
-tm: train melody model

To use pretrained models, download them using curl from the root directory:

### Bass model
```bash 
curl -L -o models/bass/bass_model_lstm.pt 'https://drive.google.com/uc?export=download&id=1z7mMV1z6KoGCzMlxXFPF9ejjAYEAj8Tx'
```

### Chord model
```bash 
curl -L -o models/chord/chord_model_lstm.pt 'https://drive.google.com/uc?export=download&id=1CYmf337orKY2R6gLCF9TW2NL1TFcu6Xq'
```

### Drum model
```bash 
curl -L -o models/drum/drum_model_original.pt 'https://drive.google.com/uc?export=download&id=1HQJVvIPPcuVBttWCIjpif5hNQn1hLKR2'
```

### Melod model
Due to file size, the melody model has to be downloaded directly from [google drive](https://drive.google.com/uc?export=download&id=1mmlUuVGOKdM5y1nBTT1LWi3wl9SVHUmz).
Place it then in models/melody/melody_model_100_2.pt

# Usage
Once the dependencies are installed and the models are either trained or downloaded, run the program:
```bash
python main.py
```
This will automatically open a localy hosted app in your browser. In the app you can tune parameters an create your own music.

## Connecting a DAW to get sound
To play the generated music, you need a MIDI player capable of listening to multiple MIDI ports simultaneously. The music will be broadcast to virtual MIDI channels on your computer and picked up by the MIDI player.

If you use Ableton, you can open the Ableton preset that is included in the repo. If not, you need to conenct the agents to the correct MIDI channel.

The agents and their channels are as follows:

- Channel 1: Drum
- Channel 2: Bass
- Channel 3: Chord
- Channel 4: Melody
- Channel 5: Harmony

Below is an example setup in Ableton Live 11:

![Image of ableton setup](media/ableton_setup.png)




## Acknowledgments
Special thanks to Kyrre Glette and Çağrı Erdem for supervising this master's project.
Thanks to Çağrı Erdem for assistance with the implementation of the MIDI broadcasting loop.
Appreciation to the Bumblebeat project for inspiration and foundational work in drum generation.
[Bumblebeat Project](https://github.com/thomasgnuttall/bumblebeat/tree/master)