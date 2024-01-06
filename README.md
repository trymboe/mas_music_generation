# Music Generation with Machine Learning and Multi-Agent Systems

## Description
This project is developed as part of a master thesis with the aim of exploring and implementing innovative solutions in the field of music generation. By using machine learning techniques in collaberation with concepts from Multi-Agent Systems, I strive to create a unique and framework capable of generating aesthetically pleasing musical compositions.

## Status: Work in Progress
Please note that this project is currently a Work in Progress (WIP). This means that not all features are fully implemented, and the project is in active development.

## Dependencies and Setup
Please refer to the `requirements.txt` file for a list of necessary Python libraries and dependencies required to run this project. You can install them using the following command:
```bash
pip install -r requirements.txt
```

### Models
Training your own models will several hours, even on high performance computers.
If you want to train your own models, it can be done by running 
```bash
python main.py
```
With these args:
-tb: train bass model
-tc: train chord model
-td: train drum model
-tm: train melody model

If you want to use pretrained models, they can be downloaded using gdown.
Download from root directory

### Bass model
```bash 
gdown https://drive.google.com/uc?id=1_QI0Ynh-nXKvvJR2tfUM_BXLT_ixuUkf -O models/drum/drum_model.pt
```

### Chord model
```bash 
gdown https://drive.google.com/uc?id=1aeACzuW1D-t0DoFUfZQQXNTX4aAajIRv -O models/chord/chord_model.pt
```

### Drum model
```bash 
gdown https://drive.google.com/uc?id=123eaHn9ab9jWdTKzkpWxlJJ1WUoRCND0 -O models/drum/drum_model.pt
```

### Melod model
```bash 
gdown https://drive.google.com/uc?id=1nUGb2Mbs4Z_ulcG374OVaY2ud_uomIjg -O models/melody/melody_model.pt
```

# Usage
When you have your dependencies installed, and models trained or downloaded, you can run the program like this:
```bash
python main.py
```
This will automatically open a localy hosted app in your browser. In the app you can tune parameters an create your own loops.


## Acknowledgments
Thanks to Kyrre Glette for supervision on my masters project.

Thanks to Çağrı Erdem for the help in implementation of the MIDI broadcasting loop.

Thanks to the Bumblebeat project for inspiring and providing a solid foundation for the drum generation aspect of this project.
[Bumblebeat Project](https://github.com/thomasgnuttall/bumblebeat/tree/master)