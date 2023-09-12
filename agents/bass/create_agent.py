from .bass_network import Bass_Network

from config import VOCAB_SIZE_BASS, EMBED_SIZE_BASS, NHEAD_BASS, NUM_LAYERS_BASS


def create_bass_agent():
    bass_agent = Bass_Network(
        VOCAB_SIZE_BASS, EMBED_SIZE_BASS, NHEAD_BASS, NUM_LAYERS_BASS
    )
    return bass_agent
