import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical
import glob

def get_notes():
    notes = []
    for file in glob.glob("midi/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts: notes_to_parse = parts.parts[0].recurse()
        else: notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    sequence_length = 100
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)
    return network_input, network_output

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

notes = get_notes()
n_vocab = len(set(notes))
network_input, network_output = prepare_sequences(notes, n_vocab)
model = create_network(network_input, n_vocab)
model.fit(network_input, network_output, epochs=5, batch_size=64)

import pickle
model.save("model.h5")
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)
