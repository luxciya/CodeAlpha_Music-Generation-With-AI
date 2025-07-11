import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import load_model
import pickle

# Load data
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(item for item in notes))
n_vocab = len(pitchnames)
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

sequence_length = 100
network_input = []
for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])
n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)

model = load_model("model.h5")

# Generate music
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start].tolist()
prediction_output = []

for note_index in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern.append([index / float(n_vocab)])
    pattern = pattern[1:]

# Convert to MIDI
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_list = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes_list.append(new_note)
        new_chord = chord.Chord(notes_list)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_output.mid')
print("✅ MIDI file 'generated_output.mid' created successfully!")
