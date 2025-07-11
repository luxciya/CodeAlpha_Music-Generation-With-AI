# CodeAlpha_Music-Generation-With-AI
This project demonstrates how to generate music using deep learning. It uses MIDI files as input, processes them into note sequences, and trains a Long Short-Term Memory (LSTM) neural network to generate new music compositions. The generated music is saved as MIDI files.
1.🚀 Features
- 🎼 Load and parse MIDI files using `music21`
- 🔁 Preprocess notes and chords into sequences
- 🧠 Build and train an LSTM-based model with Keras/TensorFlow
- 🎹 Generate new music from the trained model
- 💾 Save output as MIDI files for playback
  
2. Create a virtual environment (optional)
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux

3. Install dependencies
pip install -r requirements.txt

5. Add training data
Create a midi/ folder if it doesn’t exist.

Add some .mid files (MIDI format) for training.

5. Run the script
python music_gen_lstm.py
The model will:
Load MIDI files
Train an LSTM on note sequences
Generate new music
🧠 Model Used
LSTM (Long Short-Term Memory)
Sequential model from Keras
Input shape: (100, 1) sequences of notes
Output: Probability distribution over possible next notes

📦 Dependencies
music21
tensorflow or keras
numpy

Install using:
pip install -r requirements.txt

📌 To Do
 Improve output MIDI file generation
 Add GAN-based variant
 Add GUI to play or visualize MIDI notes

