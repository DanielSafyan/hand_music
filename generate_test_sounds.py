import numpy as np
from array import array
import wave
import os

def generate_sine_wave(frequency, duration=0.5, sample_rate=44100, amplitude=4096):
    """Generate a sine wave at the specified frequency"""
    num_samples = int(duration * sample_rate)
    
    # Generate time points
    t = np.linspace(0, duration, num_samples)
    
    # Generate sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integer values
    sine_wave = sine_wave.astype(np.int16)
    
    return sine_wave

def save_wave_file(filename, samples, sample_rate=44100):
    """Save the samples as a WAV file"""
    with wave.open(filename, 'w') as wave_file:
        n_channels = 1
        sample_width = 2  # 2 bytes for 16-bit audio
        
        wave_file.setnchannels(n_channels)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(sample_rate)
        
        # Convert samples to bytes
        sample_data = array('h', samples)
        wave_file.writeframes(sample_data.tobytes())

def main():
    # Create sounds directory if it doesn't exist
    if not os.path.exists('sounds'):
        os.makedirs('sounds')
    
    # Generate different notes (using simple frequencies)
    frequencies = {
        'note1': 262,  # C4
        'note2': 330,  # E4
        'note3': 392,  # G4
        'note4': 523,  # C5
    }
    
    # Generate and save each note
    for note_name, freq in frequencies.items():
        samples = generate_sine_wave(freq)
        filename = f'sounds/{note_name}.wav'
        save_wave_file(filename, samples)
        print(f"Generated {filename}")

if __name__ == "__main__":
    main() 