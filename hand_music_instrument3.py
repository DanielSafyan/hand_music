import cv2
import mediapipe as mp
import pygame
import numpy as np
from math import hypot, sin, pi
import pygame.sndarray
import wave
import os
from pydub import AudioSegment
import io

class HandMusicInstrument3:
    def __init__(self, audio_file=None):
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                       max_num_hands=2,
                                       min_detection_confidence=0.7,
                                       min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize Pygame for audio
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        pygame.init()

        # Audio parameters
        self.sample_rate = 44100
        self.duration = 0.05
        self.max_amplitude = 4096
        
        # Sound parameters
        self.base_freq = 440  # Base frequency (A4)
        self.min_pitch_factor = 0.5   # Start from half pitch
        self.max_pitch_factor = 8.0   # Go up to 8x pitch
        self.min_rate = 0.5   # Minimum modulation rate
        self.max_rate = 8.0   # Maximum modulation rate
        
        # Window parameters for better pitch shifting
        self.window_size = 2048  # Size of the processing window
        self.hop_length = 512    # Number of samples between windows
        self.window = np.hanning(self.window_size)  # Hanning window for smooth transitions
        
        # Control sensitivity
        self.speed_sensitivity = 0.25  # Less sensitive speed control
        self.pitch_sensitivity = 0.5   # Much less sensitive pitch control (was 2.0)
        
        # Volume sensitivity factor (1.5 times more sensitive)
        self.volume_sensitivity = 1.5
        
        # Audio file handling
        self.audio_file = audio_file
        self.audio_data = None
        if audio_file and os.path.exists(audio_file):
            self.load_audio_file(audio_file)
        
        # Initialize camera
        self.cap = None
        for device_id in range(2):  # Try first two video devices
            print(f"Trying to open camera device {device_id}...")
            self.cap = cv2.VideoCapture(device_id)
            if self.cap is not None and self.cap.isOpened():
                _, frame = self.cap.read()
                if frame is not None:
                    self.frame_height, self.frame_width = frame.shape[:2]
                    print(f"Successfully opened camera device {device_id}")
                    break
                else:
                    self.cap.release()
        
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Could not initialize camera. Please check if your webcam is connected and accessible.")
        
        # Create sound channel
        pygame.mixer.set_num_channels(1)
        self.sound_channel = pygame.mixer.Channel(0)
        
        # Landmark indices
        self.thumb_tip = 4
        self.index_tip = 8
        
        # Add default playback parameters
        self.default_rate = 1.0
        self.default_pitch = 1.0
        self.default_volume = 0.5  # Reduced default volume
        
        # Track continuous playback
        self.is_playing = False
        self.current_phase = 0.0
        self.phase_increment = 0.0
        
        # Add playback state tracking
        self.last_rate = 1.0
        self.rate_smoothing = 0.95  # Smooth rate changes
        
        # Add waveform animation parameters
        self.wave_time = 0
        self.wave_speed = 10
        self.wave_segments = 50  # Number of segments in the wave
        self.base_wave_amplitude = 20  # Base amplitude in pixels

    def load_audio_file(self, audio_file):
        """Load and prepare audio file data (MP3 or WAV)"""
        # Determine file type from extension
        file_ext = os.path.splitext(audio_file)[1].lower()
        
        if file_ext == '.mp3':
            # Load MP3 using pydub
            audio = AudioSegment.from_mp3(audio_file)
            
            # Convert to WAV format in memory
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            
            # Load the WAV data from memory
            with wave.open(wav_io) as wav:
                self._process_wav_data(wav)
        
        elif file_ext == '.wav':
            # Load WAV file directly
            with wave.open(audio_file, 'rb') as wav:
                self._process_wav_data(wav)
        
        else:
            raise ValueError(f"Unsupported audio format: {file_ext}")

    def _process_wav_data(self, wav):
        """Process WAV data from either WAV file or converted MP3"""
        # Read WAV parameters
        self.wav_channels = wav.getnchannels()
        self.wav_width = wav.getsampwidth()
        self.wav_rate = wav.getframerate()
        
        # Read the entire file
        wav_data = wav.readframes(wav.getnframes())
        
        # Convert to numpy array
        self.audio_data = np.frombuffer(wav_data, dtype=np.int16)
        
        # If stereo, convert to mono by averaging channels
        if self.wav_channels == 2:
            self.audio_data = self.audio_data.reshape(-1, 2).mean(axis=1)
        
        # Normalize to our sample rate if needed
        if self.wav_rate != self.sample_rate:
            samples_needed = int(len(self.audio_data) * self.sample_rate / self.wav_rate)
            self.audio_data = np.interp(
                np.linspace(0, len(self.audio_data), samples_needed),
                np.arange(len(self.audio_data)),
                self.audio_data
            )

    def get_line_midpoint(self, hand_landmarks):
        """Calculate midpoint of thumb-index line"""
        thumb = hand_landmarks.landmark[self.thumb_tip]
        index = hand_landmarks.landmark[self.index_tip]
        mid_x = (thumb.x + index.x) / 2
        mid_y = (thumb.y + index.y) / 2
        return (mid_x, mid_y)

    def get_hand_distance(self, hand1_landmarks, hand2_landmarks):
        """Calculate distance between the midpoints of thumb-index lines"""
        # Get midpoints of thumb-index lines
        mid1 = self.get_line_midpoint(hand1_landmarks)
        mid2 = self.get_line_midpoint(hand2_landmarks)
        
        return hypot(mid1[0] - mid2[0], mid1[1] - mid2[1])

    def get_finger_distance(self, hand_landmarks):
        """Calculate distance between thumb and index finger"""
        thumb = hand_landmarks.landmark[self.thumb_tip]
        index = hand_landmarks.landmark[self.index_tip]
        return hypot(thumb.x - index.x, thumb.y - index.y)

    def pitch_shift_frame(self, frame, pitch_factor):
        """Apply pitch shifting to a single frame while preserving formants"""
        # Apply window
        windowed = frame * self.window
        
        # Get frequency spectrum
        spectrum = np.fft.rfft(windowed)
        
        # Calculate new length after pitch shift
        new_length = int(len(spectrum) * pitch_factor)
        
        # Preserve formants by scaling the frequency spectrum
        if pitch_factor > 1.0:
            # Stretching spectrum (higher pitch)
            new_spectrum = np.zeros(new_length, dtype=complex)
            new_spectrum[:len(spectrum)] = spectrum
        else:
            # Compressing spectrum (lower pitch)
            new_spectrum = spectrum[:new_length]
        
        # Adjust magnitude to preserve energy
        magnitude_factor = np.sqrt(1.0 / pitch_factor)
        new_spectrum *= magnitude_factor
        
        # Convert back to time domain
        shifted = np.fft.irfft(new_spectrum)
        
        # Resize to original size
        if len(shifted) > len(frame):
            shifted = shifted[:len(frame)]
        else:
            shifted = np.pad(shifted, (0, len(frame) - len(shifted)))
        
        return shifted

    def generate_modulated_wave(self, pitch_factor, rate, amplitude):
        """Generate sound based on audio file if available, otherwise use sine wave"""
        if self.audio_data is not None:
            # Calculate how many samples we need for this duration
            num_samples = int(self.duration * self.sample_rate)
            
            # Calculate the position in the audio file based on current phase
            start_pos = int(self.current_phase * len(self.audio_data))
            
            # Get samples from audio file with wraparound
            samples = np.zeros(num_samples)
            for i in range(num_samples):
                pos = (start_pos + i) % len(self.audio_data)
                samples[i] = self.audio_data[pos]
            
            # Update phase for continuous playback (corrected calculation)
            samples_per_second = self.sample_rate * rate
            self.phase_increment = (samples_per_second * self.duration) / len(self.audio_data)
            self.current_phase = (self.current_phase + self.phase_increment) % 1.0
            
            # Only apply pitch shifting if pitch_factor is significantly different from 1.0
            if abs(pitch_factor - 1.0) > 0.01:
                # Process audio in overlapping windows
                processed = np.zeros_like(samples)
                for i in range(0, len(samples) - self.window_size, self.hop_length):
                    # Get current window
                    window = samples[i:i + self.window_size]
                    if len(window) < self.window_size:
                        break
                    
                    # Apply pitch shifting
                    shifted = self.pitch_shift_frame(window, pitch_factor)
                    
                    # Overlap-add
                    processed[i:i + self.window_size] += shifted
                
                # Normalize
                if np.max(np.abs(processed)) > 0:
                    processed = processed * (np.max(np.abs(samples)) / np.max(np.abs(processed)))
                
                samples = processed
            
            # Apply amplitude
            samples = samples * (amplitude / self.max_amplitude)
            
            return samples.astype(np.int16)
        else:
            # Original sine wave generation with relative pitch
            t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
            frequency = self.base_freq * pitch_factor  # Apply pitch factor to base frequency
            wave = amplitude * np.sin(2 * np.pi * frequency * t + self.current_phase)
            self.current_phase += 2 * np.pi * frequency * self.duration * rate
            self.current_phase %= 2 * np.pi
            wave = wave * (32767 / max(abs(wave)))
            return wave.astype(np.int16)

    def draw_parameter_info(self, image, frequency, rate, volume):
        """Draw current parameter values on screen"""
        # Draw frequency info
        cv2.putText(image, f"Frequency: {int(frequency)}Hz",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw rate info
        cv2.putText(image, f"Rate: {rate:.1f}x",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw volume info
        cv2.putText(image, f"Volume: {int(volume * 100)}%",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_hand_connections(self, image, hand_landmarks, is_left, value_text):
        """Draw hand landmarks with labeled thumb-index distance"""
        # Draw hand skeleton in white
        self.mp_draw.draw_landmarks(
            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
        )
        
        # Draw thumb-index connection with white line
        thumb = hand_landmarks.landmark[self.thumb_tip]
        index = hand_landmarks.landmark[self.index_tip]
        
        thumb_pos = (int(thumb.x * self.frame_width), 
                    int(thumb.y * self.frame_height))
        index_pos = (int(index.x * self.frame_width), 
                    int(index.y * self.frame_height))
        
        # Draw white line
        cv2.line(image, thumb_pos, index_pos, (255, 255, 255), 2)
        
        # Calculate midpoint for label
        mid_x = (thumb_pos[0] + index_pos[0]) // 2
        mid_y = (thumb_pos[1] + index_pos[1]) // 2
        
        # Add label
        cv2.putText(image, value_text,
                   (mid_x - 40, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_animated_waveform(self, image, start_pos, end_pos, volume):
        """Draw an animated vertical bar waveform between two points with amplitude based on volume"""
        # Update wave time with slower speed
        self.wave_time += 0.03
        
        # Calculate the direction vector between points
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return
        
        # Normalize direction vector
        dx /= distance
        dy /= distance
        
        # Calculate perpendicular vector for wave amplitude
        perpx = -dy
        perpy = dx
        
        # Increase the base amplitude significantly
        self.base_wave_amplitude = 80
        
        # Number of bars and width - increased bars, reduced width
        num_bars = 30  # Doubled from 15
        bar_width = 2  # Halved from 4
        
        # Dark orange color in BGR format (OpenCV uses BGR)
        bar_color = (0, 127, 255)  # This creates a dark orange color
        
        # Generate and draw vertical bars
        for i in range(num_bars):
            # Position along the line (0 to 1)
            t = i / (num_bars - 1)
            
            # Base position for the bar center
            center_x = int(start_pos[0] + dx * distance * t)
            center_y = int(start_pos[1] + dy * distance * t)
            
            # Create irregular wave pattern by combining multiple waves
            wave1 = np.sin(t * 4 * np.pi + self.wave_time * 3)  # Base wave
            wave2 = 0.5 * np.sin(t * 8 * np.pi + self.wave_time * 5)  # Higher frequency
            wave3 = 0.3 * np.sin(t * 2 * np.pi + self.wave_time * 2)  # Lower frequency
            noise = 0.2 * np.sin(t * 16 * np.pi + self.wave_time * 7)  # High-frequency noise
            
            # Combine waves and add some randomness
            wave = abs(wave1 + wave2 + wave3 + noise + 0.1 * np.random.random())
            # Normalize the wave to keep it in reasonable bounds
            wave = wave / 2.5  # Adjust divisor to control overall amplitude
            
            amplitude = self.base_wave_amplitude * volume * wave
            
            # Calculate bar endpoints
            bar_top = (
                int(center_x + perpx * amplitude),
                int(center_y + perpy * amplitude)
            )
            bar_bottom = (
                int(center_x - perpx * amplitude),
                int(center_y - perpy * amplitude)
            )
            
            # Draw the vertical bar with dark orange color
            cv2.line(image, bar_top, bar_bottom, bar_color, bar_width)

    def draw_hand_distance(self, image, left_hand, right_hand, volume):
        """Draw animated waveform between thumb-index midpoints with volume label"""
        # Get midpoints of thumb-index lines
        left_mid = self.get_line_midpoint(left_hand)
        right_mid = self.get_line_midpoint(right_hand)
        
        # Convert to screen coordinates
        left_pos = (int(left_mid[0] * self.frame_width),
                   int(left_mid[1] * self.frame_height))
        right_pos = (int(right_mid[0] * self.frame_width),
                    int(right_mid[1] * self.frame_height))
        
        # Draw animated waveform
        self.draw_animated_waveform(image, left_pos, right_pos, volume)
        
        # Calculate midpoint for volume label
        mid_x = (left_pos[0] + right_pos[0]) // 2
        mid_y = (left_pos[1] + right_pos[1]) // 2
        
        # Add volume label
        cv2.putText(image, f"Vol: {int(volume * 100)}%",
                   (mid_x - 40, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def update_audio(self, pitch_factor, rate, volume):
        """Update audio playback with current parameters"""
        # Smooth the rate changes to prevent sudden jumps
        self.last_rate = (self.last_rate * self.rate_smoothing + 
                         rate * (1 - self.rate_smoothing))
        
        wave = self.generate_modulated_wave(pitch_factor, self.last_rate, self.max_amplitude)
        sound = pygame.sndarray.make_sound(wave)
        self.sound_channel.play(sound)
        self.sound_channel.set_volume(volume)
        self.is_playing = True

    def run(self):
        print("\nHand Music Instrument 3 Instructions:")
        print("-------------------------------------")
        print("Control sound with both hands:")
        print("1. Left Hand: Thumb-Index distance controls sound rate")
        print("   - Close = slow rate (0.5x)")
        print("   - Far = fast rate (8.0x)")
        print("2. Right Hand: Thumb-Index distance controls pitch")
        print("   - Close = lower pitch (0.5x)")
        print("   - Far = higher pitch (8.0x)")
        print("3. Distance between hands controls volume")
        print("   - Very close = silent")
        print("   - Far = full volume")
        print("\nNo hands visible = normal playback")
        print("Press 'q' to quit")
        print("-------------------------------------\n")
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Failed to capture image from camera.")
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                hands = results.multi_hand_landmarks
                
                # Determine which hand is left/right based on x position
                left_hand = min(hands, key=lambda h: h.landmark[0].x)
                right_hand = max(hands, key=lambda h: h.landmark[0].x)
                
                # Get control parameters
                left_dist = self.get_finger_distance(left_hand)
                right_dist = self.get_finger_distance(right_hand)
                hand_dist = self.get_hand_distance(left_hand, right_hand)
                
                # Apply sensitivity to the distances
                left_dist = np.clip(left_dist * self.speed_sensitivity, 0.0, 1.0)
                right_dist = np.clip(right_dist * self.pitch_sensitivity, 0.0, 1.0)
                
                # Map distances to sound parameters
                rate = np.interp(left_dist, [0.02, 0.15],  # Less sensitive speed
                               [self.min_rate, self.max_rate])
                
                # Less sensitive pitch control with wider range
                pitch_factor = np.interp(right_dist, [0.02, 0.7],  # Wider range for less sensitivity
                                      [self.min_pitch_factor, self.max_pitch_factor])
                
                # Round pitch_factor to 1.0 if it's very close to avoid unnecessary processing
                if abs(pitch_factor - 1.0) < 0.01:
                    pitch_factor = 1.0
                
                volume = np.clip(np.interp(hand_dist * self.volume_sensitivity, 
                                         [0.1, 0.8], [0, 1]), 0, 1)
                
                # Draw hands with labels
                self.draw_hand_connections(image, left_hand, True, f"Speed: {rate:.1f}x")
                self.draw_hand_connections(image, right_hand, False, f"Pitch: {pitch_factor:.1f}x")
                self.draw_hand_distance(image, left_hand, right_hand, volume)
                
                # Update audio with current parameters
                self.update_audio(pitch_factor, rate, volume)
                
                # Draw parameter information
                self.draw_parameter_info(image, self.base_freq * pitch_factor, rate, volume)
            else:
                # When no hands are visible or not enough hands, play at default rate
                if results.multi_hand_landmarks:
                    # Draw available hands
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Update audio with default parameters
                self.update_audio(self.default_pitch, self.default_rate, self.default_volume)
                
                # Draw default parameter information
                self.draw_parameter_info(image, self.base_freq * self.default_pitch,
                                      self.default_rate, self.default_volume)

            cv2.imshow('Hand Music Instrument 3', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.sound_channel.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use the audio file specified as command line argument
        audio_file = sys.argv[1]
        print(f"Using audio file: {audio_file}")
        instrument = HandMusicInstrument3(audio_file)
    else:
        # Use sine wave synthesis (default)
        print("No audio file specified, using sine wave synthesis")
        instrument = HandMusicInstrument3()
    
    instrument.run() 