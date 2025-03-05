import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
from math import hypot, sin, pi
import pygame.sndarray

class HandMusicInstrument:
    def __init__(self):
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                       max_num_hands=1,
                                       min_detection_confidence=0.7,
                                       min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Custom drawing specs for better visualization
        self.landmark_style = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
        self.connection_style = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)

        # Initialize Pygame for audio
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        pygame.init()

        # Audio parameters
        self.sample_rate = 44100
        self.duration = 0.05  # Short duration for responsive sound
        self.max_amplitude = 4096
        
        # Base frequencies for each finger (C4, E4, G4, C5)
        self.base_frequencies = {
            'finger_1': 262,  # C4
            'finger_2': 330,  # E4
            'finger_3': 392,  # G4
            'finger_4': 523   # C5
        }
        
        # Create sound channels for each finger
        pygame.mixer.set_num_channels(8)
        self.sound_channels = {
            'finger_1': pygame.mixer.Channel(0),
            'finger_2': pygame.mixer.Channel(1),
            'finger_3': pygame.mixer.Channel(2),
            'finger_4': pygame.mixer.Channel(3)
        }

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Gesture thresholds
        self.min_distance = 0.08
        self.max_distance = 0.3
        
        # Finger names for display
        self.finger_names = {
            'finger_1': 'Index (C4)',
            'finger_2': 'Middle (E4)',
            'finger_3': 'Ring (G4)',
            'finger_4': 'Pinky (C5)'
        }

        # Finger tip indices
        self.finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Wrist landmark index
        self.wrist = 0

    def get_hand_height(self, hand_landmarks):
        """Get normalized hand height (y-coordinate of wrist)"""
        return hand_landmarks.landmark[self.wrist].y

    def generate_wave(self, frequency, amplitude, pitch_multiplier):
        """Generate a wave with pitch modification based on hand height"""
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        # Modify frequency based on hand height
        modified_freq = frequency * pitch_multiplier
        wave = amplitude * np.sin(2 * np.pi * modified_freq * t)
        
        # Normalize and convert to int16
        wave = wave * (32767 / max(abs(wave)))
        return wave.astype(np.int16)

    def update_sounds(self, distances, hand_height):
        """Update sounds based on finger distances and hand height"""
        # Convert hand height (0-1) to pitch multiplier (0.5-2.0)
        # When hand is high (y close to 0), pitch is higher
        # When hand is low (y close to 1), pitch is lower
        pitch_multiplier = 2.0 - hand_height * 1.5  # This gives range of ~0.5-2.0

        for finger, distance in distances.items():
            if distance < self.max_distance:
                # Normalize distance and invert it (closer = louder)
                normalized_dist = min(distance / self.max_distance, 1.0)
                amplitude = int(self.max_amplitude * (1 - normalized_dist))
                
                # Generate wave with pitch modification
                wave = self.generate_wave(self.base_frequencies[finger], amplitude, pitch_multiplier)
                sound = pygame.sndarray.make_sound(wave)
                
                # Play sound on corresponding channel
                channel = self.sound_channels[finger]
                channel.play(sound, -1)
                channel.set_volume(1 - normalized_dist)
            else:
                self.sound_channels[finger].stop()

    def draw_hand_height(self, image, hand_height):
        """Draw hand height information on screen"""
        cv2.putText(image, f"Hand Height: {hand_height:.3f}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)

    def calculate_finger_distances(self, hand_landmarks):
        """Calculate distances between thumb tip and other finger tips"""
        thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
        distances = {}
        
        # Calculate distance to each finger tip
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        for idx, tip in enumerate(finger_tips):
            finger_pos = (hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y)
            distance = hypot(thumb_tip[0] - finger_pos[0], thumb_tip[1] - finger_pos[1])
            distances[f'finger_{idx+1}'] = distance
            
        return distances

    def draw_distance_lines(self, image, hand_landmarks, distances):
        """Draw lines between thumb and fingers with distance labels"""
        h, w, _ = image.shape
        thumb_point = hand_landmarks.landmark[self.finger_tips['thumb']]
        thumb_pos = (int(thumb_point.x * w), int(thumb_point.y * h))
        
        for idx, (finger_name, tip_idx) in enumerate([
            ('index', self.finger_tips['index']),
            ('middle', self.finger_tips['middle']),
            ('ring', self.finger_tips['ring']),
            ('pinky', self.finger_tips['pinky'])
        ]):
            finger_point = hand_landmarks.landmark[tip_idx]
            finger_pos = (int(finger_point.x * w), int(finger_point.y * h))
            distance = distances[f'finger_{idx+1}']
            
            # Determine line color based on distance
            color = (0, 255, 0) if distance < self.min_distance else (0, 0, 255)
            
            # Draw line between thumb and finger
            cv2.line(image, thumb_pos, finger_pos, color, 2)
            
            # Calculate midpoint for text placement
            mid_x = (thumb_pos[0] + finger_pos[0]) // 2
            mid_y = (thumb_pos[1] + finger_pos[1]) // 2
            
            # Draw distance value
            cv2.putText(image, f'{distance:.2f}', 
                       (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)

    def run(self):
        print("\nHand Music Instrument Instructions:")
        print("-----------------------------------")
        print("Control sounds with your hand:")
        print("1. Finger-thumb distance -> Volume")
        print("2. Hand height -> Pitch (higher hand = higher pitch)")
        print("\nCloser distance = louder sound")
        print("Green lines indicate close proximity")
        print("Press 'q' to quit.")
        print("-----------------------------------\n")
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Failed to capture image from camera.")
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.landmark_style,
                        self.connection_style
                    )
                    
                    distances = self.calculate_finger_distances(hand_landmarks)
                    hand_height = self.get_hand_height(hand_landmarks)
                    
                    self.draw_distance_lines(image, hand_landmarks, distances)
                    self.draw_hand_height(image, hand_height)
                    
                    self.update_sounds(distances, hand_height)
                    
                    # Display distances on screen
                    y_pos = 30
                    for finger, distance in distances.items():
                        color = (0, 255, 0) if distance < self.min_distance else (0, 0, 255)
                        text = f"{self.finger_names[finger]}: {distance:.3f}"
                        cv2.putText(image, text, 
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, color, 2)
                        y_pos += 30
            else:
                for channel in self.sound_channels.values():
                    channel.stop()

            cv2.imshow('Hand Music Instrument', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        for channel in self.sound_channels.values():
            channel.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    instrument = HandMusicInstrument()
    instrument.run() 