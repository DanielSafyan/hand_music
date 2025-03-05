import cv2
import mediapipe as mp
import pygame
import numpy as np
from math import hypot, sin, pi
import pygame.sndarray

class SoundPoint:
    def __init__(self, x, y, color, base_freq, name):
        self.x = x  # Screen x coordinate
        self.y = y  # Screen y coordinate
        self.color = color  # BGR color tuple
        self.base_freq = base_freq  # Base frequency when at starting position
        self.current_freq = base_freq  # Current frequency (changes with x position)
        self.name = name  # Name of the note
        self.intensity = 10  # Starting intensity (0-100)
        self.radius = 15  # Visual radius of the point
        self.is_grabbed = False  # Whether the point is currently grabbed
        self.last_touched = 0  # Timestamp of last touch (for foreground ordering)

class HandMusicInstrument2:
    def __init__(self):
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                       max_num_hands=1,
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
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Create sound channels
        pygame.mixer.set_num_channels(4)
        self.sound_channels = [pygame.mixer.Channel(i) for i in range(4)]
        
        # Calculate 2x2 grid positions in bottom left corner
        margin = 50  # margin from edges
        spacing = 40  # spacing between points
        base_x = margin
        base_y = self.frame_height - margin
        
        # Create sound points in 2x2 grid
        self.sound_points = [
            # Bottom row
            SoundPoint(base_x, base_y, 
                      (255, 50, 50), 262, "C4"),
            SoundPoint(base_x + spacing, base_y, 
                      (50, 255, 50), 330, "E4"),
            # Top row
            SoundPoint(base_x, base_y - spacing, 
                      (50, 50, 255), 392, "G4"),
            SoundPoint(base_x + spacing, base_y - spacing, 
                      (255, 255, 50), 523, "C5")
        ]
        
        # Frequency control parameters
        self.min_freq = 220  # A3
        self.max_freq = 880  # A5
        
        # Grab parameters
        self.point_radius = 15  # Base radius for sound points
        self.grab_radius = int(self.point_radius * 1.5)  # 50% larger grab radius
        self.grab_threshold = 0.05  # Very small threshold for thumb-index touch
        self.current_time = 0  # Time counter for touch ordering
        
        # Landmark indices
        self.index_tip = 8
        self.thumb_tip = 4

    def check_grab_gesture(self, hand_landmarks):
        """Check if thumb and index finger are touching"""
        thumb = hand_landmarks.landmark[self.thumb_tip]
        index = hand_landmarks.landmark[self.index_tip]
        
        distance = hypot(thumb.x - index.x, thumb.y - index.y)
        return distance < self.grab_threshold  # True only when fingers are very close

    def get_index_position(self, hand_landmarks):
        """Get the position of the index finger tip"""
        index = hand_landmarks.landmark[self.index_tip]
        return (int(index.x * self.frame_width), int(index.y * self.frame_height))

    def is_point_near_finger(self, point, finger_pos):
        """Check if a point is near enough to the index finger to be grabbed"""
        distance = hypot(finger_pos[0] - point.x, finger_pos[1] - point.y)
        return distance < self.grab_radius

    def update_point_position(self, point, finger_pos):
        """Update position of grabbed point"""
        if point.is_grabbed:
            # Update position to follow finger
            point.x = finger_pos[0]
            point.y = finger_pos[1]
            
            # Update intensity based on y position (higher = more intense)
            point.intensity = 100 - (point.y / self.frame_height * 90 + 10)
            point.intensity = max(10, min(100, point.intensity))
            
            # Update frequency based on x position
            # Map x position from 0 to screen width to frequency range
            freq_multiplier = point.x / self.frame_width
            point.current_freq = self.min_freq * pow(self.max_freq/self.min_freq, freq_multiplier)

    def generate_wave(self, frequency, amplitude):
        """Generate a sine wave"""
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        wave = wave * (32767 / max(abs(wave)))
        return wave.astype(np.int16)

    def update_sounds(self):
        """Update all sound points"""
        for idx, point in enumerate(self.sound_points):
            # Generate wave with amplitude based on intensity
            amplitude = int(self.max_amplitude * (point.intensity / 100))
            wave = self.generate_wave(point.current_freq, amplitude)
            sound = pygame.sndarray.make_sound(wave)
            
            # Play sound continuously
            channel = self.sound_channels[idx]
            channel.play(sound, -1)
            channel.set_volume(point.intensity / 100)

    def draw_hand(self, image, hand_landmarks):
        """Draw hand landmarks and connections"""
        # Draw hand skeleton
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = hand_landmarks.landmark[start_idx]
            end_point = hand_landmarks.landmark[end_idx]
            
            start_pos = (int(start_point.x * self.frame_width), 
                        int(start_point.y * self.frame_height))
            end_pos = (int(end_point.x * self.frame_width), 
                      int(end_point.y * self.frame_height))
            
            cv2.line(image, start_pos, end_pos, (255, 255, 255), 2)
        
        # Draw smaller, more transparent grab area around index finger
        index_pos = self.get_index_position(hand_landmarks)
        is_grabbing = self.check_grab_gesture(hand_landmarks)
        
        # Create semi-transparent overlay
        overlay = image.copy()
        circle_color = (0, 255, 0) if is_grabbing else (0, 100, 255)
        cv2.circle(overlay, index_pos, self.grab_radius, circle_color, 1)  # Thinner line
        
        # Apply transparency
        alpha = 0.3  # 30% opacity
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_sound_points(self, image):
        """Draw sound points and their intensity values"""
        for point in self.sound_points:
            # Draw the point
            cv2.circle(image, (point.x, point.y), point.radius, 
                      point.color, -1 if point.is_grabbed else 2)
            
            # Draw intensity value, frequency, and grab status
            status = "GRABBED" if point.is_grabbed else f"{int(point.intensity)}%"
            freq_text = f"{int(point.current_freq)}Hz"
            cv2.putText(image, f"{point.name}: {status}, {freq_text}",
                       (point.x - 60, point.y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, point.color, 2)

    def update_point_order(self):
        """Sort sound points by last touched time (most recent first)"""
        self.sound_points.sort(key=lambda x: x.last_touched, reverse=True)

    def is_any_point_grabbed(self):
        """Check if any point is currently grabbed"""
        return any(point.is_grabbed for point in self.sound_points)

    def release_all_points(self):
        """Release all points"""
        for point in self.sound_points:
            point.is_grabbed = False

    def run(self):
        print("\nHand Music Instrument 2 Instructions:")
        print("-------------------------------------")
        print("Grab and move sound points:")
        print("1. Touch point with index finger")
        print("2. Touch thumb to index finger to grab")
        print("3. Move up/down to change volume")
        print("4. Move left/right to change frequency")
        print("5. Separate thumb and index to release")
        print("\nControls:")
        print("Volume: Bottom (10%) to Top (100%)")
        print("Frequency: Left (220Hz) to Right (880Hz)")
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

            # Increment time counter
            self.current_time += 1

            # Draw sound points in order (last touched on top)
            self.update_point_order()
            self.draw_sound_points(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand with grab area
                    self.draw_hand(image, hand_landmarks)
                    
                    # Get index finger position and grab state
                    index_pos = self.get_index_position(hand_landmarks)
                    is_grabbing = self.check_grab_gesture(hand_landmarks)
                    
                    # If not grabbing, release all points
                    if not is_grabbing:
                        self.release_all_points()
                    else:
                        # Only check for new grabs if no point is currently grabbed
                        if not self.is_any_point_grabbed():
                            # Check points in reverse order (most recently touched first)
                            for point in self.sound_points:
                                if self.is_point_near_finger(point, index_pos):
                                    point.is_grabbed = True
                                    point.last_touched = self.current_time
                                    # Only grab one point
                                    break
                    
                    # Update positions of grabbed points
                    for point in self.sound_points:
                        if point.is_grabbed:
                            self.update_point_position(point, index_pos)
                    
                    # Update sounds
                    self.update_sounds()
            else:
                # Release all points when hand is not detected
                self.release_all_points()

            cv2.imshow('Hand Music Instrument 2', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        for channel in self.sound_channels:
            channel.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    instrument = HandMusicInstrument2()
    instrument.run() 