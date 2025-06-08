import pygame
import pylsl
import time
import threading
import sys

# Pygame Initialization
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BCI Game Controller")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game Object (Ball)
ball_radius = 40
ball_color = GREEN
ball_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
ball_speed = 5 # Pixels per command

# LSL Inlet for Commands
COMMAND_STREAM_NAME = 'BCI_Commands'
command_inlet = None
latest_command = "No Command" # Global variable to store the latest command

def find_lsl_stream(name):
    """Finds an LSL stream by name."""
    print(f"Looking for {name} stream...")
    streams = pylsl.resolve_byprop('name', name, timeout=10)
    if not streams:
        print(f"No {name} stream found. Is `realtime_bci_classifier.py` running and streaming?")
        return None
    print(f"Found {name} stream.")
    return pylsl.StreamInlet(streams[0])

def lsl_command_listener():
    """Thread function to listen for BCI commands from LSL."""
    global command_inlet, latest_command
    
    command_inlet = find_lsl_stream(COMMAND_STREAM_NAME)
    if not command_inlet:
        print("Could not connect to command stream. Game will run without BCI control.")
        return

    print("Listening for BCI commands...")
    while True:
        try:
            sample, timestamp = command_inlet.pull_sample(timeout=0.01) # non-blocking pull
            if sample:
                latest_command = sample[0]
                # print(f"Received BCI command: {latest_command} at {timestamp}")
            time.sleep(0.001) # Small sleep to prevent busy-waiting
        except pylsl.timeout:
            continue
        except Exception as e:
            print(f"Error in LSL command listener: {e}")
            break

def apply_command_to_ball(command):
    """Applies the received command to the ball's position or state."""
    global ball_pos, ball_radius, ball_color

    if command == "LEFT":
        ball_pos[0] -= ball_speed
        ball_color = BLUE
    elif command == "RIGHT":
        ball_pos[0] += ball_speed
        ball_color = BLUE
    elif command == "UP":
        ball_pos[1] -= ball_speed
        ball_color = BLUE
    elif command == "DOWN":
        ball_pos[1] += ball_speed
        ball_color = BLUE
    elif command == "PUSH":
        ball_radius = min(100, ball_radius + 2) # Max radius 100
        ball_color = RED
    elif command == "PULL":
        ball_radius = max(10, ball_radius - 2) # Min radius 10
        ball_color = RED
    else: # No_Command or other
        ball_color = GREEN # Default color

    # Keep ball within screen bounds
    ball_pos[0] = max(ball_radius, min(SCREEN_WIDTH - ball_radius, ball_pos[0]))
    ball_pos[1] = max(ball_radius, min(SCREEN_HEIGHT - ball_radius, ball_pos[1]))


def main_game_loop():
    global latest_command
    running = True
    clock = pygame.time.Clock()

    # Start LSL listener thread
    listener_thread = threading.Thread(target=lsl_command_listener)
    listener_thread.daemon = True
    listener_thread.start()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.K_ESCAPE:
                    running = False

        # Apply the latest BCI command
        apply_command_to_ball(latest_command)
        
        # Reset command after processing to avoid continuous action from single command
        # This makes discrete actions. For continuous actions, you might keep the command until a new one.
        # For this demo, let's keep it to show it reacts to *each* detected command
        # latest_command = "No Command" 

        screen.fill(BLACK) # Clear screen

        # Draw the ball
        pygame.draw.circle(screen, ball_color, (int(ball_pos[0]), int(ball_pos[1])), int(ball_radius))

        # Display current command for debugging
        font = pygame.font.Font(None, 36)
        text_surface = font.render(f"Command: {latest_command}", True, WHITE)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip() # Update display
        clock.tick(60) # Limit frame rate to 60 FPS

    pygame.quit()
    print("Game exited.")
    if command_inlet:
        command_inlet.close_stream()
    sys.exit()

if __name__ == "__main__":
    main_game_loop()

