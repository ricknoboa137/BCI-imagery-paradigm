import pygame
import pylsl
import time
import random
import numpy as np

# Pygame Initialization
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Motor Imagery Training Paradigm")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0) # Originally for PUSH
BLUE = (0, 0, 255) # Originally for PULL
GREEN = (0, 255, 0)


# Font
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)

# LSL Outlet for Markers
print("Setting up LSL stream for markers...")
info = pylsl.StreamInfo('MindRove_Markers', 'Markers', 1, 0, 'string', 'myuid_markers')
outlet = pylsl.StreamOutlet(info)
print("LSL marker stream ready.")

# --- Paradigm Parameters ---
TRIAL_DURATION_IMAGERY = 4  # seconds for active imagination
TRIAL_DURATION_CUE = 2      # seconds for cue display
TRIAL_DURATION_FIXATION = 2 # seconds for fixation cross
TRIAL_DURATION_REST = 2     # seconds for rest period between trials

NUM_TRIALS_PER_CLASS = 20   # Adjust as needed for data collection
CLASSES = {
    "PUSH": {"color": RED, "direction": None},
    "PULL": {"color": BLUE, "direction": None},
    "LEFT": {"color": GREEN, "direction": (-1, 0)},
    "RIGHT": {"color": GREEN, "direction": (1, 0)},
    "UP": {"color": GREEN, "direction": (0, -1)},
    "DOWN": {"color": GREEN, "direction": (0, 1)},
}

# Ball properties
ball_radius = 50
original_ball_radius = ball_radius
ball_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
ball_color = WHITE
ball_speed = 5 # For cue animation

def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.center = (x, y)
    surface.blit(textobj, textrect)

def send_marker(marker_value):
    """Sends a string marker to the LSL stream."""
    outlet.push_sample([marker_value])
    print(f"Marker Sent: {marker_value}")

def run_paradigm():
    trial_sequence = []
    for _ in range(NUM_TRIALS_PER_CLASS):
        trial_sequence.extend(list(CLASSES.keys()))
    random.shuffle(trial_sequence) # Randomize trial order

    running = True
    trial_count = 0
    current_trial_class = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if trial_count >= len(trial_sequence):
            print("All trials completed!")
            running = False
            break

        current_trial_class = trial_sequence[trial_count]
        print(f"\n--- Starting Trial {trial_count + 1}/{len(trial_sequence)}: {current_trial_class} ---")

        # 1. Fixation Cross
        screen.fill(BLACK)
        draw_text("+", font, WHITE, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        pygame.display.flip()
        send_marker(f"FIXATION_START")
        time.sleep(TRIAL_DURATION_FIXATION)
        send_marker(f"FIXATION_END")

        # 2. Cue Presentation + Animation
        screen.fill(BLACK)
        draw_text(f"IMAGINE {current_trial_class}", font, WHITE, screen, SCREEN_WIDTH // 2, 100)
        
        # Reset ball for animation
        ball_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        ball_radius = original_ball_radius

        cue_start_time = time.time()
        while (time.time() - cue_start_time) < TRIAL_DURATION_CUE:
            screen.fill(BLACK)
            draw_text(f"IMAGINE {current_trial_class}", font, WHITE, screen, SCREEN_WIDTH // 2, 100)

            # --- FIX: Swapped PUSH and PULL animation logic ---
            if current_trial_class == "PUSH":
                ball_radius = max(5, ball_radius - 1) # Animate shrinking for PUSH
                pygame.draw.circle(screen, RED, (int(ball_pos[0]), int(ball_pos[1])), int(ball_radius))
            elif current_trial_class == "PULL":
                ball_radius += 1 # Animate expansion for PULL
                pygame.draw.circle(screen, BLUE, (int(ball_pos[0]), int(ball_pos[1])), int(ball_radius))
            # --- END FIX ---
            elif current_trial_class in ["LEFT", "RIGHT", "UP", "DOWN"]:
                direction_vector = np.array(CLASSES[current_trial_class]["direction"])
                ball_pos[0] += direction_vector[0] * ball_speed
                ball_pos[1] += direction_vector[1] * ball_speed
                pygame.draw.circle(screen, GREEN, (int(ball_pos[0]), int(ball_pos[1])), original_ball_radius)
                
                # Draw arrow
                arrow_color = CLASSES[current_trial_class]["color"]
                arrow_start = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                arrow_end = (SCREEN_WIDTH // 2 + direction_vector[0] * 100, SCREEN_HEIGHT // 2 + direction_vector[1] * 100)
                pygame.draw.line(screen, arrow_color, arrow_start, arrow_end, 5)
                # Draw arrow head (simplified)
                if current_trial_class == "LEFT":
                    pygame.draw.polygon(screen, arrow_color, [(arrow_end[0], arrow_end[1]), (arrow_end[0] + 20, arrow_end[1] - 10), (arrow_end[0] + 20, arrow_end[1] + 10)])
                elif current_trial_class == "RIGHT":
                    pygame.draw.polygon(screen, arrow_color, [(arrow_end[0], arrow_end[1]), (arrow_end[0] - 20, arrow_end[1] - 10), (arrow_end[0] - 20, arrow_end[1] + 10)])
                elif current_trial_class == "UP":
                    pygame.draw.polygon(screen, arrow_color, [(arrow_end[0], arrow_end[1]), (arrow_end[0] - 10, arrow_end[1] + 20), (arrow_end[0] + 10, arrow_end[1] + 20)])
                elif current_trial_class == "DOWN":
                    pygame.draw.polygon(screen, arrow_color, [(arrow_end[0], arrow_end[1]), (arrow_end[0] - 10, arrow_end[1] - 20), (arrow_end[0] + 10, arrow_end[1] - 20)])

            else: # Draw static ball if no specific animation
                 pygame.draw.circle(screen, ball_color, (int(ball_pos[0]), int(ball_pos[1])), original_ball_radius)

            pygame.display.flip()
            pygame.time.Clock().tick(60) # Limit frame rate

        send_marker(f"{current_trial_class}_CUE")

        # 3. Imagery Period
        screen.fill(BLACK)
        draw_text("IMAGINE NOW", font, WHITE, screen, SCREEN_WIDTH // 2, 100)
        # Display the ball in its original state for imagination
        pygame.draw.circle(screen, WHITE, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), original_ball_radius)
        pygame.display.flip()
        send_marker(f"{current_trial_class}_IMAGERY_START")
        time.sleep(TRIAL_DURATION_IMAGERY)
        send_marker(f"{current_trial_class}_IMAGERY_END")

        # 4. Rest Period
        screen.fill(BLACK)
        draw_text("REST", font, WHITE, screen, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        pygame.display.flip()
        send_marker("REST_START")
        time.sleep(TRIAL_DURATION_REST)
        send_marker("REST_END")

        trial_count += 1
        pygame.time.Clock().tick(60) # Maintain consistent loop time

    pygame.quit()
    print("Paradigm finished.")

if __name__ == "__main__":
    run_paradigm()

