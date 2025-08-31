# visualizer.py
"""
Handles the Pygame-based visualization of the simulation environment.
"""
import sys

import pygame

from config import *

# Define Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)


class Visualizer:
    def __init__(self, world_width, world_height):
        pygame.init()
        self.world_width = world_width
        self.world_height = world_height
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Multi-UAV VECN Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def _world_to_screen(self, pos):
        """Converts world coordinates to screen pixel coordinates."""
        screen_x = int((pos[0] / self.world_width) * self.screen_width)
        screen_y = int((pos[1] / self.world_height) * self.screen_height)
        return screen_x, screen_y

    def _draw_text(self, text, x, y, color=WHITE):
        """Helper to draw text on the screen."""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def draw(self, uavs, vehicles, episode, step):
        """Draws the entire simulation state for one frame."""

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # --- Drawing ---
        self.screen.fill(BLACK)

        # Draw Vehicles
        for vehicle in vehicles:
            pos = self._world_to_screen(vehicle.position)
            pygame.draw.circle(self.screen, BLUE, pos, 3)  # Small blue dots for vehicles

        # Draw UAVs and their communication range
        for uav in uavs:
            pos = self._world_to_screen(uav.position)

            # Draw communication range first (as a semi-transparent circle)
            range_radius = int((UAV_COMMUNICATION_RANGE / self.world_width) * self.screen_width)

            # Create a transparent surface for the range circle
            range_surface = pygame.Surface((range_radius * 2, range_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(range_surface, (0, 255, 0, 60), (range_radius, range_radius), range_radius)
            self.screen.blit(range_surface, (pos[0] - range_radius, pos[1] - range_radius))

            # Draw the UAV itself
            pygame.draw.circle(self.screen, GREEN, pos, 6)  # Larger green circles for UAVs

        # --- Draw Info Text ---
        self._draw_text(f"Episode: {episode}", 10, 10)
        self._draw_text(f"Step: {step}/{INNER_STEPS}", 10, 30)
        self._draw_text(f"UAVs Deployed: {len(uavs)}", 10, 50)

        # --- Update Display ---
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS to make it watchable
