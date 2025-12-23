# visualizer.py
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
CYAN = (0, 255, 255)


class Visualizer:
    def __init__(self, world_width, world_height):
        pygame.init()
        self.world_width = world_width
        self.world_height = world_height

        # Scale factor to fit screen
        self.scale_x = SCREEN_WIDTH / world_width
        self.scale_y = SCREEN_HEIGHT / world_height

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"VECN Sim - {SIMULATION_MODE}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

    def _world_to_screen(self, pos):
        """Converts world coordinates to screen pixel coordinates."""
        sx = int(pos[0] * self.scale_x)
        sy = int(pos[1] * self.scale_y)
        return sx, sy

    def _draw_text(self, text, x, y, color=WHITE):
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def draw(self, uavs, vehicles, episode, step, profit):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(BLACK)

        # 1. Draw Vehicles (Dict or List support)
        veh_list = vehicles.values() if isinstance(vehicles, dict) else vehicles
        for v in veh_list:
            pos = self._world_to_screen(v.position)
            # Color depends on task status
            color = BLUE
            if any(t.status == 'PENDING' for t in v.tasks):
                color = RED
            elif any(t.status == 'COMPUTING' for t in v.tasks):
                color = CYAN

            pygame.draw.circle(self.screen, color, pos, 3)

        # 2. Draw UAVs and Range
        for uav in uavs:
            pos = self._world_to_screen(uav.position)

            # Draw Range (Transparent)
            range_px = int(UAV_COMMUNICATION_RANGE * self.scale_x)
            surf = pygame.Surface((range_px * 2, range_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (0, 255, 0, 40), (range_px, range_px), range_px)
            self.screen.blit(surf, (pos[0] - range_px, pos[1] - range_px))

            # Draw UAV Body
            pygame.draw.circle(self.screen, GREEN, pos, 6)

            # Draw Links to served vehicles
            # (Optional visualization optimization: only draw lines if explicitly needed to save FPS)
            # for v in veh_list:
            #     if v.id in uav.serving_ids: ...

        # 3. HUD
        self._draw_text(f"Episode: {episode} | Step: {step}", 10, 10)
        self._draw_text(f"UAVs: {len(uavs)}", 10, 30)
        self._draw_text(f"Current Profit: {profit:.2f}", 10, 50)
        self._draw_text(f"Mode: {SIMULATION_MODE}", 10, 70)

        # Snapshot Logic
        if (SAVE_VISUALIZATION_IMAGES and
                episode in EPISODES_TO_SNAPSHOT and
                step in STEPS_TO_SNAPSHOT):
            os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
            path = os.path.join(IMAGE_SAVE_PATH, f"ep{episode}_step{step}.png")
            pygame.image.save(self.screen, path)

        pygame.display.flip()
        # Cap FPS
        self.clock.tick(60)

    def close(self):
        pygame.quit()
