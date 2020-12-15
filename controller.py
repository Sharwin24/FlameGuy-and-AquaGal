import sys
import pygame
from pygame.locals import *

class Controller:
    def __init__(self, player):
        self.player = player
    
    def control_player(self, events):
        for event in events:
            if event.type == KEYDOWN:
                if event.key == self.controls["right"]:
                    self.player.moving_right = True
                elif event.key == self.controls["left"]:
                    self.player.moving_left = True
                elif event.key == self.controls["up"]:
                    if self.player.air_timer < 6:
                        self.player.jumping = True

            elif event.type == KEYUP:
                if event.key == self.controls["right"]:
                    self.player.moving_right = False
                elif event.key == self.controls["left"]:
                    self.player.moving_left = False
                elif event.key == self.controls["up"]:
                    self.player.jumping = False
    
    def press_key(events, key):
        for event in events:
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == key:
                    return True
        return False           


class MagmaBoyController(Controller):
    def __init__(self, player):
        super().__init__(player)
        self.controls = {
            "left": K_LEFT,
            "right": K_RIGHT,
            "up": K_UP
        }

class HydroGirlController(Controller):
    def __init__(self, player):
        super().__init__(player)
        self.controls = {
            "left": K_a,
            "right": K_d,
            "up": K_w
        }

