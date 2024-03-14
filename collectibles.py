import sys
import pygame
from pygame.locals import *

# collectible class for a collectible with a location and filename
class Collectible:
    def __init__(self, location, image_filename):
        self.location = location
        self.image_filename = image_filename
        self.is_collected = False
        
        self.load_image()
        self.make_rects()
    
    def load_image(self):
        """
        load image for collectible
        """    
        # load image, set it to transparent
        self.image = pygame.image.load(self.image_filename)
        self.image.set_colorkey((255, 0, 255)) # i don't fully understand why we set to transparent here? maybe change
    
    def make_rects(self):
        """
        make rects for collectible
        """
        x_loc = self.location[0]
        y_loc = self.location[1]
        
        self._rect = pygame.Rect(x_loc, y_loc, self.image.get_width(), self.image.get_height())
        
    def get_collectible(self):
        return self._rect

# collectible class for flameguy
class FireCollectible(Collectible):
    def __init__(self, location):
        super().__init__(location, 'data/collectible_images/fire_gem.png')
        
# collectible class for aquagal
class WaterCollectible(Collectible):
    def __init__(self, location):
        super().__init__(location, 'data/collectible_images/water_gem.png')        