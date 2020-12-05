import sys
import pygame
from pygame.locals import *


class Board:
    def __init__(self):
        """
        Initialize pygame window and internal display.

        The internal display is a smaller resolution than the window display.
        The internal display is scaled up to match the larger window display.
        """

        # create internal game display
        CHUNK_SIZE = 16
        DISPLAY_SIZE = (32 * CHUNK_SIZE, 24 * CHUNK_SIZE)
        self.display = pygame.Surface(DISPLAY_SIZE)

    def load_map(self, path):
        '''
        Create an array which contains the type of every chunk on the map.

        Each level map is made up of 24x32 chunks. Each type of chunk has
        specfic texture and properties. Each unique chunks type has a
        unique string value.

        Args:
            path::str
                path to txt file containing chunk data
        '''
        self.game_map = []

        with open(path) as f:
            for line in f:
                line = line.strip().split(',')  # convert string to list of str
                self.game_map.append(line)

    def load_images(self):
        """
        Load all board chunk textures from local folder "data/board_textures

        Save textures in a dictionary.
        """
        self.board_image = {
            "wall": pygame.image.load('data/board_textures/wall.png'),
            "floor_100": pygame.image.load('data/board_textures/100.png'),
            "floor_100": pygame.image.load('data/board_textures/100.png'),
            "floor_111": pygame.image.load('data/board_textures/111.png'),
            "floor_112": pygame.image.load('data/board_textures/112.png'),
            "floor_113": pygame.image.load('data/board_textures/113.png'),
            "floor_114": pygame.image.load('data/board_textures/114.png'),
            "floor_121": pygame.image.load('data/board_textures/121.png'),
            "floor_122": pygame.image.load('data/board_textures/122.png'),
            "floor_123": pygame.image.load('data/board_textures/123.png'),
            "floor_124": pygame.image.load('data/board_textures/124.png'),
            "lava_image": pygame.image.load('data/board_textures/lava.png'),
            "water_image": pygame.image.load('data/board_textures/puddle.png'),
            "goo_image": pygame.image.load('data/board_textures/goo.png')
        }
        for texture in self.board_image.keys():
            self.board_image[texture].set_colorkey((255, 0, 255))

    def construct_board(self):
        """
        Draw the board.

        Iterate through the game map draw each chunk.
        """
        # draw the full background
        self.display.blit(self.board_image["wall"], (0, 0))

        # draw the solid blocks and liquids
        for y, row in enumerate(self.game_map):
            for x, tile in enumerate(row):
                if tile == '100':
                    self.display.blit(
                        self.board_image["floor_100"], (x * 16, y * 16))
                if tile == '111':
                    self.display.blit(
                        self.board_image["floor_111"], (x * 16, y * 16))
                if tile == '112':
                    self.display.blit(
                        self.board_image["floor_112"], (x * 16, y * 16))
                if tile == '113':
                    self.display.blit(
                        self.board_image["floor_113"], (x * 16, y * 16))
                if tile == '114':
                    self.display.blit(
                        self.board_image["floor_114"], (x * 16, y * 16))
                if tile == '121':
                    self.display.blit(
                        self.board_image["floor_121"], (x * 16, y * 16))
                if tile == '122':
                    self.display.blit(
                        self.board_image["floor_122"], (x * 16, y * 16))
                if tile == '123':
                    self.display.blit(
                        self.board_image["floor_123"], (x * 16, y * 16))
                if tile == '124':
                    self.display.blit(
                        self.board_image["floor_124"], (x * 16, y * 16))
                if tile == '2':
                    self.display.blit(
                        self.board_image["lava_image"], (x * 16, y * 16))
                if tile == '3':
                    self.display.blit(
                        self.board_image["water_image"], (x * 16, y * 16))
                if tile == '4':
                    self.display.blit(
                        self.board_image["goo_image"], (x * 16, y * 16))

    def make_solid_blocks(self):
        """
        Iterate through the map and make the walls and ground solid blocks
        which the player can collide with.
        """
        self.solid_blocks = []
        for y, row in enumerate(self.game_map):
            for x, tile in enumerate(row):
                if tile not in ['0', '2', '3', '4']:
                    self.solid_blocks.append(
                        pygame.Rect(x * 16, y * 16, 16, 16))

    def get_solid_blocks(self):
        """
        Return a list of pygame rects that are solid.
        """
        return self.solid_blocks

    def get_board(self):
        """
        Return the pygame internal display.
        """
        return self.display
