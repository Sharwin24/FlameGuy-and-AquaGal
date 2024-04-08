import pygame
from pygame.locals import *

# Game Classes
from game import Game
from board import Board
from character import MagmaBoy, HydroGirl
from controller import ArrowsController, WASDController, GeneralController
from ai_controller import AIController
from gates import Gates
from doors import FireDoor, WaterDoor
from level_select import LevelSelect
from collectibles import FireCollectible, WaterCollectible

# The possible tile types are:
# 100 - solid block with no borders
# 111 - solid block with top border
# 112 - solid block with right border
# 113 - solid block with bottom border
# 114 - solid block with left border
# 121 - solid block with top and right borders
# 122 - solid block with bottom and right borders
# 123 - solid block with bottom and left borders
# 124 - solid block with top and left borders
# 2 - lava
# 3 - water
# 4 - goo
# 0 - empty space
# The borders determine which side the character can walk on or collide with.

textures = {
	"100": pygame.image.load('data/board_textures/100.png'),
	"100": pygame.image.load('data/board_textures/100.png'),
	"111": pygame.image.load('data/board_textures/111.png'),
	"112": pygame.image.load('data/board_textures/112.png'),
	"113": pygame.image.load('data/board_textures/113.png'),
	"114": pygame.image.load('data/board_textures/114.png'),
	"121": pygame.image.load('data/board_textures/121.png'),
	"122": pygame.image.load('data/board_textures/122.png'),
	"123": pygame.image.load('data/board_textures/123.png'),
	"124": pygame.image.load('data/board_textures/124.png'),
	"2": pygame.image.load('data/board_textures/lava.png'),
	"3": pygame.image.load('data/board_textures/water.png'),
	"4": pygame.image.load('data/board_textures/goo.png')
}

NUM_ROWS = 25
NUM_COLS = 34
CHUNK_SIZE = 16

# The class LevelCreator is used to create levels from text files. The text files
# are used to define the layout of the level. The text files contain a 2D array of
# numbers. Each number corresponds to a tile type. The LevelCreator class reads the
# text file, creates a Board object, and populates the Board object with the correct
# tiles. The text file also contains information about the location of doors, collectibles,
# and player starting positions. The LevelCreator class reads this information and
# creates the corresponding objects.
class LevelCreator():
  pass

# We want an easy way to create these text files, so we can create a level editor
# that allows us to visually create levels and save them as text files. The level
# editor will have a grid where we can place different tiles, such as walls, lava,
# water, etc. We can also place doors, collectibles, and player starting positions.
# The level editor will save the level as a text file, which can be loaded by the
# LevelCreator class. The level editor

class Tile():
	'''A tile is a square that can be selected and placed on the grid. 
 		 Each tile has a specific value and texture'''
	def __init__(self, value: str) -> None:
		self.value = value
		self.texture = textures[value]

class Square():
  '''A square is a spot on the grid that can be populated with a tile'''
  def __init__(self, x: int, y: int, width: int, height: int) -> None:
    self.rect = pygame.Rect(x, y, width, height)
		# default texture is white square
    self.texture = pygame.Surface((width, height))
    self.tile = None
    
  def set_square(self, tile: Tile) -> None:
    self.tile = tile
    self.texture = tile.texture

  def draw(self, screen: pygame.Surface) -> None:
    screen.blit(self.texture, self.rect)

class LevelEditor():
	'''Display a grid of squares with size (NUM_ROWS x NUM_COLS) and
		 allow the user to place different tiles on the grid by clicking on a
     square within the grid to select it, then clicking on a tile-type from
     a palette underneath the grid to place the selected tile on the grid.'''
	def __init__(self) -> None:
		self.grid = self.create_grid()
		self.setup_pygame()

	def create_grid(self):
		# Create empty grid
		grid = []
		for row in range(NUM_ROWS):
			grid.append([])
			for col in range(NUM_COLS):
				grid[row].append(Square(col, row, CHUNK_SIZE, CHUNK_SIZE))


	def setup_pygame(self):
		pygame.init()
		self.screen = pygame.display.set_mode((800, 600))
		pygame.display.set_caption('Level Editor')