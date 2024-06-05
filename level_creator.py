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
  "selected" : pygame.image.load('data/board_textures/selected.png'),
  "0": pygame.image.load('data/board_textures/0.png'), # "empty
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
		if value in textures:
			self.texture = textures[value]
		else:
			self.texture = pygame.Surface((CHUNK_SIZE, CHUNK_SIZE))
			self.texture.fill((255, 255, 255))
			print("Invalid tile value: " + value)

class Square():
  '''A square is a spot on the grid that can be populated with a tile'''
  def __init__(self, x: int, y: int, width: int, height: int) -> None:
    self.rect = pygame.Rect(x, y, width, height)
		# default texture is white square
    self.texture = pygame.Surface((width, height))
    self.tile = Tile("0") # empty space
    
  def set_square(self, tile: Tile) -> None:
    self.tile = tile
    self.texture = tile.texture
    
  def draw(self, screen: pygame.Surface) -> None:
    screen.blit(self.texture, self.rect)

keystroke_to_tile = {
  pygame.K_SPACE: Tile("100"), # empty
  pygame.K_w: Tile("111"), # top
  pygame.K_d: Tile("112"), # right
  pygame.K_s: Tile("113"), # bottom
  pygame.K_a: Tile("114"), # left
  pygame.K_e: Tile("121"), # top and right
  pygame.K_c: Tile("122"), # bottom and right
  pygame.K_z: Tile("123"), # bottom and left
  pygame.K_q: Tile("124"), # top and left
  pygame.K_l: Tile("2"), # lava
  pygame.K_k: Tile("3"), # water
  pygame.K_j: Tile("4"), # goo
  pygame.K_0: Tile("0") # empty space
}

class LevelEditor():
	'''Display a grid of squares with size (NUM_ROWS x NUM_COLS) and
		 allow the user to place different tiles on the grid by clicking on a
     square within the grid to select it, then typing on the keyboard to place a tile'''
	def __init__(self) -> None:
		self.create_grid()
		self.setup_pygame()
		self.draw_grid()
		self.selected_square = None

	def create_grid(self):
		# Create empty grid
		self.grid = []
		for row in range(NUM_ROWS):
			self.grid.append([])
			for col in range(NUM_COLS):
				self.grid[row].append(Square(col, row, CHUNK_SIZE, CHUNK_SIZE))
  
	def draw_grid(self):
		for row in self.grid:
			for square in row:
				square.draw(self.screen)
		pygame.display.flip()
		# Draw grid using black lines
		for row in range(NUM_ROWS):
			pygame.draw.line(self.screen, (0, 0, 0), (0, row * CHUNK_SIZE), (NUM_COLS * CHUNK_SIZE, row * CHUNK_SIZE))
		for col in range(NUM_COLS):
			pygame.draw.line(self.screen, (0, 0, 0), (col * CHUNK_SIZE, 0), (col * CHUNK_SIZE, NUM_ROWS * CHUNK_SIZE))

	def setup_pygame(self):
		pygame.init()
		size = (NUM_COLS * CHUNK_SIZE, NUM_ROWS * CHUNK_SIZE)
		self.screen = pygame.display.set_mode(size)
		pygame.display.set_caption('Level Editor')
		# Set background color to white
		self.screen.fill((255, 255, 255))
		pygame.display.flip()
  
	def unselect_square(self):
		if self.selected_square is not None:
			self.selected_square.texture = textures[self.selected_square.tile.value]
			self.selected_square.draw(self.screen)
			self.selected_square = None
			pygame.display.flip()
   
	def highlight_selected_square(self):
		if self.selected_square is not None:
			pygame.draw.rect(self.screen, (255, 0, 0), self.selected_square.rect, 2)
			self.selected_square.draw(self.screen)
			pygame.display.flip()

	def set_selected_square(self, mouse_click):
		print("Mouse clicked at: " + str(mouse_click))
		for row in self.grid:
			for square in row:
				if square.rect.collidepoint(mouse_click):
					if self.selected_square is not None:
						self.unselect_square()
					self.selected_square = square
					self.highlight_selected_square()
					return
  
	def place_tile(self, key):
		if self.selected_square is not None:
			tile = keystroke_to_tile[key]
			self.selected_square.set_square(tile)
			self.selected_square.draw(self.screen)
			pygame.display.flip()
  
	def run(self):
    # During the game loop, check for mouse clicks to obtain
		# the selected square, then check for keyboard input to place a tile
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == QUIT:
					running = False
				elif event.type == MOUSEBUTTONDOWN:
					self.set_selected_square(event.pos)
				elif event.type == KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						self.unselect_square()
					elif event.key in keystroke_to_tile:
						self.place_tile(event.key)
					else:
						print("Invalid key pressed: " + str(event.key))
		pygame.quit()

if __name__ == '__main__':
  level_editor = LevelEditor()
  level_editor.run()