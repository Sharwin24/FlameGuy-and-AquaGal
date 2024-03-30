from game import Game
import pygame
from pygame.locals import *
from gates import Gates
from doors import FireDoor, WaterDoor
from collectibles import FireCollectible, WaterCollectible
from board import Board
from character import MagmaBoy, HydroGirl
from controller import GeneralController
import sys
import numpy as np

class Training_Game():
    def __init__(self, game, controller, level="level1"):
        self.game = game
        self.controller = controller
        
        if level == "level1":
            self.board = Board('data/level1.txt')
            self.gate_location = (285, 128)
            self.plate_locations = [(190, 168), (390, 168)]
            self.gate = Gates(self.gate_location, self.plate_locations)
            self.gates = [self.gate]

            self.fire_door_location = (64, 48)
            self.fire_door = FireDoor(self.fire_door_location)
            self.water_door_location = (128, 48)
            self.water_door = WaterDoor(self.water_door_location)
            self.doors = [self.fire_door, self.water_door]

            self.magma_boy_location = (16, 336)
            self.magma_boy = MagmaBoy(self.magma_boy_location)
            self.hydro_girl_location = (35, 336)
            self.hydro_girl = HydroGirl(self.hydro_girl_location)
            
            # arrays for collectibles
            scaling_fac = 16
            self.fire_collectibles = [FireCollectible((11.5 * scaling_fac, 21 * scaling_fac)), 
                                FireCollectible((8.5 * scaling_fac, 15 * scaling_fac)),
                                FireCollectible(((12 - (1/8)) * scaling_fac, 9 * scaling_fac))]
            self.water_collectibles = [WaterCollectible((19.5 * scaling_fac, 21 * scaling_fac)),
                                WaterCollectible((17.5 * scaling_fac, 15 * scaling_fac)),
                                WaterCollectible(((24 + (3/8)) * scaling_fac, 9 * scaling_fac))]
            
        # initializing clock for timeouts
        self.clock = pygame.time.Clock()

        # run the actual game
        self.update_loop()
        
    # runs the actual game loop
    def update_loop(self):
        # pygame management
        self.clock.tick(60)
        events = pygame.event.get()

        # draw features of level
        self.game.draw_level_background(self.board)
        self.game.draw_board(self.board)
        if self.gates:
            self.game.draw_gates(self.gates)
        self.game.draw_doors(self.doors)

        # draw player
        self.game.draw_player([self.magma_boy, self.hydro_girl])
        
        # draw collcetibles
        self.game.draw_collectibles(self.fire_collectibles + self.water_collectibles)

        self.game.move_player(self.board, self.gates, [self.magma_boy, self.hydro_girl])

        # check for player at special location
        self.game.check_for_death(self.board, [self.magma_boy, self.hydro_girl])

        self.game.check_for_gate_press(self.gates, [self.magma_boy, self.hydro_girl])

        self.game.check_for_door_open(self.fire_door, self.magma_boy)
        self.game.check_for_door_open(self.water_door, self.hydro_girl)
        
        # checking to see if collectibles were hit
        for fire_collectible in self.fire_collectibles:
            self.game.check_for_collectible_hit(fire_collectible, self.magma_boy)
            
        for water_collectible in self.water_collectibles:
            self.game.check_for_collectible_hit(water_collectible, self.hydro_girl)

        # refresh window
        self.game.refresh_window()

        # special events
        if self.hydro_girl.is_dead() or self.magma_boy.is_dead():
            sys.exit()

        if self.game.level_is_done(self.doors):
            sys.exit()

        if self.controller.press_key(events, K_ESCAPE):
            sys.exit()

        # close window is player clicks on [x]
        for event in events:
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
    # define an abstract action space (think of this as pressing a key)
    def move_player(self, player, dir):
        if dir == "right":
            player.moving_right = True
        elif dir == "left":
            player.moving_left = True
        elif dir == "jump":
            player.jumping = True
            
        self.update_loop()
        
    # define the above abstract space, but for un-moving (think of this as letting go of a key)
    def unmove_player(self, player, dir):
        if dir == "right":
            player.moving_right = False
        elif dir == "left":
            player.moving_left = False
        elif dir == "jump":
            player.jumping = False
            
        self.update_loop()
    
    # move magma boy
    def move_magma(self, dir):
        self.move_player(self.magma_boy, dir)
        
    # unmove magma boy
    def unmove_magma(self, dir):
        self.unmove_player(self.magma_boy, dir)
        
    # move hydro girl
    def move_hydro(self, dir):
        self.move_player(self.hydro_girl, dir)
        
    # unmove hydro girl
    def unmove_hydro(self, dir):
        self.move_player(self.hydro_girl, dir)
    
    # return the board as a 3d array (change this to torch tensor at some point?)
    def return_board(self):
        return pygame.surfarray.array3d(self.game.display)
    
    # loss function - sum of the time and distance from nearest gem
    def loss_function(self): 
        return 0 # that is, when i'm not incredibly lazy
                    
controller = GeneralController()
game = Game()
tg = Training_Game(game, controller)

# things to do from here - create a NN to minimize loss function
# - NN takes in tg.return_board as an input (change this to be a torch tensor, make it work on GPU)
# - make loss function work?
# - SGD optimizer?
# and then we somehow export this model into its own controller object, which we can "plug in" to main function