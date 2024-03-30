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

import torch
from neural_network import Net

class Training_Game():
    def __init__(self, game, controller, level="level1"):
        self.game = game
        self.controller = controller
        self.timer = 0
        self.is_ended = False
        
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
        self.timer += 1
        
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
            self.is_ended = True
            sys.exit()

        if self.game.level_is_done(self.doors):
            self.is_ended = True
            sys.exit()

        if self.controller.press_key(events, K_ESCAPE):
            self.is_ended = True
            sys.exit()

        # close window is player clicks on [x]
        for event in events:
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
    # define an action space to move both players, and update the game
    # - note that the order is [magma_boy_action, hydro_girl_action]
    def move_players(self, dirs):
        # parse both directions and players at the same time
        for dir, player in zip(dirs, [self.magma_boy, self.hydro_girl]):
            if dir == "right":
                player.moving_right = True
            elif dir == "notright":
                player.moving_right = False
            elif dir == "left":
                player.moving_left = True
            elif dir == "notleft":
                player.moving_left = False
            elif dir == "jump":
                if player.air_timer < 6:
                    player.jumping = True
            elif dir == "notjump":
                player.jumping = False
            else:
                raise(KeyError("Not a valid move!"))

        # update game with new actions
        self.update_loop()
    
    # return the board as a 3d array (change this to torch tensor at some point?)
    def return_board(self):
        disp = pygame.surfarray.array3d(self.game.display)
        return torch.from_numpy(np.moveaxis(disp, -1, 0)).float()
    
    # define the closest gem to magma boy - if all gems are collected, the door is the closest gem
    def get_closest_gem(self, player, collectibles, door):
        for collectible in collectibles:
            if not(collectible.is_collected):
                return np.sqrt(((player.rect[0] - collectible.location[0]) ** 2) + \
                    ((player.rect[1] - collectible.location[1]) ** 2))
        return np.sqrt(((player.rect[0] - door.door_location[0]) ** 2) + \
                    ((player.rect[1] - door.door_location[1]) ** 2))
    
    def get_closest_magma_gem(self):
        return self.get_closest_gem(self.magma_boy, self.fire_collectibles, self.fire_door)
    
    def get_closest_hydro_gem(self):
        return self.get_closest_gem(self.hydro_girl, self.water_collectibles, self.water_door)
    
    # loss function - sum of the time (mess with values) and distance from nearest gem
    def loss_function(self): 
        loss_val = self.get_closest_magma_gem() + self.get_closest_hydro_gem() + self.timer
        return torch.tensor(loss_val, requires_grad = True)

# things to do from here - create a NN to minimize loss function
# - NN takes in tg.return_board as an input (change this to be a torch tensor, make it work on GPU)
# - make loss function work?
# - SGD optimizer?
# and then we somehow export this model into its own controller object, which we can "plug in" to main function

# create both AI models, playing the same game
# both models take in a board (given by tg.return_board()) and give an action from the following action space:
# - move right
# - move left
# - move up
# - unmove right
# - unmove left
# - unmove up
magma_boy_model = Net()
hydro_girl_model = Net()

# optimizer and loss function
magma_boy_optimizer = torch.optim.SGD(magma_boy_model.parameters(), lr = 0.01)
hydro_girl_optimizer = torch.optim.SGD(hydro_girl_model.parameters(), lr = 0.01)

action_dict = {
    0: "right",
    1: "notright",
    2: "left",
    3: "notleft",
    4: "jump",
    5: "notjump"
}

games = 10
max_iterations = 1000
run_thing = False
if run_thing:
    for j in range(games):
        # initialize a new game
        controller = GeneralController()
        game = Game()
        tg = Training_Game(game, controller)
        
        for i in range(max_iterations):
            # get the initial state at the beginning of the iteration
            state = tg.return_board()
            
            # get the action from the model for both agents from the board
            magma_boy_actionspace = magma_boy_model(state)
            hydro_girl_actionspace = hydro_girl_model(state)
            magma_boy_action = torch.multinomial(magma_boy_actionspace, num_samples = 1)
            hydro_girl_action = torch.multinomial(hydro_girl_actionspace, num_samples = 1)
            
            # get loss function based on the actions taken
            tg.move_players([action_dict[int(magma_boy_action)], action_dict[int(hydro_girl_action)]])
            loss = tg.loss_function()
            
            # backward pass (don't understand what this is really)
            magma_boy_optimizer.zero_grad()
            hydro_girl_optimizer.zero_grad()
            loss.backward()
            magma_boy_optimizer.step()
            hydro_girl_optimizer.step()
            
            if tg.is_ended:
                print("someone died")
                break
            
            print(loss)
            
# once training is done, save the parameters to be used in a different file
torch.save(magma_boy_model.state_dict(), 'magma_boy_params.pth')
torch.save(hydro_girl_model.state_dict(), 'hydro_girl_params.pth')