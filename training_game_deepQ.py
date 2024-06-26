from game import Game
import pygame
from pygame.locals import *
from gates import Gates
from doors import FireDoor, WaterDoor
from collectibles import FireCollectible, WaterCollectible
from board import Board
#from skimage import io
from character import MagmaBoy, HydroGirl
from controller import GeneralController
import numpy as np
from collections import deque
import torch
from neural_network_deepQ import Net
import random
import torch.nn as nn

MAX_MEMORY = 100000
BATCH_SIZE = 32

if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")
print(device)
 
 
# ENVIRONMENT CLASS -------------------------------------------------------------------------------
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
            self.magma_boy_pos_hist = [self.magma_boy_location]
            self.hydro_girl_pos_hist = [self.hydro_girl_location]
            
            # arrays for collectibles
            scaling_fac = 16
            self.fire_collectibles = [FireCollectible((11.5 * scaling_fac, 21 * scaling_fac)), 
                                      FireCollectible((31 * scaling_fac, 18 * scaling_fac)),
                                      FireCollectible((17.5 * scaling_fac, 15 * scaling_fac)),
                                      FireCollectible(((12 - (1/8)) * scaling_fac, 9 * scaling_fac))]
            self.water_collectibles = [WaterCollectible((19.5 * scaling_fac, 21 * scaling_fac)),
                                       WaterCollectible((31 * scaling_fac, 18 * scaling_fac)),
                                       WaterCollectible((8.5 * scaling_fac, 15 * scaling_fac)),
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
        self.game.draw_player([self.magma_boy], "Magma")
        self.game.draw_player([self.hydro_girl], "Hydro")

        # draw collectibles
        self.game.draw_collectibles(self.fire_collectibles + self.water_collectibles)
        self.game.draw_collectibles(self.fire_collectibles, "Magma")
        self.game.draw_collectibles(self.water_collectibles, "Hydro")

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

        if self.game.level_is_done(self.doors):
            self.is_ended = True

        if self.controller.press_key(events, K_ESCAPE):
            self.is_ended = True

        # close window is player clicks on [x]
        for event in events:
            if event.type == QUIT:
                pygame.quit()
                self.is_ended = True
                self.keep_running = False
        
    # define an action space to move both players, and update the game
    # - note that the order is [magma_boy_action, hydro_girl_action]
    def play_step(self, dirs):
        # determine the number of collectibles before stepping
        num_before_fire = 0
        for collectible in self.fire_collectibles:
            if collectible.is_collected:
                num_before_fire += 1
        num_before_water = 0
        for collectible in self.water_collectibles:
            if collectible.is_collected:
                num_before_water += 1
        
        # parse both directions and players at the same time
        for dir, player in zip(dirs, [self.magma_boy, self.hydro_girl]):
            if dir == "right":
                player.moving_right = True
                player.moving_left = False
                player.jumping = False
            elif dir == "left":
                player.moving_right = False
                player.moving_left = True
                player.jumping = False
            elif dir == "jumpright":
                player.moving_right = True
                player.moving_left = False
                if player.air_timer < 6:
                    player.jumping = True
            elif dir == "jumpleft":
                player.moving_right = False
                player.moving_left = True
                if player.air_timer < 6:
                    player.jumping = True
            elif dir == "still":
                player.moving_right = False
                player.moving_left = False
                player.jumping = False
            else:
                raise(KeyError("Not a valid move!"))

        # update game with new actions
        self.update_loop()
        
        # determine the number of collectibles after stepping
        num_after_fire = 0
        for collectible in self.fire_collectibles:
            if collectible.is_collected:
                num_after_fire += 1
        num_after_water = 0
        for collectible in self.water_collectibles:
            if collectible.is_collected:
                num_after_water += 1

        # calculate reward, considering speed
        reward = [(num_after_fire - num_before_fire) * 10000 * self.scale_reward_by_time(), (num_after_water - num_before_water) * 10000 * self.scale_reward_by_time()]
        if (num_after_fire != num_before_fire or num_after_water != num_before_water):
            print("Gained collectible")
        
        # punish for death
        if self.magma_boy.is_dead():
            reward[0] -= 10000
        if self.hydro_girl.is_dead():
            reward[1] -= 10000

        # check for penalizing/rewarding game end conditions
        if self.game.level_is_done(self.doors):
            reward[0] += 2000
            reward[1] += 2000
            
        # punish characters for staying in the same place for 5 moves
        # magma_boy_curr_location = self.magma_boy.rect.x
        # if self.magma_boy_pos_hist[len(self.magma_boy_pos_hist) - 1] == magma_boy_curr_location:
        #     reward[0] -= len(self.magma_boy_pos_hist) * 100
        #     self.magma_boy_pos_hist.append(magma_boy_curr_location)
        # else:
        #     self.magma_boy_pos_hist = [magma_boy_curr_location]
            
        # hydro_girl_curr_location = self.hydro_girl.rect.x
        # if self.hydro_girl_pos_hist[len(self.hydro_girl_pos_hist) - 1] == hydro_girl_curr_location:
        #     reward[1] -= len(self.hydro_girl_pos_hist) * 100
        #     self.hydro_girl_pos_hist.append(hydro_girl_curr_location)
        # else:
        #     self.hydro_girl_pos_hist = [hydro_girl_curr_location]
        
        # reward characters for getting closer to gems
        reward[0] += (1 / self.get_closest_magma_gem()) * 10000
        reward[1] += (1 / self.get_closest_hydro_gem()) * 10000
            
        # returns each characters state (hiding others position), reward, terminated (similar to gymnasium)
        return self.return_board("Magma"), self.return_board("Hydro"), reward, self.is_ended
    
    # creating a scaling factor to incentivize getting the reward more quickly
    def scale_reward_by_time(self):
        if self.timer <= 120:
            return ((240 - self.timer) / 120)
        else:
            return 1

    # return the board as a 3d array (change this to torch tensor at some point?)
    def return_board(self, element="Both"):
        if element == "Magma":
            disp = pygame.surfarray.array3d(self.game.magma_display)
            #io.imsave('temp/magma_image.png', disp)
        elif element == "Hydro":
            disp = pygame.surfarray.array3d(self.game.hydro_display)
            #io.imsave('temp/hydro_image.png', disp)
        else:
            disp = pygame.surfarray.array3d(self.game.display)
        return (torch.from_numpy(np.moveaxis(disp, -1, 0)).float()).unsqueeze(0)
    
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
    
    


# MODEL CLASS -------------------------------------------------------------------------------------
class Model():
    # initializing model with hyperparameters, optimizer (SGD), and loss function (MSE)
    def __init__(self, model, lr, gamma):
        self.model = model
        self.memory = deque(maxlen = MAX_MEMORY)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = torch.nn.HuberLoss()

    # append SARS to the memory of the model; if the memory is at max length, replace previous memories with the new one
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # train the model on a SARS set - occurs at each frame
    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)
    
    # train the model on a minibatch taken from the model's memory
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # conver the priorities into probabilites, and sample with probabilities favoring the higher priority states
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.train_step(state, action, reward, next_state, done)
        
    # training step
    
    # the purpose of this is to calculate a set of Q(s, a) given s and the possible values of a
    # this gives us a 1x6 tensor, with each action having its own Q given the state
    # now, we compare the value of Q calculated by the model with a new value of Q, Q_new, which takes into account
    # the reward at the next possible states, as well as the Q values of all actions taken at the next state
    def train_step(self, state, action, reward, next_state, terminated):
        state = torch.tensor(state, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        
        self.optimizer.zero_grad()
        
        current_Q_vals = self.model(state) # get Q values from current state
        pred_Q = current_Q_vals.squeeze()[action] # get the Q value of the given action taken
        
        next_Q_vals = self.model(next_state) # get the Q values of the next state
        max_next_Q = torch.max(next_Q_vals).item() # determine the Q value of the best action at the next state
        target_Q = reward + (self.gamma * max_next_Q * (1 - int(terminated))) # bellman equation
        
        loss = self.criterion(pred_Q.to(device), target_Q.to(device)) # loss function compares current Q to Q from bellman
        loss.backward()
        self.optimizer.step()
        
        print(loss)
        
        

# create both AI models, playing the same game
# both models take in a board (given by tg.return_board()) and give an action from the following action space:
# - move right
# - move left
# - move up
# - unmove right
# - unmove left
# - unmove up
magma_boy_model = Model(Net(), 1E-7, 0.9)
hydro_girl_model = Model(Net(), 1E-7, 0.9)
magma_boy_model.model.to(device)
hydro_girl_model.model.to(device)

action_dict = {
    0: "right",
    1: "left",
    2: "jumpright",
    3: "jumpleft",
    4: "still"
}

games = 10000
epsilon = 1

# convert the rgb image to grayscale, changing number of color channels from 3 to 1
def preprocess_image(rgb_image):
    smaller_image = nn.functional.interpolate(rgb_image, size = (68, 50))
    grayscale_image = torch.sum(smaller_image * torch.tensor([0.299, 0.587, 0.114], device = device).view(1, 3, 1, 1), dim = 1)
    return grayscale_image.unsqueeze(1).to(device)

for i in range(games):
    # initialize a new game
    controller = GeneralController()
    game = Game()
    tg = Training_Game(game, controller)
    
    # decay epsilon based on the number of games that have been played; min epsilon is 0.1
    epsilon = np.maximum(epsilon - (1 / games), 0.1)
    
    print("Game ", i)
    
    while True:
        # get the initial state at the beginning of the iteration
        magma_state = preprocess_image(tg.return_board("Magma").to(device))
        hydro_state = preprocess_image(tg.return_board("Hydro").to(device))
        
        # get the action from the model for both agents from the board
        magma_boy_model_res = magma_boy_model.model(magma_state)
        hydro_girl_model_res = hydro_girl_model.model(hydro_state)
        
        # choose an action with an epsilon greedy strategy
        if np.random.rand() < epsilon:
            magma_boy_action = np.random.randint(len(action_dict))
            hydro_girl_action = np.random.randint(len(action_dict))
        else:
            magma_boy_action = torch.argmax(magma_boy_model_res).item()
            hydro_girl_action = torch.argmax(hydro_girl_model_res).item()
        
        # get loss function based on the actions taken
        next_state_magma, next_state_hydro, rewards, terminated = tg.play_step([action_dict[int(magma_boy_action)], action_dict[int(hydro_girl_action)]])
        next_state_magma = preprocess_image(next_state_magma.to(device))
        next_state_hydro = preprocess_image(next_state_hydro.to(device))
        
        # apply training based on the current state, action, reward achieved from that action, and next state
        magma_boy_model.train_short_memory(magma_state, magma_boy_action, rewards[0], next_state_magma, terminated)
        hydro_girl_model.train_short_memory(hydro_state, hydro_girl_action, rewards[1], next_state_hydro, terminated)
        
        # give a more diverse replay buffer?
        if np.random.randint(10) < 2:
            magma_boy_model.remember(magma_state, magma_boy_action, rewards[0], next_state_magma, terminated)
            hydro_girl_model.remember(hydro_state, hydro_girl_action, rewards[1], next_state_hydro, terminated)
            
        if tg.is_ended:
            magma_boy_model.train_long_memory()
            hydro_girl_model.train_long_memory()
            break
            
# once training is done, save the parameters to be used in a different file
# addresses are kept locally because i was having trouble installing the pth files to where the git dir was
torch.save(magma_boy_model.model.state_dict(), 'temp/magma_boy_params_deepQ.pth')
torch.save(hydro_girl_model.model.state_dict(), 'temp/hydro_girl_params_deepQ.pth')