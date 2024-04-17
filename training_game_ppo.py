from game import Game
import pygame
from pygame.locals import *
from gates import Gates
from doors import FireDoor, WaterDoor
from collectibles import FireCollectible, WaterCollectible
from board import Board
from character import MagmaBoy, HydroGirl
from controller import GeneralController
import numpy as np
import torch
from neural_network_ppo import Net
import torch.nn as nn

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

            self.magma_boy_index = 0
            self.magma_boy_location = [(16, 336), (24 * 16, 15 * 16), (16, 200), (24 * 16, 3 * 16)][self.magma_boy_index]
            self.magma_boy = MagmaBoy(self.magma_boy_location)
            self.hydro_girl_index = 0
            self.hydro_girl_location = [(16, 336), (24 * 16, 15 * 16), (16, 200), (24 * 16, 3 * 16)][self.hydro_girl_index]
            self.hydro_girl = HydroGirl(self.hydro_girl_location)
            self.magma_boy_pos_hist = [self.magma_boy_location]
            self.hydro_girl_pos_hist = [self.hydro_girl_location]
            
            # arrays for collectibles
            scaling_fac = 16
            self.fire_collectibles = [FireCollectible((11.5 * scaling_fac, 21 * scaling_fac)), 
                                      FireCollectible((17.5 * scaling_fac, 15 * scaling_fac)),
                                      FireCollectible(((12 - (1/8)) * scaling_fac, 9 * scaling_fac))][self.magma_boy_index:]
            self.water_collectibles = [WaterCollectible((19.5 * scaling_fac, 21 * scaling_fac)),
                                       WaterCollectible((8.5 * scaling_fac, 15 * scaling_fac)),
                                       WaterCollectible(((24 + (3/8)) * scaling_fac, 9 * scaling_fac))][self.hydro_girl_index:]
            
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
                
        # determine distance from collectible beforehand
        distance_before_fire = self.get_closest_magma_gem()
        distance_before_water = self.get_closest_hydro_gem()
        
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
                
        # determine distance from collectible after
        distance_after_fire = self.get_closest_magma_gem()
        distance_after_water = self.get_closest_hydro_gem()

        # calculate reward, considering speed
        reward = [((num_after_fire - num_before_fire) * 10 * self.scale_reward_by_time()) - (distance_after_fire - distance_before_fire) - 1, \
            ((num_after_water - num_before_water) * 10 * self.scale_reward_by_time()) - (distance_after_water - distance_before_water) - 1]
        
        if (num_after_fire != num_before_fire or num_after_water != num_before_water):
            print("Gained collectible")
        
        # punish for death
        if self.magma_boy.is_dead():
            reward[0] -= 15
        if self.hydro_girl.is_dead():
            reward[1] -= 15

        # check for penalizing/rewarding game end conditions
        if self.game.level_is_done(self.doors):
            reward[0] += 15
            reward[1] += 15
            
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
            
        # returns each characters state (hiding others position), reward, terminated (similar to gymnasium)
        return self.return_board("Magma"), self.return_board("Hydro"), reward, self.is_ended
    
    # scale reward by time to incentive quickly obtaining rewards
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
        disp_tensor = torch.from_numpy(np.moveaxis(disp, -1, 0)).float()
        disp_tensor = disp_tensor.to(device)
        return disp_tensor.unsqueeze(0)
    
    # define the closest gem to magma boy - if all gems are collected, the door is the closest gem
    def get_closest_gem(self, player, collectibles, door):
        for collectible in collectibles:
            if not(collectible.is_collected):
                # return np.sqrt(((player.rect[0] - collectible.location[0]) ** 2) + \
                #     ((player.rect[1] - collectible.location[1]) ** 2))
                return np.abs(player.rect[0] - collectible.location[0]) # only do x location to avoid the continuous and excessive jumping? (no different in practice)
        return np.sqrt(((player.rect[0] - door.door_location[0]) ** 2) + \
                    ((player.rect[1] - door.door_location[1]) ** 2))
    
    def get_closest_magma_gem(self):
        return self.get_closest_gem(self.magma_boy, self.fire_collectibles, self.fire_door)
    
    def get_closest_hydro_gem(self):
        return self.get_closest_gem(self.hydro_girl, self.water_collectibles, self.water_door)        
    



# MODEL CLASS -------------------------------------------------------------------------------------
class Model():
    def __init__(self, actor_critic, ppo_clip_val = 0.2, target_kl_div = 0.01, max_policy_iters = 80, value_train_iters = 80, policy_lr = 1E-4):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_iters = max_policy_iters
        self.value_train_iters = value_train_iters
        self.ent_reg = 0.1
        
        # initializing policy optimizer with its respecitve learning rate
        policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
        self.policy_optim = torch.optim.Adam(policy_params, lr = policy_lr)
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optim, gamma = 0.9)
        
    def train_policy(self, state, actions, old_log_probs, gaes, returns):
        for _ in range(self.max_policy_iters):
            self.policy_optim.zero_grad()
            
            # get new logitcs from policy function of model for the according state
            new_logits = self.ac.policy(torch.stack(state).squeeze(1))
            new_logits_distrib = torch.distributions.categorical.Categorical(logits = new_logits)
            
            # determine the new log probs according to the actions
            new_log_probs = new_logits_distrib.log_prob(actions)
            
            # compare the new log probs with the old log probs to calculate the policy ratio, and clip to be within the given clip range
            r = torch.exp(new_log_probs - old_log_probs)
            clipped_r = r.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            
            # get values from model and determine the loss given returns
            values = self.ac.value(torch.stack(state).squeeze(1))
            value_loss = torch.mean((returns - values) ** 2)
            
            # calculate loss with clipped ratio, full, and take the min of the two
            clipped_l = clipped_r * gaes
            full_l = r * gaes
            loss = torch.mean(-torch.min(clipped_l, full_l)) + value_loss # return as negative to minimize
            loss -= torch.mean(self.ent_reg * torch.mean(new_logits_distrib.entropy())) # subtract entropy term
            
            # backpropogate, update policy
            loss.backward()
            self.policy_optim.step()
            
            # check to see if we have gone over kldiv - if we have, break
            kl_div = torch.mean(old_log_probs - new_log_probs)
            if kl_div > self.target_kl_div:
                break
        self.policy_scheduler.step()
        



# TRAINING ----------------------------------------------------------------------------------------
# create both AI models, playing the same game
# both models take in a board (given by tg.return_board()) and give an action from the following action space:
action_dict = {
    0: "right",
    1: "left",
    2: "jumpright",
    3: "jumpleft",
    4: "still"
}

magma_boy_model = Model(Net(len(action_dict)), ppo_clip_val = 0.2, policy_lr = 3E-4, value_lr = 1E-5)
hydro_girl_model = Model(Net(len(action_dict)), ppo_clip_val = 0.2, policy_lr = 1E-4, value_lr = 1E-6)
magma_boy_model.ac.to(device)
hydro_girl_model.ac.to(device)

# convert the rgb image to grayscale, changing number of color channels from 3 to 1
def preprocess_image(rgb_image):
    smaller_image = nn.functional.interpolate(rgb_image, size = (84, 84))
    grayscale_image = torch.sum(smaller_image * torch.tensor([0.299, 0.587, 0.114], device = device).view(1, 3, 1, 1), dim = 1)
    return grayscale_image.unsqueeze(1).to(device)

# rollout training data for magma model and hydro model
def rollout(magma_model, hydro_model, tg, max_steps = 1000):
    # initialize empty training data list, as well as magma and hydro states
    # training data follows this format:
    # - state
    # - action
    # - reward
    # - value
    # - log prob of action
    magma_training_data = [[], [], [], [], []]
    hydro_training_data = [[], [], [], [], []]
    magma_state = preprocess_image(tg.return_board("Magma"))
    hydro_state = preprocess_image(tg.return_board("Hydro"))
    
    magma_reward = 0
    hydro_reward = 0
    
    for j in range(max_steps):
        # get logits and val from model given the current state
        magma_logits, magma_val = magma_model(magma_state)
        hydro_logits, hydro_val = hydro_model(hydro_state)
        magma_val = magma_val.item()
        hydro_val = hydro_val.item()
        
        # get the categorical distribution of actions and sample from it
        magma_act_distrib = torch.distributions.categorical.Categorical(logits = magma_logits)
        hydro_act_distrib = torch.distributions.categorical.Categorical(logits = hydro_logits)
        magma_act = magma_act_distrib.sample()
        hydro_act = hydro_act_distrib.sample()
        
        # get log probs from sampled actions
        magma_act_logprob = magma_act_distrib.log_prob(magma_act).item()
        hydro_act_logprob = hydro_act_distrib.log_prob(hydro_act).item()
        
        magma_act = magma_act.item()
        hydro_act = hydro_act.item()
        
        # sample new states, rewards, and term from given action
        magma_next_state, hydro_next_state, reward, done = tg.play_step((action_dict[magma_act], action_dict[hydro_act]))
        
        # update training data with what was seen from the step just played (once every four frames)
        if j % 4 == 0:
            for i, item in enumerate((magma_state, magma_act, reward[0], magma_val, magma_act_logprob)):
                magma_training_data[i].append(item)
            for i, item in enumerate((hydro_state, hydro_act, reward[1], hydro_val, hydro_act_logprob)):
                hydro_training_data[i].append(item)
        
        # update running rewards and update previous state to most recently recorded state
        magma_reward += reward[0]
        hydro_reward += reward[1]
        magma_state = preprocess_image(magma_next_state).to(device)
        hydro_state = preprocess_image(hydro_next_state).to(device)
        
        # if the game is done, finish training
        if done:
            break
    
    # convert values to GAEs
    magma_training_data[3] = calculate_gaes(np.asarray(magma_training_data[2]), np.asarray(magma_training_data[3]))
    hydro_training_data[3] = calculate_gaes(np.asarray(hydro_training_data[2]), np.asarray(hydro_training_data[3]))
    
    return magma_training_data, hydro_training_data, magma_reward, hydro_reward

# apply a discount to all future rewards
def discount_rewards(rewards, gamma = 0.99):
    new_rewards = [float(rewards[-1])]
    
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        
    return torch.tensor(new_rewards[::-1], device = device)

# calculate GAEs (general advantage estimates) based off of rewards and values
def calculate_gaes(rewards, values, gamma = 0.9, decay = 0.99):
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    
    gaes = [deltas[-1]]
    
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return torch.tensor(gaes[::-1], device = device)

games = 1000

for i in range(games):
    # initialize a new game
    controller = GeneralController()
    game = Game()
    tg = Training_Game(game, controller)
    
    print("Game ", i)
    
    # perform rollout
    magma_training_data, hydro_training_data, magma_reward, hydro_reward = rollout(magma_boy_model.ac, hydro_girl_model.ac, tg)
    
    print("Magma reward ", magma_reward)
    print("Hydro reward ", hydro_reward)
    
    # take indeices that randomly swap around the training data
    rand_index = np.random.permutation(len(magma_training_data[0])) # shouldn't actually matter which training data model is used here
    
    # randomize states based on the above index
    magma_training_data_new = [[], [], [] ,[] ,[]]
    hydro_training_data_new = [[], [], [] ,[] ,[]]
    for j in range(5):
        for i in rand_index:
            magma_training_data_new[j].append(magma_training_data[j][i])
            hydro_training_data_new[j].append(hydro_training_data[j][i])
    
    # DATA FOR DETERMINING POLICY
    
    # record states
    magma_state = magma_training_data_new[0]
    hydro_state = hydro_training_data_new[0]
    
    # record actions
    magma_acts = torch.tensor(magma_training_data_new[1], dtype = torch.int32, device = device)
    hydro_acts = torch.tensor(hydro_training_data_new[1], dtype = torch.int32, device = device)
    
    # record GAEs
    magma_gaes = torch.tensor(magma_training_data_new[3], dtype = torch.float32, device = device)
    hydro_gaes = torch.tensor(hydro_training_data_new[3], dtype = torch.float32, device = device)
    
    # record log probs
    magma_log_probs = torch.tensor(magma_training_data_new[4], dtype = torch.float32, device = device)
    hydro_log_probs = torch.tensor(hydro_training_data_new[4], dtype = torch.float32, device = device)
    
    # DATA FOR DETERMINING VALUE
    
    # calculating GAEs (simply taking the returns and applying the discount factor to them)
    magma_returns = torch.tensor(discount_rewards(magma_training_data_new[2]), dtype = torch.float32, device = device)
    hydro_returns = torch.tensor(discount_rewards(hydro_training_data_new[2]), dtype = torch.float32, device = device)
    
    # model training
    magma_boy_model.train_policy(magma_state, magma_acts, magma_log_probs, magma_gaes, magma_returns)
    hydro_girl_model.train_policy(hydro_state, hydro_acts, hydro_log_probs, hydro_gaes, hydro_returns)
            
# once training is done, save the parameters to be used in a different file
# addresses are kept locally because i was having trouble installing the pth files to where the git dir was
torch.save(magma_boy_model.ac.state_dict(), 'temp/magma_boy_params_ppo.pth')
torch.save(hydro_girl_model.ac.state_dict(), 'temp/hydro_girl_params_ppo.pth')