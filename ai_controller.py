from controller import Controller
import torch
from neural_network import Net
import pygame
import numpy as np
    
# controller that takes in a gamestate and gives outputs based off of that    
class AIController(Controller):
    def __init__(self, dir):
        # initialize and load trained NN, make sure its in eval mode
        self.net = Net()
        self.net.load_state_dict(torch.load(dir))
        self.net.eval()
        
        # actionspace
        self.action_dict = {
            0: "right",
            1: "notright",
            2: "left",
            3: "notleft",
            4: "jump",
            5: "notjump"
        }
        
    def control_player(self, events, player, game):
        disp = pygame.surfarray.array3d(game.display)
        board_tensor = torch.from_numpy(np.moveaxis(disp, -1, 0)).float()
        print(self.net(board_tensor))
        move = torch.argmax(self.net(board_tensor)).item()
        move_name = self.action_dict[int(move)]
        
        if move_name == "right":
            player.moving_right = True
        elif move_name == "notright":
            player.moving_right = False
        elif move_name == "left":
            player.moving_left = True
        elif move_name == "notleft":
            player.moving_left = False
        elif move_name == "jump":
            if player.air_timer < 6:
                player.jumping = True
        elif move_name == "notjump":
            player.jumping = False