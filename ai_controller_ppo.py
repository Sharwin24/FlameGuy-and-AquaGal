from controller import Controller
import torch
from neural_network_ppo import Net
import pygame
import numpy as np
import torch.nn as nn

if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")
print(device)

def preprocess_image(rgb_image):
    #smaller_image = nn.functional.interpolate(rgb_image, size = (68 * 2, 50 * 2))
    smaller_image = nn.functional.interpolate(rgb_image, size = (84, 84))
    grayscale_image = torch.sum(smaller_image * torch.tensor([0.299, 0.587, 0.114], device = device).view(1, 3, 1, 1), dim = 1)
    return grayscale_image.unsqueeze(1).to(device)

# controller that takes in a gamestate and gives outputs based off of that    
class AIController(Controller):
    def __init__(self, dir, element):
        # initialize and load trained NN, make sure its in eval mode
        self.net = Net(5)
        self.net.load_state_dict(torch.load(dir))
        self.net.eval()
        self.net.to(device)
        self.element = element
        
        # actionspace
        self.action_dict = {
            0: "right",
            1: "left",
            2: "jumpright",
            3: "jumpleft",
            4: "still"
        }
        
    def control_player(self, events, player, game):
        if self.element == "Magma":
            disp = pygame.surfarray.array3d(game.magma_display)
            #io.imsave('temp/magma_image.png', disp)
        elif self.element == "Hydro":
            disp = pygame.surfarray.array3d(game.hydro_display)
            #io.imsave('temp/hydro_image.png', disp)
        else:
            disp = pygame.surfarray.array3d(game.display)
        disp = torch.from_numpy(np.moveaxis(disp, -1, 0)).float()
        disp = disp.to(device)
        disp = preprocess_image(disp.unsqueeze(0))
        move = torch.argmax(self.net(disp)[0]).item()
        move_name = self.action_dict[int(move)]
        
        if move_name == "right":
            player.moving_right = True
            player.moving_left = False
            player.jumping = False
        elif move_name == "left":
            player.moving_right = False
            player.moving_left = True
            player.jumping = False
        elif move_name == "jumpright":
            player.moving_right = True
            player.moving_left = False
            if player.air_timer < 6:
                player.jumping = True
        elif move_name == "jumpleft":
            player.moving_right = False
            player.moving_left = True
            if player.air_timer < 6:
                player.jumping = True
        elif move_name == "still":
            player.moving_right = False
            player.moving_left = False
            player.jumping = False
        else:
            raise(KeyError("Not a valid move!"))
