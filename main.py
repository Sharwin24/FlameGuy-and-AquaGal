"""
Main file for Magma Boy and Hydro Girl game.
"""

# import pygame and orther needed libraries
import sys
import pygame
from pygame.locals import *

# import classes
from game import Game
from board import Board
from character import MagmaBoy, HydroGirl
from controller import ArrowsController, WASDController, GeneralController
from ai_controller_ppo import AIController
from gates import Gates
from doors import FireDoor, WaterDoor
from level_select import LevelSelect
from collectibles import FireCollectible, WaterCollectible


def main():
    if len(sys.argv) > 1 and (sys.argv[1] == "Both" or sys.argv[1] == "Magma" or sys.argv[1] == "Hydro"):
        mode = sys.argv[1]
        print("Starting in ", mode, " mode.\n")
    else:
        mode = "None"
        print("No proper mode detected; running game with no AI agents.\n")
    pygame.init()
    controller = GeneralController()
    game = Game()
    show_intro_screen(game, controller, mode)

def show_intro_screen(game, controller, mode):
    intro_screen = pygame.image.load('data/screens/intro_screen.png')
    game.display.blit(intro_screen, (0, 0))
    while True:
        game.refresh_window()
        if controller.press_key(pygame.event.get(), K_RETURN):
            show_level_screen(game, controller, mode)


def show_level_screen(game, controller, mode):
    level_select = LevelSelect()
    level = game.user_select_level(level_select, controller)
    run_game(game, controller, mode, level)


def show_win_screen(game, controller, mode):
    win_screen = pygame.image.load('data/screens/win_screen.png')
    win_screen.set_colorkey((255, 0, 255))
    game.display.blit(win_screen, (0, 0))

    while True:
        game.refresh_window()
        if controller.press_key(pygame.event.get(), K_RETURN):
            show_level_screen(game, controller, mode)


def show_death_screen(game, controller, mode, level):
    death_screen = pygame.image.load('data/screens/death_screen.png')
    death_screen.set_colorkey((255, 0, 255))
    game.display.blit(death_screen, (0, 0))
    while True:
        game.refresh_window()
        events = pygame.event.get()
        if controller.press_key(events, K_RETURN):
            run_game(game, controller, mode, level)
        if controller.press_key(events, K_ESCAPE):
            show_level_screen(game, controller, mode)


def run_game(game, controller, mode, level="level1"):
    # load level data
    if level == "level1":
        board = Board('data/level1.txt')
        gate_location = (285, 128)
        plate_locations = [(190, 168), (390, 168)]
        gate = Gates(gate_location, plate_locations)
        gates = [gate]

        fire_door_location = (64, 48)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (128, 48)
        water_door = WaterDoor(water_door_location)
        doors = [fire_door, water_door]

        magma_boy_location = (16, 336)
        magma_boy = MagmaBoy(magma_boy_location)
        hydro_girl_location = (35, 336)
        hydro_girl = HydroGirl(hydro_girl_location)
        
        # arrays for collectibles
        scaling_fac = 16
        fire_collectibles = [FireCollectible((11.5 * scaling_fac, 21 * scaling_fac)), 
                             FireCollectible((8.5 * scaling_fac, 15 * scaling_fac)),
                             FireCollectible(((12 - (1/8)) * scaling_fac, 9 * scaling_fac))]
        water_collectibles = [WaterCollectible((19.5 * scaling_fac, 21 * scaling_fac)),
                              WaterCollectible((17.5 * scaling_fac, 15 * scaling_fac)),
                              WaterCollectible(((24 + (3/8)) * scaling_fac, 9 * scaling_fac))]

    if level == "level2":
        board = Board('data/level2.txt')
        gates = []

        fire_door_location = (390, 48)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (330, 48)
        water_door = WaterDoor(water_door_location)
        doors = [fire_door, water_door]

        magma_boy_location = (16, 336)
        magma_boy = MagmaBoy(magma_boy_location)
        hydro_girl_location = (35, 336)
        hydro_girl = HydroGirl(hydro_girl_location)
        
        fire_collectibles = []
        water_collectibles = []

    if level == "level3":
        board = Board('data/level3.txt')
        gates = []

        fire_door_location = (5 * 16, 4 * 16)
        fire_door = FireDoor(fire_door_location)
        water_door_location = (28 * 16, 4 * 16)
        water_door = WaterDoor(water_door_location)
        doors = [fire_door, water_door]

        magma_boy_location = (28 * 16, 4 * 16)
        magma_boy = MagmaBoy(magma_boy_location)
        hydro_girl_location = (5 * 16, 4 * 16)
        hydro_girl = HydroGirl(hydro_girl_location)
        
        fire_collectibles = []
        water_collectibles = []


    # initialize needed classes
    
    magma_file = 'temp/magma_boy_params_ppo.pth'
    hydro_file = 'temp/hydro_girl_params_ppo.pth'

    if mode == "Both":
        arrows_controller = AIController(magma_file, "Magma")
        wasd_controller = AIController(hydro_file, "Hydro")
    elif mode == "Magma":
        arrows_controller = AIController(magma_file, "Magma")
        wasd_controller = WASDController()
    elif mode == "Hydro":
        arrows_controller = ArrowsController()
        wasd_controller = AIController(hydro_file, "Hydro")
    else:
        arrows_controller = ArrowsController()
        wasd_controller = WASDController()

    clock = pygame.time.Clock()

    # main game loop
    while True:
        # pygame management
        clock.tick(60)
        events = pygame.event.get()

        # draw features of level
        game.draw_level_background(board)
        game.draw_board(board)
        if gates:
            game.draw_gates(gates)
        game.draw_doors(doors)

        # draw player to main screen and then to any needed for AI
        game.draw_player([magma_boy, hydro_girl])
        game.draw_player([magma_boy], "Magma")
        game.draw_player([hydro_girl], "Hydro")
        
        # draw collectibles
        game.draw_collectibles(fire_collectibles + water_collectibles)
        game.draw_collectibles(fire_collectibles, "Magma")
        game.draw_collectibles(water_collectibles, "Hydro")

        # move player
        arrows_controller.control_player(events, magma_boy, game)
        wasd_controller.control_player(events, hydro_girl, game)

        game.move_player(board, gates, [magma_boy, hydro_girl])

        # check for player at special location
        game.check_for_death(board, [magma_boy, hydro_girl])

        game.check_for_gate_press(gates, [magma_boy, hydro_girl])

        game.check_for_door_open(fire_door, magma_boy)
        game.check_for_door_open(water_door, hydro_girl)
        
        # checking to see if collectibles were hit
        for fire_collectible in fire_collectibles:
            game.check_for_collectible_hit(fire_collectible, magma_boy)
            
        for water_collectible in water_collectibles:
            game.check_for_collectible_hit(water_collectible, hydro_girl)

        # refresh window
        game.refresh_window()

        # special events
        if hydro_girl.is_dead() or magma_boy.is_dead():
            show_death_screen(game, controller, mode, level)

        if game.level_is_done(doors):
            show_win_screen(game, controller, mode)

        if controller.press_key(events, K_ESCAPE):
            show_level_screen(game, controller, mode)

        # close window is player clicks on [x]
        for event in events:
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

if __name__ == '__main__':
    main()