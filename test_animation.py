import pygame
import time
from render.visualize import FaceVisualizer
from render.animation.controller import AnimationController

def main():
    visualizer = FaceVisualizer(800, 600)
    controller = AnimationController()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    controller.set_expression('happy')
                elif event.key == pygame.K_RETURN:
                    controller.set_expression('surprised')
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        values = controller.update()
        visualizer.draw_face(values)
        
        pygame.time.wait(16)  # 约60FPS
    
    pygame.quit()

if __name__ == "__main__":
    main() 