import pygame
import config

def draw_time_balls(
    ball_state,
    screen_width,
    screen_height,
    ball_radius=None,
    mode="single",
    indicator_color=None,
    single_pos=(0.50, 0.28),   # (x_ratio, y_ratio) for the single indicator
    stack_pos=(0.12, 0.45),    # (x_ratio, y_ratio) for the top ball in the stack (left side)
    stack_spacing_ratio=0.08   # vertical spacing between stacked balls (as ratio of screen height)
):
    """
    Draw a time indicator ball with 4 possible states:
      - 0 = Empty (Outlined Ball)
      - 1 = White Ball (Baseline/Neutral)
      - 2 = Red Ball (Motor Imagery)
      - 3 = Blue Ball (Rest)

    Key design choice:
    - All geometry is defined using screen ratios so it stays consistent across resolutions/monitors.

    Args:
        ball_state (int): 0..3
        screen_width (int): Current screen width in pixels
        screen_height (int): Current screen height in pixels
        ball_radius (int|None): If None, radius auto-scales with screen size
        mode (str): "single" or "stack"
        indicator_color (tuple|None): If provided, overrides the fill color of the single ball
        single_pos (tuple): (x_ratio, y_ratio) anchor for single ball
        stack_pos (tuple): (x_ratio, y_ratio) anchor for stacked balls (top ball)
        stack_spacing_ratio (float): vertical spacing between stacked balls
    """

    # --- Auto-scale radius if not provided ---
    # This keeps the ball visually consistent across 1080p/4K, etc.
    if ball_radius is None:
        ball_radius = int(min(screen_width, screen_height) * 0.035)  # ~3.5% of min dimension

    # --- Define colors for each state ---
    color_map = {
        1: (255, 255, 255),  # White (Baseline)
        2: (255, 0, 0),      # Red (MI)
        3: (0, 120, 255)     # Blue (Rest) - slightly nicer blue than pure (0,0,255)
    }

    # Default color from state
    ball_color = color_map.get(ball_state, (255, 255, 255))

    # If user wants a custom indicator (e.g. MI/REST preview), override only for FILLED ball
    if indicator_color is not None and ball_state != 0:
        ball_color = indicator_color

    surf = pygame.display.get_surface()

    if mode == "single":
        # --- Single ball anchored by screen ratios ---
        ball_x = int(screen_width * single_pos[0])
        ball_y = int(screen_height * single_pos[1])

        if ball_state == 0:
            # Outlined ball (empty)
            pygame.draw.circle(surf, (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
        else:
            # Filled ball
            pygame.draw.circle(surf, ball_color, (ball_x, ball_y), ball_radius)

    elif mode == "stack":
        # --- Three stacked balls for countdown (anchored by ratios) ---
        stack_x = int(screen_width * stack_pos[0])
        stack_y_start = int(screen_height * stack_pos[1])
        spacing = int(screen_height * stack_spacing_ratio)

        for i in range(3):
            ball_y = stack_y_start + i * spacing
            if ball_state == 0:
                pygame.draw.circle(surf, (255, 255, 255), (stack_x, ball_y), ball_radius, 2)
            else:
                pygame.draw.circle(surf, ball_color, (stack_x, ball_y), ball_radius)

def draw_arrow_fill(progress, screen_width, screen_height, show_threshold=True):
    ball_radius = 200  # Base measurement
    bar_width, bar_length = ball_radius * 2, ball_radius * 2
    offset = ball_radius * 2
    if config.ARM_SIDE == "Left":
        bar_x = screen_width // 2 - offset
    else:
        bar_x = screen_width // 2 + offset
    
    bar_y = screen_height // 2


    bar_outline = [
        (bar_x - bar_length // 2, bar_y - bar_width // 2),
        (bar_x + bar_length // 2, bar_y - bar_width // 2),
        (bar_x + bar_length // 2, bar_y + bar_width // 2),
        (bar_x - bar_length // 2, bar_y + bar_width // 2),
    ]
    pygame.draw.polygon(pygame.display.get_surface(), (255, 255, 255), bar_outline, 2)

    # Calculate fill length
    fill_length = int(progress * bar_length)

    filled_rect = pygame.Rect(
        bar_x - bar_length // 2, bar_y - bar_width // 2,
        fill_length, bar_width
    )
    pygame.draw.rect(pygame.display.get_surface(), (255, 0, 0), filled_rect)

    # Draw success threshold line if enabled
    if show_threshold:
        # Scale accuracy threshold within the shape boundaries
        scaled_threshold = (config.THRESHOLD_MI - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
        scaled_threshold = max(0, min(1, scaled_threshold))  # Keep within [0,1] range

        # Compute threshold bar position using scaled threshold
        threshold_x = bar_x - bar_length // 2 + int(scaled_threshold * bar_length)

        for i in range(0, bar_width, 10):
            pygame.draw.line(
                pygame.display.get_surface(), (255, 0, 0),
                (threshold_x, bar_y - bar_width // 2 + i),
                (threshold_x, bar_y - bar_width // 2 + i + 5), 2
            )


def draw_ball_fill(progress, screen_width, screen_height, show_threshold=True):
    ball_radius = 200
    offset = ball_radius * 2
    if config.ARM_SIDE == "Left":
        ball_x = screen_width // 2 + offset
    else:
        ball_x = screen_width // 2 - offset
    ball_y = screen_height // 2

    pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    fill_height = int(progress * ball_radius * 2)
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (0, 120, 255, 180), water_rect)
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.display.get_surface().blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

    # Draw success threshold line if enabled
    if show_threshold:
        # Scale accuracy threshold within the shape boundaries
        scaled_threshold = (config.THRESHOLD_REST - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
        scaled_threshold = max(0, min(1, scaled_threshold))  # Keep within [0,1] range

        # Compute threshold position using scaled threshold
        threshold_y = ball_y + ball_radius - int(scaled_threshold * (ball_radius * 2))

        for i in range(0, ball_radius * 2, 10):
            pygame.draw.line(
                pygame.display.get_surface(), (0, 120, 255), 
                (ball_x - ball_radius + i, threshold_y), 
                (ball_x - ball_radius + i + 5, threshold_y), 2)

def draw_fixation_cross(screen_width, screen_height):
    cross_length = 40
    line_thickness = 6
    fixation_x, fixation_y = screen_width // 2, screen_height // 2

    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), 
                     (fixation_x, fixation_y - cross_length), 
                     (fixation_x, fixation_y + cross_length), 
                     line_thickness)

    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), 
                     (fixation_x - cross_length, fixation_y), 
                     (fixation_x + cross_length, fixation_y), 
                     line_thickness)



def draw_progress_bar(progress, screen_width, screen_height, color=config.green, height_ratio=0.05):
    """
    Draws a horizontal progress bar at the bottom of the screen.

    Args:
        progress (float): Value between 0.0 and 1.0 representing completion.
        screen_width (int): Width of the display.
        screen_height (int): Height of the display.
        color (tuple): RGB color of the fill.
        height_ratio (float): Bar height as a fraction of screen height.
    """
    progress = max(0.0, min(1.0, progress))  # clamp

    # Bar geometry
    bar_width = int(screen_width * 0.6)
    bar_height = int(screen_height * height_ratio)
    bar_x = (screen_width - bar_width) // 2
    bar_y = int(screen_height * 0.8)  # 80% down the screen

    # Border
    pygame.draw.rect(
        pygame.display.get_surface(), config.white,
        (bar_x, bar_y, bar_width, bar_height), width=2
    )

    # Fill
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        pygame.draw.rect(
            pygame.display.get_surface(), color,
            (bar_x + 2, bar_y + 2, max(0, fill_width - 4), bar_height - 4)
        )




'''
def draw_arrow_fill(progress, screen_width, screen_height):
    arrow_width, arrow_length, tip_length = 80, 200, 40
    arrow_x, arrow_y = (6 * screen_width) // 8, screen_height // 2

    arrow_outline = [
        (arrow_x - arrow_length // 2, arrow_y - arrow_width // 2),
        (arrow_x + arrow_length // 2 - tip_length, arrow_y - arrow_width // 2),
        (arrow_x + arrow_length // 2, arrow_y),
        (arrow_x + arrow_length // 2 - tip_length, arrow_y + arrow_width // 2),
        (arrow_x - arrow_length // 2, arrow_y + arrow_width // 2),
    ]
    pygame.draw.polygon(pygame.display.get_surface(), (255, 255, 255), arrow_outline, 2)
    filled_rect = pygame.Rect(
        arrow_x - arrow_length // 2, arrow_y - arrow_width // 2,
        int(progress * (arrow_length - tip_length)),
        arrow_width
    )
    pygame.draw.rect(pygame.display.get_surface(), (255, 0, 0), filled_rect)

def draw_ball_fill(progress, screen_width, screen_height):
    ball_radius = 100
    ball_x, ball_y = screen_width // 2, screen_height // 4

    pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    fill_height = int(progress * ball_radius * 2)
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (0, 0, 255, 180), water_rect)
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.display.get_surface().blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

def draw_fixation_cross(screen_width, screen_height):
    cross_length = 40
    line_thickness = 6
    fixation_x, fixation_y = screen_width // 2, screen_height // 2
    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), (fixation_x, fixation_y - cross_length), (fixation_x, fixation_y + cross_length), line_thickness)
    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), (fixation_x - cross_length, fixation_y), (fixation_x + cross_length, fixation_y), line_thickness)
'''