"""
Reusable UI components for rendering.
"""

import pygame
from typing import List, Tuple, Optional


def render_text_panel(
    surface: pygame.Surface,
    texts: List[str],
    position: Tuple[int, int],
    font: pygame.font.Font,
    title_font: Optional[pygame.font.Font] = None,
    bg_color: Tuple[int, int, int] = (40, 40, 40),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = 10,
    line_spacing: int = 25,
    alpha: int = 200,
):
    """
    Render a text panel with semi-transparent background.

    Args:
        surface: Surface to render on
        texts: List of text lines
        position: (x, y) position for top-left corner
        font: Font for regular text
        title_font: Font for first line (title), None to use same font
        bg_color: Background color RGB
        text_color: Text color RGB
        padding: Padding around text
        line_spacing: Spacing between lines
        alpha: Background transparency (0-255)
    """
    if not texts:
        return

    # Calculate panel size
    panel_width = 0
    panel_height = len(texts) * line_spacing + padding * 2

    for i, text in enumerate(texts):
        current_font = title_font if (i == 0 and title_font) else font
        text_surface = current_font.render(text, True, text_color)
        panel_width = max(panel_width, text_surface.get_width())

    panel_width += padding * 2

    # Create semi-transparent background
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(alpha)
    panel_surface.fill(bg_color)
    surface.blit(panel_surface, position)

    # Render text
    x, y = position
    for i, text in enumerate(texts):
        current_font = title_font if (i == 0 and title_font) else font
        text_surface = current_font.render(text, True, text_color)
        surface.blit(text_surface, (x + padding, y + padding + i * line_spacing))


def render_progress_bar(
    surface: pygame.Surface,
    position: Tuple[int, int],
    width: int,
    height: int,
    progress: float,
    bg_color: Tuple[int, int, int] = (60, 60, 60),
    fill_color: Tuple[int, int, int] = (76, 175, 80),
    border_color: Tuple[int, int, int] = (100, 100, 100),
    border_width: int = 2,
):
    """
    Render a progress bar.

    Args:
        surface: Surface to render on
        position: (x, y) top-left position
        width: Bar width
        height: Bar height
        progress: Progress value (0.0 - 1.0)
        bg_color: Background color
        fill_color: Fill color
        border_color: Border color
        border_width: Border width
    """
    x, y = position
    progress = max(0.0, min(1.0, progress))

    # Draw background
    pygame.draw.rect(surface, bg_color, (x, y, width, height))

    # Draw fill
    fill_width = int(width * progress)
    if fill_width > 0:
        pygame.draw.rect(surface, fill_color, (x, y, fill_width, height))

    # Draw border
    pygame.draw.rect(surface, border_color, (x, y, width, height), border_width)


def render_comparison_stats(
    surface: pygame.Surface,
    position: Tuple[int, int],
    agent1_name: str,
    agent2_name: str,
    stat_name: str,
    stat1_value: float,
    stat2_value: float,
    font: pygame.font.Font,
    better_is_higher: bool = True,
):
    """
    Render comparison statistics with winner highlighting.

    Args:
        surface: Surface to render on
        position: (x, y) position
        agent1_name: Name of first agent
        agent2_name: Name of second agent
        stat_name: Name of statistic
        stat1_value: Value for agent 1
        stat2_value: Value for agent 2
        font: Font to use
        better_is_higher: True if higher value is better
    """
    x, y = position

    # Determine winner
    if better_is_higher:
        agent1_better = stat1_value > stat2_value
    else:
        agent1_better = stat1_value < stat2_value

    # Colors
    winner_color = (255, 215, 0)  # Gold
    loser_color = (150, 150, 150)  # Gray
    neutral_color = (255, 255, 255)  # White

    # Render stat name
    name_surface = font.render(f"{stat_name}:", True, neutral_color)
    surface.blit(name_surface, (x, y))

    # Render values
    agent1_color = winner_color if agent1_better else loser_color
    agent2_color = loser_color if agent1_better else winner_color

    # Format values
    if isinstance(stat1_value, int):
        value1_text = f"{agent1_name}: {stat1_value}"
        value2_text = f"{agent2_name}: {stat2_value}"
    else:
        value1_text = f"{agent1_name}: {stat1_value:.2f}"
        value2_text = f"{agent2_name}: {stat2_value:.2f}"

    value1_surface = font.render(value1_text, True, agent1_color)
    value2_surface = font.render(value2_text, True, agent2_color)

    surface.blit(value1_surface, (x + 150, y))
    surface.blit(value2_surface, (x + 350, y))


def render_help_text(
    surface: pygame.Surface,
    position: Tuple[int, int],
    font: pygame.font.Font,
    show_full: bool = True,
):
    """
    Render help text with controls.

    Args:
        surface: Surface to render on
        position: (x, y) top-left position
        font: Font to use
        show_full: Show full help or abbreviated
    """
    if show_full:
        help_texts = [
            "Controls:",
            "  [Mouse Wheel] Zoom",
            "  [W/A/S/D] Pan Camera",
            "  [Tab] Switch Active Camera",
            "  [1] Focus Left / [2] Focus Right / [3] Both",
            "  [F] Toggle Follow Mode",
            "  [R] Reset Camera",
            "  [=] Sync Cameras",
            "  [Cmd+F / F11] Fullscreen",
            "  [ESC] Quit",
        ]
    else:
        help_texts = [
            "Controls: [Tab] Switch Camera | [Cmd+F] Fullscreen | [ESC] Quit",
        ]

    render_text_panel(
        surface,
        help_texts,
        position,
        font,
        bg_color=(20, 20, 20),
        alpha=180,
    )


def render_metric_comparison(
    surface: pygame.Surface,
    position: Tuple[int, int],
    width: int,
    agent1_value: float,
    agent2_value: float,
    agent1_color: Tuple[int, int, int],
    agent2_color: Tuple[int, int, int],
    label: str,
    font: pygame.font.Font,
):
    """
    Render a visual comparison of two metrics with bars.

    Args:
        surface: Surface to render on
        position: (x, y) position
        width: Total width for comparison
        agent1_value: Value for agent 1
        agent2_value: Value for agent 2
        agent1_color: Color for agent 1
        agent2_color: Color for agent 2
        label: Label for metric
        font: Font to use
    """
    x, y = position
    bar_height = 20

    # Render label
    label_surface = font.render(label, True, (255, 255, 255))
    surface.blit(label_surface, (x, y))

    # Calculate bar widths
    max_value = max(agent1_value, agent2_value, 1.0)  # Avoid division by zero
    bar1_width = int((agent1_value / max_value) * width)
    bar2_width = int((agent2_value / max_value) * width)

    # Render bars
    y_bar = y + 25

    # Agent 1 bar
    if bar1_width > 0:
        pygame.draw.rect(surface, agent1_color, (x, y_bar, bar1_width, bar_height))

    # Agent 2 bar
    if bar2_width > 0:
        pygame.draw.rect(surface, agent2_color, (x, y_bar + bar_height + 5, bar2_width, bar_height))

    # Render values
    value1_text = font.render(f"{agent1_value:.1f}", True, agent1_color)
    value2_text = font.render(f"{agent2_value:.1f}", True, agent2_color)

    surface.blit(value1_text, (x + bar1_width + 5, y_bar))
    surface.blit(value2_text, (x + bar2_width + 5, y_bar + bar_height + 5))
