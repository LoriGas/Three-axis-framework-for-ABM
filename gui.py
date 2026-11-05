"""Pygame interactive visualisation of the simulation."""

import math
from typing import Dict, List, Optional, Tuple

from config import (
    LLM_VARIANT_TAGS,
    MLP_OFFLINE_INIT,
    MLP_ONLINE_LEARNING,
    MLP_VARIANT_TAGS,
    RULE_VARIANT_TAGS,
    VALID_SCENARIOS,
    WEIGHTS_DIR,
    ModelParams,
    get_params_for_scenario,
)
from model import World
from simulation import _normalize_weight_tag, _resolve_rb_variant, load_weights

# ---------------------------------------------------------------------------
# Graphic constants
# ---------------------------------------------------------------------------
FPS_DEFAULT: int = 10
FPS_MIN: int = 1
FPS_MAX: int = 120

BG_COLOR = (10, 14, 22)
GRID_COLOR = (31, 42, 56)
HUD_BG = (17, 23, 34)
HUD_TEXT = (232, 238, 247)
HUD_ACCENT = (92, 207, 255)
HUD_DIM = (136, 151, 174)
HUD_SUCCESS = (92, 218, 157)
HUD_WARNING = (255, 194, 92)
CARD_BG = (23, 31, 45)
CARD_BORDER = (53, 69, 92)

AGENT_MLP = (90, 130, 255)
AGENT_RULE = (255, 140, 60)
AGENT_RANDOM = (180, 180, 180)
AGENT_LLM = (80, 210, 170)
AGENT_BORDER_MLP = (60, 90, 200)
AGENT_BORDER_RULE = (200, 100, 30)
AGENT_BORDER_RANDOM = (120, 120, 120)
AGENT_BORDER_LLM = (35, 150, 120)

FOOD_EMPTY = (14, 22, 29)
FOOD_LOW = (15, 48, 45)
FOOD_HIGH = (45, 214, 145)

PANEL_WIDTH: int = 340
MIN_WIN_HEIGHT: int = 560
CELL_SIZE: int = 36


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _draw_bar(surface, x, y, w, h, fraction, color, bg=(40, 40, 50)) -> None:
    import pygame
    pygame.draw.rect(surface, bg, (x, y, w, h))
    fill_w = max(0, int(w * max(0.0, min(1.0, fraction))))
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x, y, fill_w, h))
    pygame.draw.rect(surface, (80, 80, 100), (x, y, w, h), 1)


def _draw_card(surface, rect, fill, border) -> None:
    import pygame
    pygame.draw.rect(surface, fill, rect, border_radius=10)
    pygame.draw.rect(surface, border, rect, 1, border_radius=10)


def _draw_pill(surface, rect, text: str, font, fill, text_color) -> None:
    import pygame
    pygame.draw.rect(surface, fill, rect, border_radius=rect.height // 2)
    label = font.render(text, True, text_color)
    surface.blit(label, (
        rect.centerx - label.get_width() // 2,
        rect.centery - label.get_height() // 2,
    ))


def _draw_button(surface, rect, text: str, font, mouse_pos, active: bool = True) -> None:
    import pygame
    if not active:
        fill, border, tc = (44, 44, 56), (65, 65, 78), (95, 95, 108)
    elif rect.collidepoint(mouse_pos):
        fill, border, tc = (54, 104, 142), (92, 180, 225), (250, 252, 255)
    else:
        fill, border, tc = (39, 51, 70), (72, 93, 124), (225, 233, 244)
    pygame.draw.rect(surface, fill, rect, border_radius=7)
    pygame.draw.rect(surface, border, rect, 1, border_radius=7)
    label = font.render(text, True, tc)
    surface.blit(label, (
        rect.x + (rect.width - label.get_width()) // 2,
        rect.y + (rect.height - label.get_height()) // 2,
    ))


def _cycle_option(options: List[str], current: str, delta: int) -> str:
    if not options:
        return current
    idx = options.index(current) if current in options else 0
    return options[(idx + delta) % len(options)]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Layout and GUI state helpers
# ---------------------------------------------------------------------------

def _compute_gui_layout(screen_w: int, screen_h: int, params: ModelParams) -> Tuple[int, int, int, int, int]:
    cols, rows = params.grid_width, params.grid_height
    panel_target = max(300, int(screen_w * 0.30))
    panel_w = int(_clamp(panel_target, 300, 520))
    dim_cella = int(max(6, min((screen_w - panel_w) // max(1, cols), screen_h // max(1, rows))))
    grid_w = cols * dim_cella
    panel_w = max(260, screen_w - grid_w)
    while panel_w < 260 and dim_cella > 6:
        dim_cella -= 1
        grid_w = cols * dim_cella
        panel_w = screen_w - grid_w
    return dim_cella, max(260, panel_w), grid_w, rows * dim_cella, max(screen_h, MIN_WIN_HEIGHT)


def _default_runtime_overrides(scenario: str) -> Dict[str, float]:
    p = get_params_for_scenario(scenario)
    return {
        "food_regen":             p.food_regen,
        "base_metabolism":        p.base_metabolism,
        "initial_agents":         float(p.initial_agents),
        "initial_alpha":          p.initial_alpha,
        "initial_beta":           p.initial_beta,
        "obs_noise_sigma":        p.obs_noise_sigma,
    }


def _clone_state(state: Dict) -> Dict:
    cloned = dict(state)
    cloned["overrides"] = dict(state["overrides"])
    return cloned


def _runtime_agent_options() -> List[str]:
    return [f"rule::{tag}" for tag in RULE_VARIANT_TAGS] + MLP_VARIANT_TAGS + LLM_VARIANT_TAGS + ["random"]


def _state_to_agent_option(state: Dict) -> str:
    return f"rule::{state['rule_variant']}" if state["mode"] == "rule" else state["mode"]


def _apply_agent_option(state: Dict, option: str) -> None:
    if option.startswith("rule::"):
        state["mode"] = "rule"
        state["rule_variant"] = option.split("::", 1)[1]
    else:
        state["mode"] = option


def _short_mode_label(mode: str, rule_variant: str) -> str:
    if mode == "random":
        return "random"
    if mode == "llm":
        return "llm"
    if mode == "rule":
        return f"rule/{rule_variant.replace('rb_', '')}"
    return mode.replace("mlp_", "mlp/")


def _build_world(state: Dict) -> Tuple[World, str]:
    scenario = state["scenario"]
    mode = state["mode"]
    n_layers = int(state["n_layers"])
    rule_variant = state["rule_variant"]
    weight_tag = state["weight_tag"]

    params = get_params_for_scenario(scenario)
    params.net_hidden_layers = n_layers
    for key, value in state["overrides"].items():
        if hasattr(params, key):
            setattr(params, key, int(value) if key == "initial_agents" else value)

    prop_fixed, beta_fixed, mutate_prop, random_offspring, propensity_learning = None, None, True, False, False
    net_template = None

    if mode == "rule":
        prop_fixed, beta_fixed, mutate_prop, random_offspring, propensity_learning = _resolve_rb_variant(rule_variant)
        if prop_fixed is not None:
            params.initial_alpha = float(prop_fixed)
        if beta_fixed is not None:
            params.initial_beta = float(beta_fixed)
    if mode in MLP_OFFLINE_INIT:
        effective_weight_tag = weight_tag
        if params.obs_noise_sigma > 0 and not effective_weight_tag.endswith("_uncertainty"):
            uncertainty_tag = f"{effective_weight_tag}_uncertainty"
            uncertainty_path = WEIGHTS_DIR / f"weights_{scenario}_L{n_layers}_{uncertainty_tag}.pth"
            if uncertainty_path.exists():
                effective_weight_tag = uncertainty_tag
        net_template = load_weights(scenario, n_layers, effective_weight_tag)

    world = World(
        params,
        agent_type=mode,
        net_template=net_template,
        mutate_propensity=mutate_prop,
        random_offspring_propensity=random_offspring,
        propensity_learning=propensity_learning,
        net_learning=(mode in MLP_ONLINE_LEARNING),
    )
    return world, scenario.upper()


# ---------------------------------------------------------------------------
# Runtime controls panel
# ---------------------------------------------------------------------------

def _draw_runtime_controls(
    screen, fonts: dict, panel_x: int, panel_w: int, win_h: int,
    draft: Dict, active: Dict, paused: bool, mouse_pos: tuple,
) -> List[Tuple[object, str]]:
    import pygame

    clickable: List[Tuple[object, str]] = []
    small, tiny, title = fonts["small"], fonts["tiny"], fonts["main"]

    row_count = 8 + (2 if draft["mode"] == "rule" else 0)
    card_h = min(370, max(300, 92 + row_count * 26))
    card = pygame.Rect(panel_x + 10, win_h - card_h - 10, panel_w - 20, card_h)
    _draw_card(screen, card, CARD_BG, CARD_BORDER)

    top = card.y + 10
    left = card.x + 10
    inner_w = card.width - 20

    dirty = draft != active
    screen.blit(title.render("CONTROL CENTER", True, HUD_TEXT), (left, top))
    status_text = "CHANGES PENDING" if dirty else "CONFIG ACTIVE"
    status_color = HUD_WARNING if dirty else HUD_SUCCESS
    pill_w = max(92, tiny.size(status_text)[0] + 16)
    _draw_pill(screen, pygame.Rect(card.right - pill_w - 10, top, pill_w, 20),
               status_text, tiny, (*status_color[:3],), (15, 22, 30))
    top += 24

    btn_h, gap = 24, 8
    btn_w = (inner_w - 2 * gap) // 3
    primary_label = "Resume" if paused else "Pause"
    for i, (action, label) in enumerate([
        ("btn_pause", primary_label), ("btn_restart", "Restart"), ("btn_apply", "Apply changes")
    ]):
        rect = pygame.Rect(left + i * (btn_w + gap), top, btn_w, btn_h)
        enabled = action != "btn_apply" or dirty
        _draw_button(screen, rect, label, small, mouse_pos, active=enabled)
        if enabled:
            clickable.append((rect, action))
    top += btn_h + 10

    def selector_row(label: str, value: str, action_prev: str, action_next: str) -> None:
        nonlocal top
        screen.blit(tiny.render(label, True, HUD_DIM), (left, top + 5))
        minus_r = pygame.Rect(left + 96, top, 24, 22)
        plus_r = pygame.Rect(left + inner_w - 24, top, 24, 22)
        val_r = pygame.Rect(left + 124, top, inner_w - 152, 22)
        _draw_button(screen, minus_r, "‹", small, mouse_pos)
        _draw_button(screen, plus_r, "›", small, mouse_pos)
        pygame.draw.rect(screen, (32, 38, 52), val_r, border_radius=6)
        pygame.draw.rect(screen, (70, 84, 110), val_r, 1, border_radius=6)
        vs = tiny.render(value, True, HUD_TEXT)
        screen.blit(vs, (val_r.centerx - vs.get_width() // 2,
                         val_r.y + (val_r.height - vs.get_height()) // 2))
        clickable.append((minus_r, action_prev))
        clickable.append((plus_r, action_next))
        top += 26

    def numeric_row(label: str, key: str, decimals: int, step_txt: str) -> None:
        v = draft["overrides"][key]
        if key in {"initial_alpha", "initial_beta"} and v < 0:
            txt = "random"
        else:
            txt = f"{v:.{decimals}f}" if decimals > 0 else f"{int(v)}"
        selector_row(label, f"{txt} (+/-{step_txt})", f"{key}_dec", f"{key}_inc")

    selector_row("Scenario", draft["scenario"], "scenario_dec", "scenario_inc")
    selector_row("Agent", _short_mode_label(draft["mode"], draft["rule_variant"]), "mode_dec", "mode_inc")
    if draft["mode"] in ("rule", "random", "llm"):
        selector_row("Hidden layers", "n/a (mlp only)", "noop", "noop")
    else:
        selector_row("Hidden layers", str(int(draft["n_layers"])), "layers_dec", "layers_inc")
    selector_row("Max steps", "inf" if draft["steps"] == -1 else str(int(draft["steps"])), "steps_dec", "steps_inc")
    numeric_row("Regeneration", "food_regen", 3, "0.005")
    numeric_row("Metabolism", "base_metabolism", 3, "0.05")
    numeric_row("Init. agents", "initial_agents", 0, "5")
    numeric_row("Obs. noise σ", "obs_noise_sigma", 2, "0.10")
    if draft["mode"] == "rule":
        numeric_row("Alpha", "initial_alpha", 2, "0.05")
        numeric_row("Beta", "initial_beta", 2, "0.05")
    screen.blit(tiny.render(
        "SPACE pause  •  R restart  •  ENTER apply  •  F11 fullscreen",
        True, (110, 120, 146),
    ), (left, card.bottom - 18))

    return [(r, a) for (r, a) in clickable if a != "noop"]


# ---------------------------------------------------------------------------
# World rendering
# ---------------------------------------------------------------------------

def draw_world(
    screen, world: World, fonts: dict, title: str,
    step: int, pop_history: List[int], energy_history: List[float],
    prop_history: List[float], beta_history: List[float], fps_target: int, paused: bool,
    cell_size: int, panel_width: int,
    mouse_pos: tuple = (0, 0),
) -> None:
    import pygame

    cols = world.params.grid_width
    rows = world.params.grid_height
    grid_w = cols * cell_size
    grid_h = rows * cell_size

    screen.fill(BG_COLOR)

    # Food field: inset cells keep the grid visible without heavy crossing lines.
    for x in range(cols):
        for y in range(rows):
            c = world.food_grid[x, y]
            frac = max(0.0, min(1.0, c / world.params.max_food_cell))
            color = _lerp_color(FOOD_EMPTY, FOOD_HIGH, frac) if c > 0 else FOOD_EMPTY
            inset = 1 if cell_size >= 12 else 0
            pygame.draw.rect(
                screen, color,
                (x * cell_size + inset, y * cell_size + inset,
                 cell_size - inset, cell_size - inset),
                border_radius=2 if cell_size >= 16 else 0,
            )

    # Grid lines
    if cell_size < 12:
        for x in range(cols + 1):
            pygame.draw.line(screen, GRID_COLOR, (x * cell_size, 0), (x * cell_size, grid_h))
        for y in range(rows + 1):
            pygame.draw.line(screen, GRID_COLOR, (0, y * cell_size), (grid_w, y * cell_size))

    # Agents (Pac-Man sized by energy)
    threshold = max(1e-6, world.params.reproduction_scale)
    for a in world.agents:
        is_mlp = hasattr(a, "net")
        is_random = a.__class__.__name__ == "RandomAgent"
        is_llm = a.__class__.__name__ == "LLMAgent"
        if is_random:
            color, border = AGENT_RANDOM, AGENT_BORDER_RANDOM
        elif is_llm:
            color, border = AGENT_LLM, AGENT_BORDER_LLM
        elif is_mlp:
            color, border = AGENT_MLP, AGENT_BORDER_MLP
        else:
            color, border = AGENT_RULE, AGENT_BORDER_RULE

        cx = int(a.x * cell_size + cell_size / 2)
        cy = int(a.y * cell_size + cell_size / 2)
        radius = int(3 + (cell_size / 2 - 4) * max(0.0, min(1.0, a.energy / threshold)))

        pygame.draw.circle(screen, (7, 10, 16), (cx + 1, cy + 2), radius + 1)

        if a.energy >= threshold:
            glow_r = radius + 6
            glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*color, 70), (glow_r, glow_r), glow_r)
            screen.blit(glow, (cx - glow_r, cy - glow_r))

        dx, dy = a.last_dir
        mouth_angle = math.atan2(-dy, dx) if (dx, dy) != (0, 0) else 0.0
        mouth_half = math.pi / 6
        start_a = mouth_angle + mouth_half
        end_a = mouth_angle - mouth_half + 2 * math.pi
        pts = [(cx, cy)] + [
            (cx + int(radius * math.cos(start_a + (end_a - start_a) * i / 20)),
             cy - int(radius * math.sin(start_a + (end_a - start_a) * i / 20)))
            for i in range(21)
        ]
        pygame.draw.polygon(screen, color, pts)
        pygame.draw.polygon(screen, border, pts, 1)

    # Agent tooltip
    mx, my = mouse_pos
    if 0 <= mx < grid_w and 0 <= my < grid_h:
        gx, gy = mx // cell_size, my // cell_size
        for a in world.agents:
            if a.x == gx and a.y == gy:
                tip = f"E: {a.energy:.2f}  Pos: ({a.x},{a.y})  Age: {a.age}"
                if hasattr(a, "alpha"):
                    tip += f"  alpha: {a.alpha:.2f}"
                if hasattr(a, "beta"):
                    tip += f"  beta: {a.beta:.2f}"
                ts = fonts["small"].render(tip, True, (255, 255, 255))
                tw, th = ts.get_size()
                pad = 4
                tx = min(mx + 12, grid_w - tw - pad * 2)
                ty = max(0, my - th - pad * 2 - 4)
                bg_surf = pygame.Surface((tw + pad * 2, th + pad * 2), pygame.SRCALPHA)
                pygame.draw.rect(bg_surf, (20, 20, 30, 220), (0, 0, tw + pad * 2, th + pad * 2), border_radius=5)
                screen.blit(bg_surf, (tx, ty))
                screen.blit(ts, (tx + pad, ty + pad))
                break

    # Side panel
    panel_x = grid_w
    win_h = screen.get_height()
    pygame.draw.rect(screen, HUD_BG, (panel_x, 0, panel_width, win_h))
    for i in range(win_h):
        t = i / max(1, win_h - 1)
        pygame.draw.line(screen, _lerp_color((22, 24, 34), (28, 32, 46), t),
                         (panel_x + 1, i), (panel_x + panel_width - 1, i))
    pygame.draw.line(screen, (50, 50, 70), (panel_x, 0), (panel_x, win_h), 2)

    px, py = panel_x + 15, 12
    graph_w = panel_width - 35
    graph_h = 60

    header = pygame.Rect(px, py, graph_w, 58)
    _draw_card(screen, header, CARD_BG, CARD_BORDER)
    screen.blit(fonts["title"].render(title, True, HUD_ACCENT), (px + 12, py + 8))
    mode_label = _short_mode_label(world.agent_type, "") if world.agent_type != "rule" else "rule-based"
    screen.blit(fonts["tiny"].render(mode_label.upper(), True, HUD_DIM), (px + 12, py + 34))
    status_text = "PAUSED" if paused else "RUNNING"
    status_color = HUD_WARNING if paused else HUD_SUCCESS
    status_w = max(72, fonts["tiny"].size(status_text)[0] + 18)
    _draw_pill(screen, pygame.Rect(header.right - status_w - 10, py + 9, status_w, 20),
               status_text, fonts["tiny"], status_color, (12, 20, 27))
    screen.blit(fonts["tiny"].render(f"{fps_target} FPS", True, HUD_DIM),
                (header.right - 48, py + 36))
    py += 70

    n_agents = len(world.agents)
    stats = [("STEP", str(step)), ("FOOD", f"{world.total_food:.1f}"),
             ("MOVES", str(world.moves_step))]
    stat_gap = 6
    stat_w = (graph_w - stat_gap * 2) // 3
    for i, (label, value) in enumerate(stats):
        stat = pygame.Rect(px + i * (stat_w + stat_gap), py, stat_w, 48)
        _draw_card(screen, stat, (21, 29, 42), (45, 59, 80))
        screen.blit(fonts["tiny"].render(label, True, HUD_DIM), (stat.x + 8, stat.y + 6))
        screen.blit(fonts["main"].render(value, True, HUD_TEXT), (stat.x + 8, stat.y + 23))
    py += 60

    avg_e = sum(a.energy for a in world.agents) / n_agents if n_agents > 0 else 0.0

    if n_agents > 0 and world.agent_type == "rule":
        avg_prop = sum(a.alpha for a in world.agents) / n_agents
        avg_beta = sum(a.beta for a in world.agents) / n_agents
        avg_prop_text = f"{avg_prop:.2f}"
        avg_beta_text = f"{avg_beta:.2f}"
    else:
        avg_prop_text = "n/a"
        avg_beta_text = "n/a"

    def _mini_graph(history: List[float], color: tuple, label: str, value_txt: str) -> None:
        nonlocal py
        screen.blit(fonts["small"].render(label, True, HUD_DIM), (px, py))
        screen.blit(fonts["main"].render(value_txt, True, HUD_TEXT), (px + 185, py - 1)); py += 18
        rect = pygame.Rect(px, py, graph_w, graph_h)
        _draw_card(screen, rect, (25, 28, 39), (59, 70, 94))
        if len(history) > 1:
            mx_h = max(history) or 1.0
            n_pts = min(len(history), graph_w)
            step_sz = max(1, len(history) // n_pts)
            sampled = history[::step_sz][-n_pts:]
            pts = [
                (px + int(i * graph_w / max(len(sampled) - 1, 1)),
                 py + graph_h - int(v / mx_h * (graph_h - 4)) - 2)
                for i, v in enumerate(sampled)
            ]
            if len(pts) >= 2:
                pygame.draw.lines(screen, color, False, pts, 2)
        py += graph_h + 10

    if world.agent_type == "rule":
        _mini_graph(prop_history, (255, 194, 92), "Mean propensity", avg_prop_text)
        _mini_graph(beta_history, (235, 120, 140), "Mean beta gene", avg_beta_text)

    _mini_graph([float(v) for v in pop_history], HUD_ACCENT,
                "Live agents", str(n_agents))
    _mini_graph(energy_history, (80, 220, 120), "Mean energy", f"{avg_e:.1f}")
    py += 5

    # Legend
    screen.blit(fonts["small"].render("Legend", True, HUD_DIM), (px, py)); py += 18
    for color, label in [
        (AGENT_MLP, "MLP"),
        (AGENT_RULE, "Rule-based"),
        (AGENT_LLM, "LLM"),
        (AGENT_RANDOM, "Random"),
    ]:
        pygame.draw.circle(screen, color, (px + 8, py + 6), 6)
        screen.blit(fonts["small"].render(label, True, HUD_TEXT), (px + 20, py)); py += 18


# ---------------------------------------------------------------------------
# GUI entry point
# ---------------------------------------------------------------------------

def run_gui(
    scenario: str,
    mode: str,
    steps: int,
    n_layers: int,
    weight_tag: str = "sup_rb_fix_fix_auto",
    rule_variant: str = "rb_fix_fix",
    obs_noise_sigma: float = 0.0,
) -> None:
    import pygame

    print(f"--- GUI: {scenario.upper()} [{mode}] ---")

    mode_options = _runtime_agent_options()
    if rule_variant not in RULE_VARIANT_TAGS:
        rule_variant = RULE_VARIANT_TAGS[0]
    if mode == "rule":
        mode_token = f"rule::{rule_variant}"
    elif mode in MLP_VARIANT_TAGS or mode in LLM_VARIANT_TAGS or mode == "random":
        mode_token = mode
    else:
        mode_token = mode_options[0]

    draft: Dict = {
        "scenario": scenario, "mode": mode, "steps": int(steps),
        "n_layers": int(n_layers), "weight_tag": weight_tag,
        "rule_variant": rule_variant, "overrides": _default_runtime_overrides(scenario),
    }
    draft["overrides"]["obs_noise_sigma"] = max(0.0, float(obs_noise_sigma))
    _apply_agent_option(draft, mode_token)
    active = _clone_state(draft)
    world, title = _build_world(active)

    pygame.init()
    info = pygame.display.Info()
    desktop_w = max(1024, info.current_w)
    desktop_h = max(720, info.current_h)
    target_w = max(1024, int(desktop_w * 0.88))
    target_h = max(720, int(desktop_h * 0.88))
    fullscreen = False

    cell_size = CELL_SIZE
    panel_w = PANEL_WIDTH
    screen = None
    grid_w = grid_h = win_h = 0

    def _apply_video(override_size=None) -> None:
        nonlocal screen, grid_w, grid_h, win_h, target_w, target_h, cell_size, panel_w
        if fullscreen:
            target_w, target_h = desktop_w, desktop_h
            flags = pygame.FULLSCREEN
        else:
            if override_size:
                target_w, target_h = max(900, override_size[0]), max(620, override_size[1])
            flags = pygame.RESIZABLE
        cell_size, panel_w, grid_w, grid_h, win_h = _compute_gui_layout(target_w, target_h, world.params)
        screen = pygame.display.set_mode((grid_w + panel_w, win_h), flags)

    _apply_video()
    pygame.display.set_caption(f"Simulation: {active['scenario']} - {active['mode']}")
    clock = pygame.time.Clock()

    fonts = {
        "title": pygame.font.SysFont("Segoe UI", 17, bold=True),
        "main":  pygame.font.SysFont("Segoe UI", 14, bold=True),
        "small": pygame.font.SysFont("Segoe UI", 12),
        "tiny":  pygame.font.SysFont("Segoe UI", 11),
    }

    running = True
    paused = False
    fps_target = FPS_DEFAULT
    step = 0
    pop_history: List[int] = []
    energy_history: List[float] = []
    prop_history: List[float] = []
    beta_history: List[float] = []
    clickable: List[Tuple[object, str]] = []

    def _restart(state: Dict) -> None:
        nonlocal world, title, step, pop_history, energy_history, prop_history, beta_history, paused, active
        world, title = _build_world(state)
        _apply_video()
        active = _clone_state(state)
        step, pop_history, energy_history, prop_history, beta_history, paused = 0, [], [], [], [], False
        pygame.display.set_caption(f"Simulation: {active['scenario']} - {active['mode']}")

    STEPS_PARAMS = {
        "food_regen":             (0.005, (0.0, 1.5)),
        "base_metabolism":        (0.05,  (0.01, 3.5)),
        "initial_agents":         (5.0,   (5.0, float(world.params.grid_width * world.params.grid_height))),
        "initial_alpha":          (0.05,  (-1.0, 2.0)),
        "initial_beta":           (0.05,  (-1.0, 1.0)),
        "obs_noise_sigma":        (0.10,  (0.0, 3.0)),
    }

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                _apply_video((event.w, event.h))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, action in clickable:
                    if not rect.collidepoint(event.pos):
                        continue
                    if action == "btn_pause":
                        paused = not paused
                    elif action == "btn_restart":
                        _restart(active)
                    elif action == "btn_apply":
                        _restart(draft)
                    elif action in {"scenario_dec", "scenario_inc"}:
                        draft["scenario"] = _cycle_option(
                            VALID_SCENARIOS, draft["scenario"], -1 if action.endswith("dec") else 1)
                        draft["overrides"] = _default_runtime_overrides(draft["scenario"])
                    elif action in {"mode_dec", "mode_inc"}:
                        next_opt = _cycle_option(mode_options, _state_to_agent_option(draft),
                                                 -1 if action.endswith("dec") else 1)
                        _apply_agent_option(draft, next_opt)
                    elif action in {"layers_dec", "layers_inc"} and draft["mode"] not in ("rule", "random", "llm"):
                        draft["n_layers"] = int(_clamp(draft["n_layers"] + (-1 if action.endswith("dec") else 1), 1, 6))
                    elif action in {"steps_dec", "steps_inc"}:
                        if draft["steps"] == -1:
                            draft["steps"] = 500
                        else:
                            delta = -100 if action.endswith("dec") else 100
                            draft["steps"] = int(_clamp(draft["steps"] + delta, 100, 20000))
                            if action.endswith("dec") and draft["steps"] <= 100:
                                draft["steps"] = -1
                    elif action.endswith(("_inc", "_dec")):
                        key = action.rsplit("_", 1)[0]
                        if key in draft["overrides"] and key in STEPS_PARAMS:
                            step_sz, (lo, hi) = STEPS_PARAMS[key]
                            sign = -1.0 if action.endswith("dec") else 1.0
                            val = _clamp(draft["overrides"][key] + sign * step_sz, lo, hi)
                            if key == "initial_agents":
                                val = float(int(val // 5) * 5)
                            draft["overrides"][key] = val
                    break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    _restart(active)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    _restart(draft)
                elif event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    _apply_video()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps_target = min(FPS_MAX, fps_target + 5)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps_target = max(FPS_MIN, fps_target - 5)

        mouse_pos = pygame.mouse.get_pos()
        draw_world(screen, world, fonts, title, step,
                   pop_history, energy_history, prop_history, beta_history,
                   fps_target, paused, cell_size, panel_w, mouse_pos)
        clickable = _draw_runtime_controls(
            screen, fonts, panel_x=grid_w, panel_w=panel_w, win_h=win_h,
            draft=draft, active=active, paused=paused, mouse_pos=mouse_pos,
        )
        if active["mode"] == "llm" and not paused:
            thinking_surf = fonts["small"].render("LLM thinking...", True, (255, 208, 80))
            screen.blit(thinking_surf, (grid_w + 15, 8))
        pygame.display.flip()
        clock.tick(fps_target)

        step_limit = int(active["steps"])
        if not paused and (step_limit == -1 or step < step_limit):
            world.step()
            step += 1
            n = len(world.agents)
            pop_history.append(n)
            energy_history.append(sum(a.energy for a in world.agents) / n if n > 0 else 0.0)
            if n > 0 and world.agent_type == "rule":
                prop_history.append(sum(a.alpha for a in world.agents) / n)
                beta_history.append(sum(a.beta for a in world.agents) / n)
            else:
                prop_history.append(0.0)
                beta_history.append(0.0)
            if not world.agents:
                paused = True

    pygame.quit()
