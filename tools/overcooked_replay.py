"""Render snapshots from the classic Overcooked-AI simulator to an MP4."""

import json
import logging
from pathlib import Path
import subprocess
from tempfile import TemporaryFile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


GRAPHICS_DIR = Path(__file__).with_name("overcooked_graphics")
TERRAIN = {" ": "floor", "X": "counter", "O": "onions", "T": "tomatoes",
           "D": "dishes", "P": "pot", "S": "serve"}
DIRECTION = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}
TILE = 15
SCALE = 4
FPS = 6


def load_sprites(atlas):
    metadata = json.loads((GRAPHICS_DIR / f"{atlas}.json").read_text(encoding="utf-8"))
    with Image.open(GRAPHICS_DIR / f"{atlas}.png") as source:
        sheet = source.convert("RGBA")
    sprites = {}
    for name, entry in metadata["frames"].items():
        x, y, width, height = (entry["frame"][key] for key in ("x", "y", "w", "h"))
        sprites[Path(name).stem] = sheet.crop((x, y, x + width, y + height))
    return sprites


class KitchenRenderer:
    def __init__(self, mdp):
        self.terrain = mdp.terrain_mtx
        self.cook_time = mdp.soup_cooking_time
        self.capacity = mdp.num_items_for_soup
        self.objects = load_sprites("objects")
        self.chefs = load_sprites("chefs")
        tiles = load_sprites("terrain")
        self.board = Image.new("RGBA", (len(self.terrain[0]) * TILE, len(self.terrain) * TILE),
                               "#a97123")
        for y, row in enumerate(self.terrain):
            for x, tile in enumerate(row):
                self.board.alpha_composite(tiles[TERRAIN[tile]], (x * TILE, y * TILE))
        self.frame_size = (self.board.width * SCALE, self.board.height * SCALE + 64)
        self.font = ImageFont.load_default(size=16)
        self.small_font = ImageFont.load_default(size=12)

    def frame(self, state, tick, outcome):
        board = self.board.copy()
        pot_labels = []
        for (x, y), obj in state.objects.items():
            if obj.name == "soup":
                ingredient, count, cooked = obj.state
                status = "dish"
                if self.terrain[y][x] == "P":
                    ready = cooked >= self.cook_time
                    status = "cooked" if ready else f"{count}-cooking"
                    label = f"{count}/{self.capacity}"
                    if count == self.capacity:
                        label = "Ready" if ready else f"{cooked}/{self.cook_time}"
                    pot_labels.append((x, y, label))
                sprite = self.objects[f"soup-{ingredient}-{status}"]
            else:
                sprite = self.objects[obj.name]
            board.alpha_composite(sprite, (x * TILE, y * TILE))

        for player, color in zip(state.players, ("blue", "green")):
            facing = DIRECTION[player.orientation]
            x, y = player.position
            held = player.held_object
            body = facing
            if held is not None:
                item = f"soup-{held.state[0]}" if held.name == "soup" else held.name
                body = f"{facing}-{item}"
            board.alpha_composite(self.chefs[body], (x * TILE, y * TILE))
            board.alpha_composite(self.chefs[f"{facing}-{color}hat"], (x * TILE, y * TILE))

        tile_pixels = TILE * SCALE
        frame = Image.new("RGB", self.frame_size, "#18232c")
        frame.paste(board.resize((self.frame_size[0], self.board.height * SCALE),
                                 Image.Resampling.NEAREST), (0, 32))
        draw = ImageDraw.Draw(frame)
        draw.text((10, 6), f"Grid step {tick}", font=self.font, fill="white")
        if outcome:
            status = "Soup delivered" if outcome == "goal_reached" else outcome.replace("_", " ").capitalize()
            draw.text((frame.width - 10, 7), status, anchor="rt",
                      font=self.small_font, fill="white")
        draw.text((10, frame.height - 24), "Blue  RightChef", font=self.font, fill="#8dccff")
        draw.text((frame.width - 10, frame.height - 22), "Green  LeftChef", anchor="rt",
                  font=self.font, fill="#9fe9ae")
        for x, y, label in pot_labels:
            draw.text((x * tile_pixels + tile_pixels // 2, 32 + (y + 1) * tile_pixels - 4),
                      label, anchor="mb", font=self.small_font, fill="white", stroke_width=1,
                      stroke_fill="#18232c")
        return frame


def save_replay(path, mdp, states, outcome):
    """Write the initial state and one frame per real step; keep the rollout if export fails."""
    temporary = path.with_suffix(".tmp.mp4")
    try:
        renderer = KitchenRenderer(mdp)
        width, height = renderer.frame_size
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-y", str(temporary),
        ]
        with TemporaryFile() as encoder_log, subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=encoder_log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as encoder:
            try:
                with encoder.stdin as frames:
                    for tick, state in enumerate(states):
                        status = outcome if tick == len(states) - 1 else None
                        frames.write(renderer.frame(state, tick, status).tobytes())
            except BrokenPipeError:
                pass
            if encoder.wait() != 0:
                encoder_log.seek(0)
                raise RuntimeError(encoder_log.read().decode("utf-8", errors="replace").strip())
        temporary.replace(path)
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        logging.warning("Could not save kitchen replay %s: %s", path, error)
    finally:
        temporary.unlink(missing_ok=True)
