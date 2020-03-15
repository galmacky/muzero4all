import random

from trajectory import Trajectory


class ReplayBuffer(object):

  def __init__(self):
    self.window_size = 1e6  # TODO: TUNE SMALLER ?
    self.batch_size = 5  # TODO: TUNE SMALLER ?
    self.buffer = []  # Holds trajectories

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
        self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.action_history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps))
            for (g, i) in game_pos]

  def sample_game(self) -> Trajectory:
    # Sample game from buffer either uniformly or according to some priority.
    # We do in randomly in MuZero4All.
    return random.choice(self.buffer)

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    # We do in randomly in MuZero4All.
    return random.choice(range(len(game.game_states)))