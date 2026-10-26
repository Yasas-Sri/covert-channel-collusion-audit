"""Payoff-irrelevant timing channel: one noisy bit from sender to partner per round.

The environment, not the sender, decides whether a round is active. In an
inactive round the partner sees a bit drawn from the sender's running P(1), so
the share of 1s does not reveal which rounds were active, even for a skewed
sender. Use one channel per episode, since the running count is per episode.
"""
import random


class TimingChannel:
    def __init__(self, q: float, epsilon: float = 0.10, seed=None):
        if not (0 <= q <= 1 and 0 <= epsilon <= 1):
            raise ValueError(f"q and epsilon must be in [0, 1], got q={q}, epsilon={epsilon}")
        self.q, self.epsilon = q, epsilon
        self.rng = random.Random(seed)
        self.n_sent = self.n_ones = 0

    def transmit(self, intended_bit: int) -> tuple[int, bool]:
        if intended_bit not in (0, 1):
            raise ValueError(f"intended_bit must be 0 or 1, got {intended_bit!r}")
        # P(1) uses only earlier bits, so the current intent cannot shape its own decoy.
        p1 = self.n_ones / self.n_sent if self.n_sent else 0.5
        self.n_sent += 1
        self.n_ones += intended_bit
        if self.rng.random() >= self.q:
            return int(self.rng.random() < p1), False
        flipped = self.rng.random() < self.epsilon
        return intended_bit ^ flipped, True
