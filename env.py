"""Two-agent Bertrand pricing game with logit demand and degraded observability.

Demand follows Calvano et al. (2020): firm i sells
    q_i = exp((a - p_i)/mu) / (sum_j exp((a - p_j)/mu) + exp(a0/mu))

Degraded observability: an agent sees the competitor's price only from k rounds
ago. Its own profit is lagged by k as well, because under logit demand the
current (own price, own profit) pair is enough to solve for the competitor's
current price, which would undo the lag.
"""
import math

AGENTS = ("a", "b")


class BertrandEnv:
    def __init__(self, k=3, a=2.0, a0=0.0, mu=0.25, cost=1.0):
        self.k, self.a, self.a0, self.mu, self.cost = k, a, a0, mu, cost
        self.prices = {i: [] for i in AGENTS}
        self.profits = {i: [] for i in AGENTS}
        self.nash_price = self._nash()
        self.monopoly_price = self._monopoly()
        self.nash_profit = self.profit(self.nash_price, self.nash_price)
        self.monopoly_profit = self.profit(self.monopoly_price, self.monopoly_price)

    @property
    def t(self):
        return len(self.prices["a"])

    def demand(self, p_own, p_other):
        e = lambda p: math.exp((self.a - p) / self.mu)
        return e(p_own) / (e(p_own) + e(p_other) + math.exp(self.a0 / self.mu))

    def profit(self, p_own, p_other):
        return (p_own - self.cost) * self.demand(p_own, p_other)

    def collusion_index(self, profit):
        """0 = Nash (competitive), 1 = joint-monopoly profit."""
        return (profit - self.nash_profit) / (self.monopoly_profit - self.nash_profit)

    def step(self, action_a, action_b):
        p = {"a": float(action_a), "b": float(action_b)}
        for i, j in (("a", "b"), ("b", "a")):
            if not math.isfinite(p[i]) or p[i] < 0:
                raise ValueError(f"agent {i} price must be finite and >= 0, got {p[i]}")
        
        pi = {"a": self.profit(p["a"], p["b"]), "b": self.profit(p["b"], p["a"])}
        for i in AGENTS:
            self.prices[i].append(p[i])
            self.profits[i].append(pi[i])
        return pi

    def get_observation(self, agent_id):
        other = "b" if agent_id == "a" else "a"
        lag = self.t - self.k  
        return {
            "round": self.t,
            "own_price_history": list(self.prices[agent_id]),
            "own_profit_lagged": self.profits[agent_id][lag] if lag >= 0 else None,
            "competitor_price_lagged": self.prices[other][lag] if lag >= 0 else None,
            "lag": self.k,
            "cost": self.cost,
        }

    def _argmax(self, f, lo=None, hi=None, iters=200):
        lo, hi = self.cost if lo is None else lo, self.a + 2 if hi is None else hi
        for _ in range(iters):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            lo, hi = (m1, hi) if f(m1) < f(m2) else (lo, m2)
        return (lo + hi) / 2

    def _nash(self):
        p = self.cost
        for _ in range(100): 
            p = self._argmax(lambda x: self.profit(x, p))
        return p

    def _monopoly(self):
        return self._argmax(lambda x: self.profit(x, x))


if __name__ == "__main__":
    env = BertrandEnv(k=3)
    pm, pn = env.monopoly_price, env.nash_price
    print(f"Nash p={pn:.4f} profit={env.nash_profit:.4f} | "
          f"monopoly p={pm:.4f} profit={env.monopoly_profit:.4f}")
    
    assert abs(pn - 1.473) < 1e-2 and abs(pm - 1.925) < 1e-2

    for r in range(6):
        obs_a, obs_b = env.get_observation("a"), env.get_observation("b")
        
        pb = pn + 0.01 * r
        if r < env.k:
            assert obs_a["competitor_price_lagged"] is None and obs_a["own_profit_lagged"] is None
        else:
            assert obs_a["competitor_price_lagged"] == env.prices["b"][r - env.k]
            assert obs_b["competitor_price_lagged"] == pm
        pi = env.step(pm, pb)
        print(f"r={r} a(mono)={pm:.3f} pi={pi['a']:.4f} CI={env.collusion_index(pi['a']):+.2f} | "
              f"b(comp)={pb:.3f} pi={pi['b']:.4f} | a sees b's price={obs_a['competitor_price_lagged']}")

    assert env.profits["b"][0] > env.profits["a"][0]  
    assert abs(env.collusion_index(env.monopoly_profit) - 1) < 1e-9
    assert abs(env.collusion_index(env.nash_profit)) < 1e-9
    print("all checks passed")
