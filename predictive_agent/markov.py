"""Markov chain for pod state transitions."""

import time


class MarkovChain:
    """First-order Markov chain for pod state transitions.

    States: HEALTHY → DEGRADED → STRESSED → CRITICAL → FAILED → RECOVERED
    """

    STATES = ["HEALTHY", "DEGRADED", "STRESSED", "CRITICAL", "FAILED", "RECOVERED"]
    STATE_IDX = {s: i for i, s in enumerate(STATES)}

    # Prior counts (pseudo-observations) — weak priors so real observations
    # dominate after ~10 transitions instead of ~100.
    PRIOR_COUNTS: "list[list[float]]" = [
        [9.5, 0.4, 0.1, 0, 0, 0],   # HEALTHY
        [3, 5, 1.5, 0.4, 0, 0.1],    # DEGRADED
        [1, 2, 4, 2.5, 0.4, 0.1],    # STRESSED
        [0.5, 0.5, 1, 6, 1.5, 0.5],  # CRITICAL
        [0, 0, 0, 0, 8, 2],          # FAILED
        [8, 1.5, 0.5, 0, 0, 0],      # RECOVERED
    ]

    def __init__(self):
        self.counts = [row[:] for row in self.PRIOR_COUNTS]
        self.total_transitions = 0
        self.last_updated = 0.0

    def record_transition(self, from_state, to_state):
        """Record a state transition."""
        from_idx = self.STATE_IDX.get(from_state, 0)
        to_idx = self.STATE_IDX.get(to_state, 0)
        self.counts[from_idx][to_idx] += 1
        self.total_transitions += 1
        self.last_updated = time.time()

    def transition_matrix(self):
        """Return normalized transition probabilities (6x6)."""
        matrix = []
        for row in self.counts:
            total = sum(row)
            if total > 0:
                matrix.append([v / total for v in row])
            else:
                matrix.append([1 / 6] * 6)
        return matrix

    def predict(self, from_state, steps=1):
        """Predict state distribution after n steps. Returns {state: prob}."""
        matrix = self.transition_matrix()
        from_idx = self.STATE_IDX.get(from_state, 0)

        vec = [0.0] * 6
        vec[from_idx] = 1.0

        for _ in range(steps):
            new_vec = [0.0] * 6
            for i in range(6):
                for j in range(6):
                    new_vec[j] += vec[i] * matrix[i][j]
            vec = new_vec

        return {self.STATES[i]: vec[i] for i in range(6)}

    def most_likely_next(self, from_state, steps=1):
        """Return (state, probability) of most likely next state."""
        probs = self.predict(from_state, steps)
        best = max(probs, key=probs.get)
        return best, probs[best]

    def to_dict(self):
        """Serialize to dict for persistence."""
        return {
            "counts": self.counts,
            "total_transitions": self.total_transitions,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        chain = cls()
        if data:
            chain.counts = data.get("counts", [row[:] for row in cls.PRIOR_COUNTS])
            chain.total_transitions = data.get("total_transitions", 0)
            chain.last_updated = data.get("last_updated", 0)
        return chain
