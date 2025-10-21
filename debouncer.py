
def norm_state(name: str) -> str:
    if not name:
        return "CALM"
    return str(name).strip().lower().replace(" ", "_")

#------------------ DEBOUNCING -----------------------
#debouncing to avoid rapid flips of state -> rapid flips of parameters
class StateDebouncer:
    """
    Require a state to persist for `hold_sec` before accepting it.
    """
    def __init__(self, initial="calm", hold_sec=0.60):
        self.current = norm_state(initial)
        self.candidate = self.current
        self.hold_sec = hold_sec
        self.accum = 0.0

    def update(self, new_state: str, dt: float) -> str:
        s = norm_state(new_state)
        if s == self.current:
            # stable, clear any candidate timing
            self.candidate = s
            self.accum = 0.0
            return self.current

        if s != self.candidate:
            # new candidate – start counting
            self.candidate = s
            self.accum = 0.0
        else:
            # same candidate continuing; accumulate time
            self.accum += max(dt, 0.0)
            if self.accum >= self.hold_sec:
                # accept transition
                self.current = self.candidate
                self.accum = 0.0

        return self.current

