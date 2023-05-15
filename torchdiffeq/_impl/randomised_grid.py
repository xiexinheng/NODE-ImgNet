from .solvers import FixRandomisedGridODESolver,RandomRandomisedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb


class FixRandomisedEuler(FixRandomisedGridODESolver):
    order = 1

    def _step_func(self, func, t0, tau0, dt, t1, y0):
        randomt = t0+dt*tau0
        f0 = func(randomt, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0

class RandomisedRandomisedEuler(RandomRandomisedGridODESolver):
    order = 1

    def _step_func(self, func, t0, tau0, dt, t1, y0):
        randomt = t0 + dt * tau0
        f0 = func(randomt, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0


