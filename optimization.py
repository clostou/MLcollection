
import numpy as np
from numpy import linalg as LA
from traceback import format_exc


__all__ = ['Newton']


class baseOpt():

    def __init__(self):
        pass

    def _iter_direction(self) -> np.matrix:
        pass

    def _iter_step(self) -> float:
        pass

    def _converge_cond(self) -> bool:
        return LA.norm(self.x_new - self.x_old) / LA.norm(self.x_new) <= self.e

    def _iter(self):
        self.x_old = self.x_new
        self.x_new = self.x_old + self._iter_step() * self._iter_direction()

    def _calc(self, max_iter):
        i = 0
        while i < max_iter:
            try:
                self._iter()
                if self._converge_cond():
                    break
            except:
                print("[%s] Error occurred in Step %i." % (self.__class__.__name__, i))
                print(format_exc())
                break
            i += 1
        if i >= max_iter:
            print("[%s] Max iteration reached." % self.__class__.__name__)

    def run(self, x0, epsilon=1e-5, max_iter=1e3):
        if not isinstance(x0, np.ndarray) or x0.ndim != 2 or x0.shape[1] != 1:
            self.x_new = np.array(x0, dtype=np.float).reshape((-1, 1), order='F')
        else:
            self.x_new = x0.astype(np.float)
        self.n = self.x_new.shape[0]
        self.e = epsilon
        self._calc(max_iter)
        return self.x_new


class Newton(baseOpt):

    def __init__(self, dydx, d2ydx2):
        self.dy = dydx
        self.d2y = d2ydx2

    def _iter_direction(self):
        return - np.dot(LA.inv(self.d2y(self.x_old)), self.dy(self.x_old))

    def _iter_step(self):
        return 1


