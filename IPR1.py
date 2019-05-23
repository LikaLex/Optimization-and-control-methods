import numpy as np

from math import inf


def inverse_matrix(Ar, x, i):
    l = Ar @ x
    if l[i] == 0:
        raise ValueError("Matrix is irreversible")
    else:
        l1 = l.copy()
        l1[i] = -1
        l2 = (-1.0 / l[i]) * l1
        Q = np.eye(len(x))
        Q[:, i] = l2
        return Q @ Ar


def is_optimum(non_basis_deltas):
    for _, delta in non_basis_deltas:
        if delta <= 0:
            return False
    return True


def get_j_0(non_basis_deltas):
    for index, delta in non_basis_deltas:
        if delta < 0:
            return index


def is_not_limited(z):
    for z_i in z:
        if z_i >= 0:
            return False
    return True


class MainPhaseSimplex:
    def __init__(self, A, c, initial_plan, J_b):
        self.A = A
        self.c = c
        self.initial_plan = initial_plan

        self.x = initial_plan
        self.J_b = J_b

    def _get_non_basis_deltas(self, deltas):
        non_basis_deltas = []
        for index, delta in enumerate(deltas):
            if index not in self.J_b:
                non_basis_deltas.append((index, delta))
        return non_basis_deltas

    def _get_tetas(self, z):
        tetas = []
        for index, value in zip(self.J_b, z):
            if value > 0:
                tetas.append(self.x[index] / value)
            else:
                tetas.append(inf)
        return tetas

    def solve(self):
        A_b = self.A[:, self.J_b]
        inverse_A_b = np.linalg.inv(A_b)

        while True:
            c_b = self.c[self.J_b]

            # STEP 1
            u = c_b @ inverse_A_b
            deltas = u @ self.A - self.c
            non_basis_deltas = self._get_non_basis_deltas(deltas)

            # STEP 2
            if is_optimum(non_basis_deltas):
                print('STOP!')
                print(f'Optimum basis plan = {self.x}, basis = {self.J_b}')
                print(f'c`x_0 = {np.array(self.x) @ self.c}')
                break

            # STEP 3
            else:
                j_0 = get_j_0(non_basis_deltas)
                z = inverse_A_b @ self.A[:, j_0]

                if is_not_limited(z):
                    print("Cost function feasible plans aren't limited at the top")
                    break

                # STEP 4
                else:
                    tetas = self._get_tetas(z)
                    teta_0 = min(tetas)
                    s = tetas.index(teta_0)

                    # STEP 5
                    self.J_b[s] = j_0
                    new_plan = []

                    for index, value in enumerate(self.x):
                        if index in self.J_b:
                            if index == j_0:
                                new_plan.append(teta_0)
                            else:
                                new_plan.append(value - teta_0 * z[self.J_b.index(index)])
                        else:
                            new_plan.append(0)
                    self.x = new_plan

                    # STEP 6
                    inverse_A_b = inverse_matrix(inverse_A_b, self.A[:, j_0], tetas.index(teta_0))


if __name__ == '__main__':
    p = MainPhaseSimplex(
        A=np.array([[0, 1, 4, 1, 0, -8, 1, 5],
                    [0, -1, 0, -1, 0, 0, 0, 0],
                    [0, 2, -1, 0, -1, 3, -1, 0],
                    [1, 1, 1, 1, 0, 3, 1, 1]]),
        c=np.array([-5, 2, 3, -4, -6, 0, 1, -5]),
        initial_plan=np.array([4, 5, 0, 6, 0, 0, 0, 5]),
        J_b=[0, 1, 3, 7]
    )
    p.solve()
