from math import inf

import numpy as np


# noinspection PyPep8Naming,PyShadowingNames
class QuadraticProgrammingProblemSolver:
    HR = '=' * 100

    def _set_D(self, D, B):
        if D is None:
            if B is None:
                print("Задайте либо матрицу D, либо матрицу B, "
                      "для подсчета матрицы D")
                return
            else:
                self.D = np.transpose(B) @ B
        else:
            self.D = D

    def _set_c(self, c, d, B):
        if c is None:
            if d is None or B is None:
                print("Задайте либо вектор с, "
                      "либо вектор d и B для подсчета вектора с")
            else:
                self.c = -d @ B
        else:
            self.c = c

    def __init__(self, A, b, x, Jb, B=None, D=None, c=None, d=None):
        self.A = A
        self.pivot_A = None
        self.n, self.m = self.A.shape
        self.b = b
        self.B = B

        self._set_D(D, B)

        self._set_c(c, d, B)
        self.c_pivot = None

        self.x = x
        self.J_pivot = Jb
        self.J_star = Jb[:]
        self.j0 = None

        self.l = None
        self.deltas = None

    function = property(lambda self: np.transpose(self.c) @ self.x + 0.5 *
                        np.transpose(self.x) @ self.D @ self.x)

    def is_optimal(self):
        no_star_deltas = []
        for index, delta in enumerate(self.deltas):
            if index not in self.J_star:
                no_star_deltas.append(delta >= 0)
        return all(no_star_deltas)

    def get_delta_index(self):
        for index, delta in enumerate(self.deltas):
            if index not in self.J_star and delta < 0:
                return index

    def _get_H(self):
        D_star = self.D[self.J_star, :][:, self.J_star]
        A_star = self.A[:, self.J_star]
        top = np.concatenate((D_star, np.transpose(A_star)), axis=1)
        bottom = np.concatenate((A_star, np.zeros((self.n, self.n))), axis=1)
        return np.concatenate((top, bottom), axis=0)

    def _get_bb(self):
        D_j0 = self.D[self.J_star, :][:, self.j0]
        A_j0 = self.A[:, self.j0]
        return np.concatenate((D_j0, A_j0))

    def count_l(self):
        l = np.zeros(self.m)
        l[self.j0] = 1
        H = self._get_H()
        print("H:", H)
        bb = self._get_bb()
        print("bb:", bb)
        temp_vector = -np.linalg.inv(H) @ bb
        print('-H^(-1) bb: ', temp_vector)
        for index, j in enumerate(self.J_star):
            l[j] = temp_vector[index]
        self.l = l
        print("l:", l)

    def _get_tetas(self):
        tetas = []
        for index, l_j in enumerate(self.l):
            if l_j >= 0:
                tetas.append(inf)
            else:
                temp = - self.x[index] / l_j
                tetas.append(temp)
        return tetas

    def _get_teta_delta(self):
        self.delta = np.transpose(self.l) @ self.D @ self.l
        if self.delta == 0:
            teta_j0 = inf
        else:
            teta_j0 = abs(self.deltas[self.j0]) / self.delta
        return teta_j0

    def get_min_teta(self):
        tetas = self._get_tetas()

        teta_delta = self._get_teta_delta()
        tetas[self.j0] = teta_delta

        teta_0 = min(tetas)
        print('Θ: ', tetas)
        return teta_0, tetas.index(teta_0), teta_delta

    def _case_a(self, j_star, j_0):
        print('На данной итерации реализовался случай а)')
        if j_star == j_0:
            self.J_star.append(j_0)
            return True

    def _case_b(self, j_star, j_0, diff, teta):
        print('На данной итерации реализовался случай b)')
        if j_star in diff:
            self.J_star.remove(j_star)
            self.deltas[j_0] = self.deltas[j_0] + teta * self.delta
            return True

    def _case_c(self, j_star, j_0, diff, teta):
        print('На данной итерации реализовался случай c)')
        js_index = self.J_pivot.index(j_star)
        for index in diff:
            if np.linalg.inv(self.pivot_A)[js_index, :] @ self.A[:, index] != 0:
                self.J_pivot.remove(j_star)
                self.J_pivot.append(index)
                self.J_star.remove(j_star)
                self.deltas[j_0] = self.deltas[j_0] + teta * self.delta
                return True

    def _case_d(self, j_star, j_0):
        print('На данной итерации реализовался случай d)')
        self.J_pivot.remove(j_star)
        self.J_pivot.append(j_0)
        self.J_star.remove(j_star)
        self.J_star.append(j_0)

    def count_basis_indexes(self, j_star, j_0, teta_0=None):
        diff = set(self.J_star) - set(self.J_pivot)
        if self._case_a(j_star, j_0):
            return True
        elif self._case_b(j_star, j_0, diff, teta_0):
            return False
        elif self._case_c(j_star, j_0, diff, teta_0):
            return False
        else:
            self._case_d(j_star, j_0)
            return True

    @staticmethod
    def _print_iteration_number(iteration_number):
        print("\nИтерация №: ", iteration_number)
        print(QuadraticProgrammingProblemSolver.HR)

    def solve(self):
        i = 1
        while True:
            should_skip = False

            QuadraticProgrammingProblemSolver._print_iteration_number(i)
            i += 1

            # ШАГ 1
            # Находим вектор с
            c_new = self.D @ self.x + self.c
            print('c(x): ', c_new)
            A_pivot = self.A[:, self.J_pivot]
            self.pivot_A = A_pivot
            print('A_оп: ', self.pivot_A)

            c_pivot = c_new[self.J_pivot]
            self.c_pivot = c_pivot
            print('c_оп(x): ', c_pivot)

            # Находим вектор потенциалов
            u = -c_pivot @ np.linalg.inv(A_pivot)
            print('u: ', u)

            # ШАГ 2
            # Проверяем критерий оптимальности
            self.deltas = u @ self.A + c_new
            print("Δ:", self.deltas)
            if not self.is_optimal():
                # Зафиксируем индекс j_0 для которго delta j_0 < 0
                self.j0 = self.get_delta_index()
                print('j_0: ', self.j0)
                while not should_skip:
                    # ШАГ 3
                    # Построим направление l
                    self.count_l()

                    # ШАГ 4
                    teta_0, j_star, teta_delta = self.get_min_teta()
                    print('Θ_0: ', teta_0)
                    print('j_*: ', j_star)

                    # ШАГ 5
                    self.x = self.x + teta_0 * self.l
                    print("x:", self.x)

                    # ШАГ 6
                    should_skip = self.count_basis_indexes(j_star, self.j0,
                                                           teta_0=teta_0)
                    print("Индексы Jon и J*:", self.J_pivot, self.J_star)
            else:
                return self.x


def test_1():
    print(f'{"-" * 46} TEST 2 {"-" * 46}')
    p_2 = QuadraticProgrammingProblemSolver(
        A=np.array([
            [11, 0, 0, 1, 0, -4, -1, 1],
            [1, 1, 0, 0, 1, -1, -1, 1],
            [1, 1, 1, 0, 1, 2, -2, 1]
        ]),
        b=np.array([8, 2, 5]),
        B=np.array([
            [1, -1, 0, 3, -1, 5, -2, 1],
            [2, 5, 0, 0, -1, 4, 0, 0],
            [-1, 3, 0, 5, 4, -1, -2, 1]
        ]),
        d=np.array([6, 10, 9]),
        x=np.array([0.7273, 1.2727, 3.0, 0, 0, 0, 0, 0]),
        Jb=[0, 1, 2]
    )
    print("\nРешение:", p_2.solve())
    print("f(x) = ", p_2.function)


if __name__ == '__main__':
    test_1()

