import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

def armijo_goldstein():
    # Armijo-Goldstein step size rule.
    sigma = 0.8
    beta = 0.8
    n = 20
    x = np.zeros(n)

    def fun1(x, n):
        N1 = n
        a1, b1, c1 = 2, -1, -1
        diagonals_B1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
        B1 = diags(diagonals_B1, [0, 1, -1]).toarray()
        A1 = B1 * (n + 1)
        a1, b1, c1 = 4, 1, 1
        diagonals_C1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
        C1 = diags(diagonals_C1, [0, 1, -1]).toarray()
        M = C1 * (1 / (6 * (n + 1)))
        O = np.ones(n)
        return 0.5 * (x @ A1 @ x) - O @ M @ x

    def grad1(x):
        N1 = n
        a1, b1, c1 = 2, -1, -1
        diagonals_B1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
        B1 = diags(diagonals_B1, [0, 1, -1]).toarray()
        A1 = B1 * (n + 1)
        a1, b1, c1 = 4, 1, 1
        diagonals_C1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
        C1 = diags(diagonals_C1, [0, 1, -1]).toarray()
        M = C1 * (1 / (6 * (n + 1)))
        O = np.ones(n)
        return A1 @ x - M @ O

    def hess1(n):
        N1 = n
        a1, b1, c1 = 2, -1, -1
        diagonals_B1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
        B1 = diags(diagonals_B1, [0, 1, -1]).toarray()
        A1 = B1 * (n + 1)
        return np.linalg.inv(A1)

    m = 0
    step_size = beta ** m
    step_sizes = []
    final_sz = None

    while m < 100 and step_size > 0.001:
        direction = grad1(x)
        step_size = beta ** m
        if fun1(x - step_size * direction, n) < fun1(x, n) + sigma * step_size * grad1(x).dot(-direction):
            final_sz = step_size
            break
        step_sizes.append(step_size)
        m += 1

    print('Step size variation is:', step_sizes)
    print('Final step size:', final_sz)

    # optimizing the cost function with steepest descent
    epsi = 1e-4
    v = x.copy()
    grad_ini = grad1(v)
    print('initial gradient is:')
    print(grad_ini)
    rhs = epsi * (1 + np.linalg.norm(grad_ini))
    xnew_arr = []
    funct = []
    m = 0
    while m < 100:
        dk = hess1(n) @ grad1(x)
        xnew = x - final_sz * dk
        if np.linalg.norm(grad_ini) <= rhs:
            break
        grad_ini = grad1(xnew)
        xnew_arr.append(xnew)
        funct.append(fun1(xnew, n))
        m += 1
        x = xnew

    print('solution is:')
    print(xnew)
    print('Dimension of the system is:')
    print(xnew.shape)
    print('Iterations for convergence:')
    print(m)
    print('minimum value using SD:')
    value = fun1(xnew, n)
    print(value)
    print('norm of the gradient at the minimum point:')
    print(np.linalg.norm(grad1(xnew)))

    print('Verifying it with pcg method:')
    N1 = 20
    a1, b1, c1 = 2, -1, -1
    diagonals_B1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
    B1 = diags(diagonals_B1, [0, 1, -1]).toarray()
    A1 = B1 * (N1 + 1)
    a1, b1, c1 = 4, 1, 1
    diagonals_C1 = [a1 * np.ones(N1), b1 * np.ones(N1 - 1), c1 * np.ones(N1 - 1)]
    C1 = diags(diagonals_C1, [0, 1, -1]).toarray()
    M1 = C1 * (1 / (6 * (N1 + 1)))
    O = np.ones(N1)
    M2 = M1 @ O

    y, info = cg(A1, M2)
    print('PCG solution:')
    print(y)
    print('minimum of cost function is (using pcg method):')
    fvalue_p = fun1(y, n)
    print(fvalue_p)
    print('norm of the gradient at the minimum point:')
    print(np.linalg.norm(grad1(y)))

if __name__ == "__main__":
    armijo_goldstein()

