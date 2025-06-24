import numpy as np
import matplotlib.pyplot as plt

def grad1(n):
    """Gradient of Rosenbrock function"""
    return np.array([
        2 * n[0] - 400 * n[0] * (-n[0]**2 + n[1]) - 2,
        -200 * n[0]**2 + 200 * n[1]
    ])

def hess(x):
    """Hessian of Rosenbrock function"""
    return np.array([
        [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ])

def rose_step():
    x = np.array([-1.2, 1.0])
    n = x.copy()
    max_iter = 20
    step_length = 1
    e = 0.001
    k = 1
    x3 = [np.array([0.0, 0.0])]  # history of points
    nor = 1.0

    while k <= max_iter:
        d_k = np.linalg.inv(hess(n)) @ grad1(n)
        g_d = grad1(n)
        nor = np.linalg.norm(g_d)
        nnew = n - step_length * d_k
        x3.append(nnew.copy())

        if np.linalg.norm(g_d) <= e * np.linalg.norm(x):
            break

        k += 1
        n = nnew

    print("The gradient value is:")
    print(g_d)
    print("Number of iterations:")
    print(k)
    print("Norm of the gradient is:")
    print(nor)
    print("Solution for Rosenbrock for given condition:")
    print(n)

    # Visualization
    x3 = np.array(x3).T  # convert to array for plotting
    plt.figure()
    plt.plot(x3[0], x3[1], "r*")

    P, Q = np.meshgrid(np.linspace(-2, 2, 160), np.linspace(-1, 3, 160))
    R = 100 * (Q - P**2)**2 + (1 - P)**2
    plt.contour(P, Q, R, levels=50)
    plt.title("Optimization Path on Rosenbrock Function (2D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, Q, R, cmap='viridis', alpha=0.7)
    ax.plot(x3[0], x3[1], 100 * (x3[1] - x3[0]**2)**2 + (1 - x3[0])**2, 'r*-', label='Optimization Path')
    ax.set_title("Rosenbrock Function Optimization Path (3D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.legend()
    plt.show()

    print("Optimization path (x3):")
    print(x3.T)

# Call the function
rose_step()

