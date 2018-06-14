import numpy as np

from com_enviroments import wcs

if __name__ == "__main__":
    # c = [-1, 4]
    # A = [[-3, 1], [1, 2]]
    # b = [6, 4]
    # x0_bounds = (None, None)
    # x1_bounds = (-3, None)
    # from scipy.optimize import linprog
    #
    # res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options={"disp": True})


    n = 20
    c = np.zeros(n*n)
    for j in range(n):
        for i in range(n):
            c[i+n*j] = 2 * wcs.sim(i, j) - 1

    A = np.zeros([n*n*n,n*n])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                A[i * j * k, i + n * k] = 1
                A[i * j * k, i + n * j] = -1
                A[i * j * k, j + n * k] = -1

                # r = [0]*n*n
                # r[i+n*k] = 1
                # r[i+n*j] = -1
                # r[j+n*k] = -1
                # A.append(r)


    b = [0] * n*n*n

    bounds = tuple([(0,1) for i in range(n*n)])

    from scipy.optimize import linprog

    #print(pandas.DataFrame(c))
    #print(pandas.DataFrame(A))
    #print(pandas.DataFrame(b))
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, options={"disp": True})

    print(res)



                # b = [0]*



    # from cvxopt import matrix, solvers
    # A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
    # b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
    # c = matrix([ 2.0, 1.0 ])
    #
    # sol=solvers.lp(c,A,b)
    #
    # print(sol['x'])


