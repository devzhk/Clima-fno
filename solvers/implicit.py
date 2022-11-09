import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse


def implicit_solve(model_jac, f, dbc, dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.linspace(dbc[0], dbc[1], Ny)
    # q = -yy*(yy - 1)
    # q[0], q[-1] = dbc[0], dbc[1]
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :], t_data[0] = q, t

    I, J = np.zeros(3*(Ny-2)-2), np.zeros(3*(Ny-2)-2)
    for i in range(Ny-2):
        if i == 0:
            I[0], I[1] = i, i 
            J[0], J[1] = i, i+1
        elif i == Ny-3:
            I[2 + 3*(i - 1) + 0], I[2 + 3*(i - 1) + 1] = i, i
            J[2 + 3*(i - 1) + 0], J[2 + 3*(i - 1) + 1] = i-1, i
        else:
            I[2 + 3*(i - 1) + 0], I[2 + 3*(i - 1) + 1], I[2 + 3*(i - 1) + 2] = i, i, i
            J[2 + 3*(i - 1) + 0], J[2 + 3*(i - 1) + 1], J[2 + 3*(i - 1) + 2] = i-1, i, i+1
            
            
    res = np.zeros(Ny-2)  
    V = np.zeros(3*(Ny-2)-2)
        
    for i in range(1, Nt+1): 
        
        # this include dt in both V
        model_jac(q, yy, res, V)
        # print("error : ", np.linalg.norm(sparse.coo_matrix((V,(I,J)),shape=(Ny-2,Ny-2)).tocsc() * q[1:Ny-1] - res) )
        
        V *= -dt
        V[0::3] += 1.0
        
        A = sparse.coo_matrix((V,(I,J)),shape=(Ny-2,Ny-2)).tocsc()
        dq = spsolve(A, dt*(f[1:Ny-1] + res))
        q[1:Ny-1] += dq

        
        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q: ", np.max(q), " L2 res: ", np.linalg.norm(f[1:Ny-1] + res))
        if i == Nt:
            print("error dq = ", np.linalg.norm(dq/dt))
            
    return  yy, t_data, q_data