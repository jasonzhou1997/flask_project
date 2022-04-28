import math as ma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def ADI_Heat(alpha,t,nt,x,Mesh_Grid_Point,T1,T2,T3,T4,T5):
  
  alpha = float(alpha)
  t = float(t)
  nt = int(nt)
  x = float(x)
  Mesh_Grid_Point = float(Mesh_Grid_Point)
  T1 = float(T1)
  T2 = float(T2)
  T3 = float(T3)
  T4 = float(T4)
  T5 = float(T5)


  #alpha = 9.7E-5
  #t = 100
  #nt = 10
  dt = t / nt
  #x = 1.0
  y = x
  Mesh_Grid_Point = 10
  dx = x / Mesh_Grid_Point
  dy = y / Mesh_Grid_Point

  r_x = alpha * dt / (dx * dx)
  r_y = r_x

  #T1 = 100  # BC
  #T2 = 100
  #T3 = 100
  #T4 = 100

  # Initial Conditions
  #T5 = 200
  N_total = (x / dx + 1) * (y / dy + 1)
  N_interior = (x / dx - 1) * (y / dy - 1)
  R_total = int(ma.pow(N_total, 0.5))
  R_total_float = (ma.pow(N_total, 0.5))
  R_interior = int(ma.pow(N_interior, 0.5))
  R_interior_float = ma.pow(N_interior, 0.5)

  T_interior = np.full((R_interior, R_interior), T5)
  # Setting up Temperature Matrix
  T_Matrix_ini = np.zeros((R_total, R_total))
  T_Matrix_ini[:, 0] = T1
  T_Matrix_ini[:, -1] = T2
  T_Matrix_ini[0, :] = T3
  T_Matrix_ini[-1, :] = T4
  T_Matrix_BC = T_Matrix_ini
  T_Matrix_ini[1:-1, 1:-1] = T5
  beta = 2 * (1 + r_x)
  gamma = 2 * (1 - r_x)
  A = np.zeros((R_interior * R_interior, R_interior * R_interior))
  # print(T_Matrix_ini)
  for i in range(R_interior * R_interior):
    for j in range(R_interior * R_interior):
      if i == j:
        A[i, j] = beta
      elif (i == j - 1) and ((i + 1) % R_interior != 0):
        A[i, j] = -1 * r_x
      elif (j == i - 1) and ((j + 1) % R_interior != 0):
        A[i, j] = -1 * r_x
      else:
        A[i, j] = 0

  T_n = np.full((R_interior * R_interior, 1), T5)

  B = np.zeros((R_interior * R_interior, R_interior * R_interior))

  gamma_arr = np.ones(R_interior * R_interior)
  gamma_arr = np.multiply(gamma_arr, gamma)
  v_diag_arr = np.ones(R_interior * R_interior - R_interior)
  v_diag_arr = np.multiply(v_diag_arr, r_x)
  B_main = np.diag(gamma_arr, 0)
  B_upper = np.diag(v_diag_arr, R_interior)
  B_lower = np.diag(v_diag_arr, -1 * R_interior)
  B = B_main + B_upper + B_lower

  # print(B)

  # Constructing C Matrix
  C_arr = np.zeros((R_interior, R_interior))
  # print(C_arr)
  for i in range(1, R_total - 1):
    for j in range(1, R_total - 1):
      if i == 1 and j == 1:
        C_arr[i - 1, j - 1] = T_Matrix_ini[i - 1, j] + T_Matrix_ini[i, j - 1]
      elif i == 1 and j != 1 and j != (R_total - 2):
        C_arr[i - 1, j - 1] = T_Matrix_ini[i - 1, j]
      elif i == 1 and j == (R_total - 2):
        C_arr[i - 1, j - 1] = T_Matrix_ini[i - 1, j] + T_Matrix_ini[i, j + 1]
      elif i == (R_total - 2) and j == 1:
        C_arr[i - 1, j - 1] = T_Matrix_ini[i + 1, j] + T_Matrix_ini[i, j - 1]
      elif i == (R_total - 2) and j == (R_total - 2):
        C_arr[i - 1, j - 1] = T_Matrix_ini[i + 1, j] + T_Matrix_ini[i, j + 1]
      elif i == (R_total - 2) and j != 1 and j != (R_total - 2):
        C_arr[i - 1, j - 1] = T_Matrix_ini[i + 1, j]
      elif j == 1 and (i != 1 or j != R_total - 2):
        C_arr[i - 1, j - 1] = T_Matrix_ini[i, j - 1]
      elif j == R_total - 2 and (i != 1 or j != R_total - 2):
        C_arr[i - 1, j - 1] = T_Matrix_ini[i, j + 1]
      else:
        C_arr[i - 1, j - 1] = 0

  C_arr = np.multiply(C_arr, r_x)
  # print(C_arr)
  C_Vec = C_arr.flatten()
  C_Vec = C_Vec.reshape((-1, 1))

  RHS_Vec = np.dot(B, T_n) + C_Vec
  # print(RHS_Vec)
  A_Diag_Vec = np.diag(A, 0)
  A_Diag_Vec = A_Diag_Vec.reshape((-1, 1))
  A_L_Diag_Vec = np.diag(A, -1)
  A_L_Diag_Vec = A_L_Diag_Vec.reshape((-1, 1))
  A_U_Diag_Vec = np.diag(A, 1)
  A_U_Diag_Vec = A_U_Diag_Vec.reshape((-1, 1))
  # print(A_Diag_Vec,A_L_Diag_Vec, A_U_Diag_Vec)
  # numpy.diag(v, k=0)
  A_L = np.zeros(1)
  A_L = A_L.reshape((-1, 1))
  A_L = np.append(A_L, A_L_Diag_Vec)
  A_L = A_L.reshape((-1, 1))

  A_U = np.zeros(1)
  A_U = A_U.reshape((-1, 1))
  A_U = np.append(A_U_Diag_Vec, A_U)
  A_U = A_U.reshape((-1, 1))
  #print(RHS_Vec.size)

  A_Diag_Vec = A_Diag_Vec.reshape((1, -1))
  A_L_Diag_Vec = A_L_Diag_Vec.reshape((1, -1))
  A_U_Diag_Vec = A_U_Diag_Vec.reshape((1, -1))
  # RHS_Vec = RHS_Vec.reshape((1,-1))
  A_L = A_L.reshape((1, -1))
  A_U = A_U.reshape((1, -1))
  #print('1', A_L_Diag_Vec, '2', A_Diag_Vec, '3', A_U_Diag_Vec, '4', RHS_Vec)
  Temp_full_step = np.linalg.solve(A, RHS_Vec)
  #print(Temp_full_step)

  for k in range(nt):
    # Temp_half_step = TDMAsolver(A_L_Diag_Vec, A_Diag_Vec, A_U_Diag_Vec, RHS_Vec)
    # Temp_half_step = Temp_half_step.reshape((-1, 1))
    Temp_half_step = np.linalg.solve(A, RHS_Vec)

    RHS_Vec_Full = np.dot(B, Temp_half_step) + C_Vec

    # RHS_Vec_Full = RHS_Vec_Full.reshape((1, -1))
    # print(RHS_Vec_Full)
    Temp_full_step = np.linalg.solve(A, RHS_Vec_Full)
    RHS_Vec = np.dot(B, Temp_full_step) + C_Vec
    HS_Vec = RHS_Vec.reshape((1, -1))
    if k == int(nt/2):
      Temp_half_step_mid = Temp_half_step
      Temp_full_step_mid = Temp_full_step

  #print(Temp_half_step)
  #print(Temp_full_step)
  Inner_Temp_Profile = Temp_full_step.reshape(R_interior, R_interior)
  #print(Inner_Temp_Profile)
  Total_Temp_Profile = T_Matrix_BC
  Total_Temp_Profile[1:R_interior + 1, 1:R_interior + 1] = Inner_Temp_Profile

  Inner_Temp_Profile_mid = Temp_full_step_mid.reshape(R_interior, R_interior)
  #print(Inner_Temp_Profile)
  Total_Temp_Profile_mid = T_Matrix_BC
  Total_Temp_Profile_mid[1:R_interior + 1, 1:R_interior + 1] = Inner_Temp_Profile_mid
  #print(Total_Temp_Profile)

  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

  # Make data.
  X = np.linspace(0, 1, num=Mesh_Grid_Point + 1)
  Y = np.linspace(0, 1, num=Mesh_Grid_Point + 1)
  # X = np.arange(0, 1.25, dx)
  #print(X)

  X, Y = np.meshgrid(X, Y)

  # Plot the surface.
  #surf = ax.plot_surface(X, Y, Total_Temp_Profile)
  #plt.show()
  return X,Y,Total_Temp_Profile,Total_Temp_Profile_mid
  # TDMAsolver()

#from numba import jit, f8












