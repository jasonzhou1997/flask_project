import numpy as np
import pandas as pd
import math as ma

def quad_failure(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,Stress_Res_Arr, Mom_Res_Arr):
    # Quadratic Failure Criteria
    F_xx = 1 / (X_t * X_c)
    F_x = 1 / X_t - 1 / X_c
    F_yy = 1 / (Y_t * Y_c)
    F_y = 1 / Y_t - 1 / Y_c
    F_ss = 1 / (S_c * S_c)
    F_xy = -0.5 * ma.pow(F_xx * F_yy, 0.5)

    Tau_x_bot = On_Axis_Stress_bot_Matrix[:, 0]
    Tau_y_bot = On_Axis_Stress_bot_Matrix[:, 1]
    Tau_s_bot = On_Axis_Stress_bot_Matrix[:, 2]

    Tau_x_top = On_Axis_Stress_top_Matrix[:, 0]
    Tau_y_top = On_Axis_Stress_top_Matrix[:, 1]
    Tau_s_top = On_Axis_Stress_top_Matrix[:, 2]

    Pos_R_bot = np.zeros(Ply_Num)
    Neg_R_bot = np.zeros(Ply_Num)
    Pos_R_top = np.zeros(Ply_Num)
    Neg_R_top = np.zeros(Ply_Num)

    for i in range(Ply_Num):
        R_Square_Term_bot = F_xx * Tau_x_bot[i] * Tau_x_bot[i] + 2 * F_xy * Tau_x_bot[i] * Tau_y_bot[i] \
                            + F_yy * ma.pow(Tau_y_bot[i], 2) + F_ss * ma.pow(Tau_s_bot[i], 2)
        R_term_bot = F_x * Tau_x_bot[i] + F_y * Tau_y_bot[i]
        p_bot_coeff = [R_Square_Term_bot, R_term_bot, -1]
        r_bot = np.roots(p_bot_coeff)
        if r_bot[0] > 0:
            Pos_R_bot[i] = r_bot[0]
            Neg_R_bot[i] = r_bot[1]
        else:
            Pos_R_bot[i] = r_bot[1]
            Neg_R_bot[i] = r_bot[0]

        R_Square_Term_top = F_xx * Tau_x_top[i] * Tau_x_top[i] + 2 * F_xy * Tau_x_top[i] * Tau_y_top[i] \
                            + F_yy * ma.pow(Tau_y_top[i], 2) + F_ss * ma.pow(Tau_s_top[i], 2)
        R_term_top = F_x * Tau_x_top[i] + F_y * Tau_y_top[i]
        p_top_coeff = [R_Square_Term_top, R_term_top, -1]
        r_top = np.roots(p_top_coeff)
        if r_top[0] > 0:
            Pos_R_top[i] = r_top[0]
            Neg_R_top[i] = r_top[1]
        else:
            Pos_R_top[i] = r_top[1]
            Neg_R_top[i] = r_top[0]

    Minimum_R_Quad_bot = np.amin(Pos_R_bot)
    Index_Min_R_Quad_bot = np.where(Pos_R_bot == np.amin(Pos_R_bot))[0]
    Minimum_R_Quad_top = np.amin(Pos_R_top)
    Index_Min_R_Quad_top = np.where(Pos_R_top == np.amin(Pos_R_top))[0]

    print('Minimum_R_Quad_bot = ', Minimum_R_Quad_bot)
    print('Minimum_R_Quad_top = ', Minimum_R_Quad_top)

    if Minimum_R_Quad_bot < Minimum_R_Quad_top:
        R_Quad = Minimum_R_Quad_bot
        Quad_Failure_Lay = Index_Min_R_Quad_bot
        str_Quad = 'Bottom'
    else:
        R_Quad = Minimum_R_Quad_top
        Quad_Failure_Lay = Index_Min_R_Quad_top
        str_Quad = 'top'

    #print('For Quadratic Failure Criteria:')
    #print('Minimum R Value = ', R_Quad, ',', str_Quad, 'Side of Ply', Quad_Failure_Lay, 'would fail first')

    #print('The load vectors which would cause failures are: ')
    Minimum_R_N_Quad = np.multiply(Stress_Res_Arr, R_Quad)
    Minimum_R_M_Quad = np.multiply(Mom_Res_Arr, R_Quad)
    #print(Minimum_R_N_Quad, '[N]', Minimum_R_M_Quad, '[N*M]')
    Layer_Arr = np.arange(1, Ply_Num + 1)

    Quad_Fail_Arr = np.array([Layer_Arr, Pos_R_bot, Neg_R_bot, Pos_R_top, Neg_R_top])
    Quad_Fail_Arr = pd.DataFrame(Quad_Fail_Arr.transpose(), columns=['Ply Number',
                                                                     'Plus_Bot',
                                                                     'Minus_Bot',
                                                                     'Plus_top',
                                                                     'Minus_top', ])
    Quad_Fail_list = [str_Quad,Quad_Failure_Lay+1, R_Quad,Minimum_R_N_Quad,Minimum_R_M_Quad]
    return Quad_Fail_Arr, Quad_Fail_list
