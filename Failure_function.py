import numpy as np
import pandas as pd
import math as ma

def failure_criteria(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,Stress_Res_Arr, Mom_Res_Arr):
    # Maximum Failure Criteria
    FT_bot = np.zeros(Ply_Num)
    FC_bot = np.zeros(Ply_Num)
    MT_bot = np.zeros(Ply_Num)
    MC_bot = np.zeros(Ply_Num)
    S_bot = np.zeros(Ply_Num)
    FT_top = np.zeros(Ply_Num)
    FC_top = np.zeros(Ply_Num)
    MT_top = np.zeros(Ply_Num)
    MC_top = np.zeros(Ply_Num)
    S_top = np.zeros(Ply_Num)

    for i in range(Ply_Num):
        if On_Axis_Stress_bot_Matrix[i, 0] < 0:
            FT_bot[i] = 0
            FC_bot[i] = abs(X_c / On_Axis_Stress_bot_Matrix[i, 0])
        else:
            FT_bot[i] = X_t / On_Axis_Stress_bot_Matrix[i, 0]
            FC_bot[i] = 0
        if On_Axis_Stress_bot_Matrix[i, 1] < 0:
            MT_bot[i] = 0
            MC_bot[i] = abs(Y_c / On_Axis_Stress_bot_Matrix[i, 1])
        else:
            MT_bot[i] = Y_t / On_Axis_Stress_bot_Matrix[i, 0]
            MC_bot[i] = 0
        S_bot[i] = abs(S_c / On_Axis_Stress_bot_Matrix[i, 2])
        if On_Axis_Stress_top_Matrix[i, 0] < 0:
            FT_top[i] = 0
            FC_top[i] = abs(X_c / On_Axis_Stress_top_Matrix[i, 0])
        else:
            FT_top[i] = X_t / On_Axis_Stress_top_Matrix[i, 0]
            FC_top[i] = 0
        if On_Axis_Stress_top_Matrix[i, 1] < 0:
            MT_top[i] = 0
            MC_top[i] = abs(Y_c / On_Axis_Stress_top_Matrix[i, 1])
        else:
            MT_top[i] = Y_t / On_Axis_Stress_top_Matrix[i, 0]
            MC_top[i] = 0
        S_top[i] = abs(S_c / On_Axis_Stress_top_Matrix[i, 2])
    #(FT_bot)
    #print(FC_bot)
    #print(MT_bot)
    #print(MC_bot)
    #print(S_bot)

    if np.array_equal(On_Axis_Stress_top_Matrix, On_Axis_Stress_bot_Matrix) == True:
        R_lowest_Arr = np.array([FT_top[0], MT_top[0], S_top[0]])
        Minimum_R = np.amin(R_lowest_Arr)
        Index_Min_R = np.where(R_lowest_Arr == np.amin(R_lowest_Arr))
        if Index_Min_R[0] == 0:
            str_max = "Failure Mode is Fiber Tension occurs on Top layer of ply"
        elif Index_Min_R[0] == 1:
            str_max = "Failure Mode is Matrix Tension occurs on Top layer of ply"
        elif Index_Min_R[0] == 2:
            str_max = "Failure Mode is Shear occurs on Top layer of ply"
        Failure_Layer = Ply_Num
    else:
        FT_bot_lowest = np.min(FT_bot[np.nonzero(FT_bot)])
        index_FT_bot = np.where(FT_bot == FT_bot_lowest)
        FT_top_lowest = np.min(FT_top[np.nonzero(FT_top)])
        index_FT_top = np.where(FT_top == FT_top_lowest)

        FC_bot_lowest = np.min(FC_bot[np.nonzero(FC_bot)])
        index_FC_bot = np.where(FC_bot == FC_bot_lowest)
        FC_top_lowest = np.min(FC_top[np.nonzero(FC_top)])
        index_FC_top = np.where(FC_top == FC_top_lowest)

        MT_bot_lowest = np.min(MT_bot[np.nonzero(MT_bot)])
        index_MT_bot = np.where(MT_bot == MT_bot_lowest)
        MT_top_lowest = np.min(MT_top[np.nonzero(MT_top)])
        index_MT_top = np.where(MT_top == MT_top_lowest)

        MC_bot_lowest = np.min(MC_bot[np.nonzero(MC_bot)])
        index_MC_bot = np.where(MC_bot == MC_bot_lowest)
        MC_top_lowest = np.min(MC_top[np.nonzero(MC_top)])
        index_MC_top = np.where(MC_top == MC_top_lowest)

        S_bot_lowest = np.min(S_bot[np.nonzero(S_bot)])
        index_S_bot = np.where(S_bot == S_bot_lowest)
        S_top_lowest = np.min(S_top[np.nonzero(S_top)])
        index_S_top = np.where(S_top == S_top_lowest)

        R_lowest_Arr = np.array(
            [FT_bot_lowest, FT_top_lowest, FC_bot_lowest, FC_top_lowest, MT_bot_lowest, MT_top_lowest, MC_bot_lowest,
             MC_top_lowest, S_bot_lowest, S_top_lowest])
        Minimum_R = np.amin(R_lowest_Arr)
        Index_Min_R = np.where(R_lowest_Arr == np.amin(R_lowest_Arr))
        if Index_Min_R == 0:
            str_max = "Failure Mode is Fiber Tension occurs on Bottom layer of ply"
            Failure_Layer = index_FT_bot
        elif Index_Min_R == 1:
            str_max = "Failure Mode is Fiber Tension occurs on Top layer of ply"
            Failure_Layer == index_FT_top
        elif Index_Min_R == 2:
            str_max = "Failure Mode is Fiber Compression occurs on Bottom layer of ply"
            Failure_Layer == index_FC_bot
        elif Index_Min_R == 3:
            str_max = "Failure Mode is Fiber Compression occurs on Top layer of ply"
            Failure_Layer == index_FC_top
        elif Index_Min_R == 4:
            str_max = "Failure Mode is Matrix Tension occurs on Bottom layer of ply"
            Failure_Layer = index_MT_bot
        elif Index_Min_R == 5:
            str_max = "Failure Mode is Matrix Tension occurs on Top layer of ply"
            Failure_Layer == index_MT_top
        elif Index_Min_R == 6:
            str_max = "Failure Mode is Matrix Compression occurs on Bottom layer of ply"
            Failure_Layer == index_MC_bot
        elif Index_Min_R == 7:
            str_max = "Failure Mode is Matrix Compression occurs on Top layer of ply"
            Failure_Layer == index_MC_top
        elif Index_Min_R == 8:
            str_max = "Failure Mode is Shear occurs on Bottom layer of ply"
            Failure_Layer == index_S_bot
        elif Index_Min_R == 9:
            str_max = "Failure Mode is Shear occurs on top layer of ply"
            Failure_Layer == index_S_top

    # Minimum_R_N = Minimum_R * Stress_Res_Arr

    print('The load vectors which would cause failures are: ')
    Minimum_R_N = np.multiply(Stress_Res_Arr, Minimum_R)
    Minimum_R_M = np.multiply(Mom_Res_Arr, Minimum_R)
    print(Minimum_R_N, '[N]', Minimum_R_M, '[N*M]')

    print("Index_Min_R =", Index_Min_R)

    # Minimum_R_M = Minimum_R * Mom_Res_Arr
    Layer_Arr = np.arange(1, Ply_Num + 1)

    Max_Fail_Arr = np.array([Layer_Arr, FT_bot, FC_bot, MT_bot, MC_bot, S_bot, FT_top, FC_top, MT_top, MC_top, S_top])

    Max_Fail_Arr = pd.DataFrame(Max_Fail_Arr.transpose(), columns=['Ply Number',
                                                                   'FT_Bot',
                                                                   'FC_Bot',
                                                                   'MT_Bot',
                                                                   'MC_Bot',
                                                                   'S_Bot',
                                                                   'FT_Top',
                                                                   'FC_Top',
                                                                   'MT_Top',
                                                                   'MC_Top',
                                                                   'S_Top'])
    print(Max_Fail_Arr)

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
    Index_Min_R_Quad_bot = np.where(Minimum_R_Quad_bot == np.amin(Minimum_R_Quad_bot))
    Minimum_R_Quad_top = np.amin(Pos_R_top)
    Index_Min_R_Quad_top = np.where(Minimum_R_Quad_top == np.amin(Minimum_R_Quad_top))

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

    Quad_Fail_Arr = np.array([Layer_Arr, Pos_R_bot, Neg_R_bot, Pos_R_top, Neg_R_top])
    Quad_Fail_Arr = pd.DataFrame(Quad_Fail_Arr.transpose(), columns=['Ply Number',
                                                                     'Plus_Bot',
                                                                     'Minus_Bot',
                                                                     'Plus_top',
                                                                     'Minus_top', ])

    #print(Quad_Fail_Arr)

    # Hashin Failure Criteria
    H_FT_bot = np.zeros(Ply_Num)
    H_FC_bot = np.zeros(Ply_Num)
    H_FT_top = np.zeros(Ply_Num)
    H_FC_top = np.zeros(Ply_Num)
    H_MT_bot = np.zeros(Ply_Num)
    H_MT_top = np.zeros(Ply_Num)
    H_MC_bot = np.zeros(Ply_Num)
    H_MC_top = np.zeros(Ply_Num)
    r_MC_bot_arr = np.zeros(Ply_Num)
    r_MC_top_arr = np.zeros(Ply_Num)

    Q_term = ma.pow((Y_c / (2 * S_c)), 2) - 1
    for x in range(Ply_Num):
        if Tau_x_bot[x] > 0:
            H_FT_bot[x] = ma.pow((1 / ma.pow(Tau_x_bot[i] / X_t, 2) + ma.pow(Tau_s_bot[i] / S_c, 2)), 0.5)
            H_FC_bot[x] = 0
        else:
            H_FT_bot[x] = 0
            H_FC_bot[x] = X_c / abs(Tau_x_bot[x])
        if Tau_x_top[x] > 0:
            H_FT_top[x] = ma.pow((1 / ma.pow(Tau_x_top[i] / X_t, 2) + ma.pow(Tau_s_top[i] / S_c, 2)), 0.5)
            H_FC_top[x] = 0
        else:
            H_FT_top[x] = 0
            H_FC_top[x] = X_c / abs(Tau_x_top[x])

        if Tau_y_bot[x] > 0:
            H_MT_bot[x] = ma.pow((1 / ma.pow(Tau_y_bot[i] / Y_t, 2) + ma.pow(Tau_s_bot[i] / S_c, 2)), 0.5)
        else:
            H_MT_bot[x] = 0
        if Tau_y_top[x] > 0:
            H_MT_top[x] = ma.pow((1 / ma.pow(Tau_y_top[i] / Y_t, 2) + ma.pow(Tau_s_top[i] / S_c, 2)), 0.5)
        else:
            H_MT_top[x] = 0

        if Tau_y_bot[x] < 0:
            R_CM_bot_Square_term = ma.pow(Tau_y_bot[x] / (2 * S_c), 2) + ma.pow(Tau_s_bot[x] / S_c, 2)
            R_CM_bot_term = (Tau_y_bot[x] / Y_c) * (ma.pow(Y_c / (2 * S_c), 2) - 1)
            P_bot_Hash = [R_CM_bot_Square_term, R_CM_bot_term, -1]
            r_MC_bot = np.roots(P_bot_Hash)
            H_MC_bot[x] = np.max(r_MC_bot)
        else:
            H_MC_bot[x] = 0

        if Tau_y_top[x] < 0:
            R_CM_top_Square_term = ma.pow(Tau_y_top[x] / (2 * S_c), 2) + ma.pow(Tau_s_top[x] / S_c, 2)
            R_CM_top_term = (Tau_y_top[x] / Y_c) * (ma.pow(Y_c / (2 * S_c), 2) - 1)
            P_top_Hash = [R_CM_top_Square_term, R_CM_top_term, -1]
            r_MC_top = np.roots(P_top_Hash)
            H_MC_top[x] = np.max(r_MC_top)
        else:
            H_MC_top[x] = 0

    if np.array_equal(On_Axis_Stress_top_Matrix, On_Axis_Stress_bot_Matrix) == True:
        R_FT_top_min = np.min(H_FT_top[np.nonzero(H_FT_top)])
        R_MT_top_min = np.min(H_MT_top[np.nonzero(H_MT_top)])

        Hashin_Arr_R = np.array([R_FT_top_min, R_MT_top_min])
        Min_R_Hashin = np.amin(Hashin_Arr_R)
        Index_min_R_Hashin = np.where(Hashin_Arr_R == np.amin(Hashin_Arr_R))
        if Index_min_R_Hashin == 0:
            Hashin_str = 'Failure Mode is Fiber Tension occurs on Top layer of ply'
        elif Index_min_R_Hashin == 1:
            Hashin_str = 'Failure Mode is Matrix Tension occurs on Top layer of ply '
        Failure_Lay_Hashin = Ply_Num
    else:
        H_FT_bot_lowest = np.min(H_FT_bot[np.nonzero(H_FT_bot)])
        H_index_FT_bot = np.where(H_FT_bot == H_FT_bot_lowest)
        H_FT_top_lowest = np.min(H_FT_top[np.nonzero(H_FT_top)])
        H_index_FT_top = np.where(H_FT_top == H_FT_top_lowest)

        H_FC_bot_lowest = np.min(H_FC_bot[np.nonzero(H_FC_bot)])
        H_index_FC_bot = np.where(H_FC_bot == H_FC_bot_lowest)
        H_FC_top_lowest = np.min(H_FC_top[np.nonzero(H_FC_top)])
        H_index_FC_top = np.where(H_FC_top == H_FC_top_lowest)

        H_MT_bot_lowest = np.min(H_MT_bot[np.nonzero(H_MT_bot)])
        H_index_MT_bot = np.where(H_MT_bot == H_MT_bot_lowest)
        H_MT_top_lowest = np.min(H_MT_top[np.nonzero(H_MT_top)])
        H_index_MT_top = np.where(H_MT_top == H_MT_top_lowest)

        H_MC_bot_lowest = np.min(H_MC_bot[np.nonzero(H_MC_bot)])
        H_index_MC_bot = np.where(H_MC_bot == H_MC_bot_lowest)
        H_MC_top_lowest = np.min(H_MC_top[np.nonzero(H_MC_top)])
        H_index_MC_top = np.where(H_MC_top == H_MC_top_lowest)

        H_R_lowest_Arr = np.array(
            [H_FT_bot_lowest, H_FT_top_lowest, H_FC_bot_lowest, H_FC_top_lowest, H_MT_bot_lowest, H_MT_top_lowest,
             H_MC_bot_lowest, H_MC_top_lowest])
        H_Minimum_R = np.amin(H_R_lowest_Arr)
        H_Index_Min_R = np.where(H_R_lowest_Arr == np.amin(H_R_lowest_Arr))

        if H_Index_Min_R == 0:
            Hashin_str = "Failure Mode is Fiber Tension occurs on Bottom layer of ply"
            Failure_Lay_Hashin = H_index_FT_bot
        elif H_Index_Min_R == 1:
            Hashin_str = "Failure Mode is Fiber Tension occurs on Top layer of ply"
            Failure_Lay_Hashin == H_index_FT_top
        elif H_Index_Min_R == 2:
            Hashin_str = "Failure Mode is Fiber Compression occurs on Bottom layer of ply"
            Failure_Lay_Hashin == H_index_FC_bot
        elif H_Index_Min_R == 3:
            Hashin_str = "Failure Mode is Fiber Compression occurs on Top layer of ply"
            Failure_Lay_Hashin == H_index_FC_top
        elif H_Index_Min_R == 4:
            Hashin_str = "Failure Mode is Matrix Tension occurs on Bottom layer of ply"
            Failure_Lay_Hashin = H_index_MT_bot
        elif H_Index_Min_R == 5:
            Hashin_str = "Failure Mode is Matrix Tension occurs on Top layer of ply"
            Failure_Lay_Hashin == H_index_MT_top
        elif H_Index_Min_R == 6:
            Hashin_str = "Failure Mode is Matrix Compression occurs on Bottom layer of ply"
            Failure_Lay_Hashin == H_index_MC_bot
        elif H_Index_Min_R == 7:
            Hashin_str = "Failure Mode is Matrix Compression occurs on Top layer of ply"
            Failure_Lay_Hashin == H_index_MC_top

    H_Max_Fail_Arr = np.array(
        [Layer_Arr, H_FT_bot, H_FC_bot, H_MT_bot, H_MC_bot, H_FT_top, H_FC_top, H_MT_top, H_MC_top])
    Minimum_R_N_Hash= np.multiply(Stress_Res_Arr, H_Minimum_R)
    Minimum_R_M_Hash = np.multiply(Mom_Res_Arr, H_Minimum_R)

    H_Max_Fail_Arr = pd.DataFrame(H_Max_Fail_Arr.transpose(), columns=['Ply Number',
                                                                       'FT_Bot',
                                                                       'FC_Bot',
                                                                       'MT_Bot',
                                                                       'MC_Bot',
                                                                       'FT_Top',
                                                                       'FC_Top',
                                                                       'MT_Top',
                                                                       'MC_Top'])
    #print(H_Max_Fail_Arr)
    #Failure_mode_list = np.array([[Failure_Layer,Minimum_R_N,Minimum_R_M],[Quad_Failure_Lay,Minimum_R_N_Quad,Minimum_R_M_Quad],[Failure_Lay_Hashin,Minimum_R_N_Hash],Minimum_R_M_Hash])
    return Max_Fail_Arr, Quad_Fail_Arr, H_Max_Fail_Arr



