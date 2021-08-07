import numpy as np
import pandas as pd
import math as ma

def hashin_failure(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,Stress_Res_Arr, Mom_Res_Arr):
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
    Tau_x_bot = On_Axis_Stress_bot_Matrix[:, 0]
    Tau_y_bot = On_Axis_Stress_bot_Matrix[:, 1]
    Tau_s_bot = On_Axis_Stress_bot_Matrix[:, 2]

    Tau_x_top = On_Axis_Stress_top_Matrix[:, 0]
    Tau_y_top = On_Axis_Stress_top_Matrix[:, 1]
    Tau_s_top = On_Axis_Stress_top_Matrix[:, 2]

    Q_term = ma.pow((Y_c / (2 * S_c)), 2) - 1
    for x in range(Ply_Num):
        if Tau_x_bot[x] > 0:
            H_FT_bot[x] = ma.pow((1 / ma.pow(Tau_x_bot[x] / X_t, 2) + ma.pow(Tau_s_bot[x] / S_c, 2)), 0.5)
            H_FC_bot[x] = 0
        else:
            H_FT_bot[x] = 0
            H_FC_bot[x] = X_c / abs(Tau_x_bot[x])
        if Tau_x_top[x] > 0:
            H_FT_top[x] = ma.pow((1 / ma.pow(Tau_x_top[x] / X_t, 2) + ma.pow(Tau_s_top[x] / S_c, 2)), 0.5)
            H_FC_top[x] = 0
        else:
            H_FT_top[x] = 0
            H_FC_top[x] = X_c / abs(Tau_x_top[x])

        if Tau_y_bot[x] > 0:
            H_MT_bot[x] = ma.pow((1 / ma.pow(Tau_y_bot[x] / Y_t, 2) + ma.pow(Tau_s_bot[x] / S_c, 2)), 0.5)
        else:
            H_MT_bot[x] = 0
        if Tau_y_top[x] > 0:
            H_MT_top[x] = ma.pow((1 / ma.pow(Tau_y_top[x] / Y_t, 2) + ma.pow(Tau_s_top[x] / S_c, 2)), 0.5)
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
        Index_min_R_Hashin = np.where(Hashin_Arr_R == np.amin(Hashin_Arr_R))[0]
        if Index_min_R_Hashin[0] == 0:
            Hashin_str = 'Failure Mode is Fiber Tension occurs on Top layer of ply'
        elif Index_min_R_Hashin[0] == 1:
            Hashin_str = 'Failure Mode is Matrix Tension occurs on Top layer of ply '
        Failure_Lay_Hashin = Ply_Num
    else:
        H_FT_bot_lowest = np.min(H_FT_bot[np.nonzero(H_FT_bot)])
        H_index_FT_bot = np.where(H_FT_bot == H_FT_bot_lowest)[0]
        H_FT_top_lowest = np.min(H_FT_top[np.nonzero(H_FT_top)])
        H_index_FT_top = np.where(H_FT_top == H_FT_top_lowest)[0]

        H_FC_bot_lowest = np.min(H_FC_bot[np.nonzero(H_FC_bot)])
        H_index_FC_bot = np.where(H_FC_bot == H_FC_bot_lowest)[0]
        H_FC_top_lowest = np.min(H_FC_top[np.nonzero(H_FC_top)])
        H_index_FC_top = np.where(H_FC_top == H_FC_top_lowest)[0]

        H_MT_bot_lowest = np.min(H_MT_bot[np.nonzero(H_MT_bot)])
        H_index_MT_bot = np.where(H_MT_bot == H_MT_bot_lowest)[0]
        H_MT_top_lowest = np.min(H_MT_top[np.nonzero(H_MT_top)])
        H_index_MT_top = np.where(H_MT_top == H_MT_top_lowest)[0]

        H_MC_bot_lowest = np.min(H_MC_bot[np.nonzero(H_MC_bot)])
        H_index_MC_bot = np.where(H_MC_bot == H_MC_bot_lowest)[0]
        H_MC_top_lowest = np.min(H_MC_top[np.nonzero(H_MC_top)])
        H_index_MC_top = np.where(H_MC_top == H_MC_top_lowest)[0]

        H_R_lowest_Arr = np.array(
            [H_FT_bot_lowest, H_FT_top_lowest, H_FC_bot_lowest, H_FC_top_lowest, H_MT_bot_lowest, H_MT_top_lowest,
             H_MC_bot_lowest, H_MC_top_lowest])
        H_Minimum_R = np.amin(H_R_lowest_Arr)
        H_Index_Min_R = np.where(H_R_lowest_Arr == np.amin(H_R_lowest_Arr))[0]

        if H_Index_Min_R[0] == 0:
            Hashin_str = "Failure Mode is Fiber Tension occurs on Bottom layer of ply"
            Failure_Lay_Hashin = H_index_FT_bot[0]
        elif H_Index_Min_R[0] == 1:
            Hashin_str = "Failure Mode is Fiber Tension occurs on Top layer of ply"
            Failure_Lay_Hashin = H_index_FT_top[0]
        elif H_Index_Min_R[0] == 2:
            Hashin_str = "Failure Mode is Fiber Compression occurs on Bottom layer of ply"
            Failure_Lay_Hashin = H_index_FC_bot[0]
        elif H_Index_Min_R[0] == 3:
            Hashin_str = "Failure Mode is Fiber Compression occurs on Top layer of ply"
            Failure_Lay_Hashin = H_index_FC_top[0]
        elif H_Index_Min_R[0] == 4:
            Hashin_str = "Failure Mode is Matrix Tension occurs on Bottom layer of ply"
            Failure_Lay_Hashin = H_index_MT_bot[0]
        elif H_Index_Min_R[0] == 5:
            Hashin_str = "Failure Mode is Matrix Tension occurs on Top layer of ply"
            Failure_Lay_Hashin = H_index_MT_top[0]
        elif H_Index_Min_R[0] == 6:
            Hashin_str = "Failure Mode is Matrix Compression occurs on Bottom layer of ply"
            Failure_Lay_Hashin = H_index_MC_bot[0]
        elif H_Index_Min_R[0] == 7:
            Hashin_str = "Failure Mode is Matrix Compression occurs on Top layer of ply"
            Failure_Lay_Hashin = H_index_MC_top[0]

    Layer_Arr = np.arange(1, Ply_Num + 1)

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
    Hashin_Fail_list = [Hashin_str, Failure_Lay_Hashin+1, H_Minimum_R , Minimum_R_N_Hash, Minimum_R_M_Hash]
    return H_Max_Fail_Arr,Hashin_Fail_list