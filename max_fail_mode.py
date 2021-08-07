import numpy as np
import pandas as pd
import math as ma


def max_failure(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,Stress_Res_Arr, Mom_Res_Arr):
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
            MT_bot[i] = Y_t / On_Axis_Stress_bot_Matrix[i, 1]
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
            MT_top[i] = Y_t / On_Axis_Stress_top_Matrix[i, 1]
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
        Index_Min_R = np.where(R_lowest_Arr == np.amin(R_lowest_Arr))[0]
        if Index_Min_R[0] == 0:
            str_max = "Failure Mode is Fiber Tension occurs on Top layer of ply"
        elif Index_Min_R[0] == 1:
            str_max = "Failure Mode is Matrix Tension occurs on Top layer of ply"
        elif Index_Min_R[0] == 2:
            str_max = "Failure Mode is Shear occurs on Top layer of ply"
        Failure_Layer = Ply_Num
    else:
        FT_bot_lowest = np.min(FT_bot[np.nonzero(FT_bot)])
        index_FT_bot = np.where(FT_bot == FT_bot_lowest)[0]
        FT_top_lowest = np.min(FT_top[np.nonzero(FT_top)])
        index_FT_top = np.where(FT_top == FT_top_lowest)[0]

        FC_bot_lowest = np.min(FC_bot[np.nonzero(FC_bot)])
        index_FC_bot = np.where(FC_bot == FC_bot_lowest)[0]
        FC_top_lowest = np.min(FC_top[np.nonzero(FC_top)])
        index_FC_top = np.where(FC_top == FC_top_lowest)[0]

        MT_bot_lowest = np.min(MT_bot[np.nonzero(MT_bot)])
        index_MT_bot = np.where(MT_bot == MT_bot_lowest)[0]
        MT_top_lowest = np.min(MT_top[np.nonzero(MT_top)])
        index_MT_top = np.where(MT_top == MT_top_lowest)[0]

        MC_bot_lowest = np.min(MC_bot[np.nonzero(MC_bot)])
        index_MC_bot = np.where(MC_bot == MC_bot_lowest)[0]
        MC_top_lowest = np.min(MC_top[np.nonzero(MC_top)])
        index_MC_top = np.where(MC_top == MC_top_lowest)[0]

        S_bot_lowest = np.min(S_bot[np.nonzero(S_bot)])
        index_S_bot = np.where(S_bot == S_bot_lowest)[0]
        S_top_lowest = np.min(S_top[np.nonzero(S_top)])
        index_S_top = np.where(S_top == S_top_lowest)[0]

        R_lowest_Arr = np.array(
            [FT_bot_lowest, FT_top_lowest, FC_bot_lowest, FC_top_lowest, MT_bot_lowest, MT_top_lowest, MC_bot_lowest,
             MC_top_lowest, S_bot_lowest, S_top_lowest])
        Minimum_R = np.amin(R_lowest_Arr)
        Index_Min_R = np.where(R_lowest_Arr == np.amin(R_lowest_Arr))[0]

        #print('Index_Min_R[0]=',Index_Min_R[0])
        #print('R_lowest_Arr = ', R_lowest_Arr)

        #print('Index_Min_R[0] =', Index_Min_R[0])
        if Index_Min_R[0] == 0:
            str_max = "Failure Mode is Fiber Tension occurs on Bottom layer of ply"
            Failure_Layer = index_FT_bot[0]
        elif Index_Min_R[0] == 1:
            str_max = "Failure Mode is Fiber Tension occurs on Top layer of ply"
            Failure_Layer = index_FT_top[0]
        elif Index_Min_R[0] == 2:
            str_max = "Failure Mode is Fiber Compression occurs on Bottom layer of ply"
            Failure_Layer = index_FC_bot[0]
        elif Index_Min_R[0] == 3:
            str_max = "Failure Mode is Fiber Compression occurs on Top layer of ply"
            Failure_Layer = index_FC_top[0]
        elif Index_Min_R[0] == 4:
            str_max = "Failure Mode is Matrix Tension occurs on Bottom layer of ply"
            Failure_Layer = index_MT_bot[0]
        elif Index_Min_R[0] == 5:
            str_max = "Failure Mode is Matrix Tension occurs on Top layer of ply"
            Failure_Layer = index_MT_top[0]
        elif Index_Min_R[0] == 6:
            str_max = "Failure Mode is Matrix Compression occurs on Bottom layer of ply"
            Failure_Layer = index_MC_bot[0]
        elif Index_Min_R[0] == 7:
            str_max = "Failure Mode is Matrix Compression occurs on Top layer of ply"
            Failure_Layer = index_MC_top[0]
        elif Index_Min_R[0] == 8:
            str_max = "Failure Mode is Shear occurs on Bottom layer of ply"
            Failure_Layer = index_S_bot[0]
        elif Index_Min_R[0] == 9:
            str_max = "Failure Mode is Shear occurs on top layer of ply"
            Failure_Layer = index_S_top[0]

    # Minimum_R_N = Minimum_R * Stress_Res_Arr

    #print('The load vectors which would cause failures are: ')
    Minimum_R_N = np.multiply(Stress_Res_Arr, Minimum_R)
    Minimum_R_M = np.multiply(Mom_Res_Arr, Minimum_R)
    #print(Minimum_R_N, '[N]', Minimum_R_M, '[N*M]')

    #print("Failure_Layer =", Failure_Layer)

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
    Max_Fail_list = [str_max,Failure_Layer+1,Minimum_R,Minimum_R_N,Minimum_R_M]

    return Max_Fail_Arr,Max_Fail_list
