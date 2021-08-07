
import numpy as np
import pandas as pd
import math as ma
import sys
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Two_d_ADI_Heat import ADI_Heat
from max_fail_mode import max_failure
from quad_fail_mode import quad_failure
from hashin_fail_mode import hashin_failure
from flask import Flask, render_template, request
import base64

from io import BytesIO

app = Flask(__name__)
if __name__ == "__main__":
    app.run(debug=True)
@app.route('/')
def main():
    return render_template('app.html')
'''

'''
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/ADI_heat',endpoint = 'Second Page')
def main():
    return render_template('ADI.html')

@app.route('/Send_1',methods=['POST'],endpoint = 'Temp_Graph Page')
def ADI():
    if request.method == 'POST':

        alpha = request.form['alpha']
        t = request.form['t']
        nt = request.form['nt']
        x = request.form['x']
        Mesh_Grid_Point = request.form['Mesh_Grid_Point']
        T1 = request.form['T1']
        T2 = request.form['T2']
        T3 = request.form['T3']
        T4 = request.form['T4']
        T5 = request.form['T5']

        X_1, Y_1, Total_Temp_Profile,Total_Temp_Profile_mid = ADI_Heat(alpha, t, nt, x, Mesh_Grid_Point, T1, T2, T3, T4, T5)
        #Plotting 3D Surface Plot

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1,2,1,projection ='3d')
        surf = ax.plot_surface(X_1, Y_1, Total_Temp_Profile, cmap=cm.jet, rstride = 1, cstride = 1)
        ax.set_title('ADI Method to solve for the\n Temperature Distribution of a \nsquared plate')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('Temperature', labelpad=20)
        fig.colorbar(surf)

        ax = fig.add_subplot(1,2,2,projection = '3d')
        ax.plot_surface(X_1,Y_1,Total_Temp_Profile, cmap=cm.jet, rstride = 1, cstride = 1)
        ax.view_init(azim=0, elev=90)
        ax.set_title('ADI Method to solve for the \nTemperature Distribution of \na squared plate resulted temperature \nin 2D')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('Temperature', labelpad=20)
        fig.colorbar(surf)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data_1 = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data_1}'/>"
#Direct to Homepage for Composite Mat Compute

@app.route('/send', methods=['POST'],endpoint = 'Home page')
def send():
    if request.method == 'POST':

        Mat_name = request.form['operation']
        Mat_Geo_Input = request.form['Mat_layup']
        symmetric_check = request.form['Sym_checker']
        Off_Axis_Stress_Input = request.form['Off_Axis_Stress_Input']
        honey_comb = request.form['honey_comb']
        Stress_Res_Vec = request.form['Stress_Res_Vec']
        Mom_Res_Input = request.form['Mom_Res_Input']

        print(Mom_Res_Input, file = sys.stderr)

        url = "https://raw.githubusercontent.com/jasonzhou1997/Composite-Mat/main/Material%20Prop%20from%20Appendix%20C.csv"
        Material_Prop = pd.read_csv(url)
        # print(Material_Prop)
        F_M_names = pd.DataFrame(Material_Prop, columns=['Fiber Matrix'])
        print(F_M_names,file = sys.stderr)

        if Mat_name == 'Mat_1':
            Row_num = 0
        elif Mat_name == 'Mat_2':
            Row_num = 1
        elif Mat_name == 'Mat_3':
            Row_num = 2
        elif Mat_name == 'Mat_4':
            Row_num = 3
        elif Mat_name == 'Mat_5':
            Row_num = 4
        # print(Row_num)

        Sel_Mat_Prop = Material_Prop.iloc[Row_num]
        # print(Sel_Mat_Prop)

        Material_name = Sel_Mat_Prop["Material Name"][0:None]
        Fiber_Matrix = Sel_Mat_Prop["Fiber Matrix"][0:None]
        E_x = Sel_Mat_Prop["Ex [Gpa]"] * 1000
        E_y = Sel_Mat_Prop["Ey [Gpa]"] * 1000
        nu_over_x = Sel_Mat_Prop["nu/x"]
        Es = Sel_Mat_Prop["Es [Gpa]"] * 1000
        X_t = Sel_Mat_Prop["Xt [Mpa]"]
        X_c = Sel_Mat_Prop["Xc [Mpa]"]
        Y_t = Sel_Mat_Prop["Yt [Mpa]"]
        Y_c = Sel_Mat_Prop["Yc [Mpa]"]
        S_c = Sel_Mat_Prop["Sc [Mpa]"]
        h_o = Sel_Mat_Prop["ho [m]"]

        nu_over_y = nu_over_x * E_y / E_x
        headings = ("Name", "Units", "Value")
        data_1 = (
            ("Material_name","[-]" ,Material_name),
            ("Fiber_Matrix", "[-]", Fiber_Matrix),
            ("E_x", "[MPa]",E_x),
            ("E_y", "[MPa}", E_y),
            ("nu/x", "[-]", nu_over_x),
            ("E_s", "[MPa]", Es),
            ("X_t", "[MPa]", X_t),
            ("X_c", "[MPa]", X_c),
            ("Y_t", "[MPa]", Y_t),
            ("Y_c", "[MPa]", Y_c),
            ("S_c", "[MPa]", S_c),
            ("h_o", "[m]", h_o)
        )
        '''
        print("Material Name is ", Material_name,",", "Fiber Matrix is", Fiber_Matrix, ", Material Property is shown below:")
        print("Ex = ", E_x, "[Mpa]")
        print("Ey = ", E_y, "[Mpa]")
        print("nu/x = ",nu_over_x)
        print("nu/y = ",nu_over_y)
        print("Es = ", Es, "[Mpa]")
        print("Xt = ", X_t, "[Mpa]")
        print("Xc = ", X_c, "[Mpa]")
        print("Yt = ", Y_t, "[Mpa]")
        print("Yc = ", Y_c, "[Mpa]")
        print("Sc = ", S_c, "[Mpa]")
        print("ho (Thickness of a ply) = ", h_o, "[m]")
        '''
        """Calculate on axis Q and S Matrix:"""

        m = 1 / (1 - nu_over_x * nu_over_y)
        print(m, file = sys.stderr)
        # print("m = ",m)
        Q_on_axis = np.array([[m * E_x, m * E_x * nu_over_y, 0], [m * nu_over_x * E_y, m * E_y, 0], [0, 0, Es]])
        #return render_template('app.html', Q_on_axis=Q_on_axis)
        print("On Axis Q Matrix [Mpa]: ",file = sys.stderr)
        print(Q_on_axis, file = sys.stderr)
        S_on_axis = np.linalg.inv(Q_on_axis)
        print(S_on_axis, file = sys.stderr)
        #return render_template('app.html', Q_on_axis=Q_on_axis)
        #return render_template('app.html', S_on_axis=S_on_axis,Q_on_axis=Q_on_axis)
        #print("On Axis S Matrix [1/Mpa]: ")
        #print(S_on_axis)

        """Calculate U Matrices for Q and S:"""
        U_1_Q = 1 / 8 * (3 * Q_on_axis[0, 0] + 3 * Q_on_axis[1, 1] + 2 * Q_on_axis[0, 1] + 4 * Q_on_axis[2, 2])
        U_2_Q = 1 / 2 * (Q_on_axis[0, 0] - Q_on_axis[1, 1])
        U_3_Q = 1 / 8 * (Q_on_axis[0, 0] + Q_on_axis[1, 1] - 2 * Q_on_axis[0, 1] - 4 * Q_on_axis[2, 2])
        U_4_Q = 1 / 8 * (Q_on_axis[0, 0] + Q_on_axis[1, 1] + 6 * Q_on_axis[0, 1] - 4 * Q_on_axis[2, 2])
        U_5_Q = 1 / 8 * (Q_on_axis[0, 0] + Q_on_axis[1, 1] - 2 * Q_on_axis[0, 1] + 4 * Q_on_axis[2, 2])

        U_1_S = 1 / 8 * (3 * S_on_axis[0, 0] + 3 * S_on_axis[1, 1] + 2 * S_on_axis[0, 1] + S_on_axis[2, 2])
        U_2_S = 1 / 2 * (S_on_axis[0, 0] - S_on_axis[1, 1])
        U_3_S = 1 / 8 * (S_on_axis[0, 0] + S_on_axis[1, 1] - 2 * S_on_axis[0, 1] - S_on_axis[2, 2])
        U_4_S = 1 / 8 * (S_on_axis[0, 0] + S_on_axis[1, 1] + 6 * S_on_axis[0, 1] - S_on_axis[2, 2])
        U_5_S = 1 / 2 * (S_on_axis[0, 0] + S_on_axis[1, 1] - 2 * S_on_axis[0, 1] + S_on_axis[2, 2])

        Geo_Str_list = Mat_Geo_Input.split("/")
        Geo_Str_Arr = np.array(Geo_Str_list)
        Geo_Str_Arr = list(map(float, Geo_Str_Arr))
        if symmetric_check == 'Y':
          Geo_flip_Arr = np.flip(Geo_Str_Arr)
          Geo_Sym_Arr = np.concatenate([Geo_Str_Arr,Geo_flip_Arr])
          Ply_Num = np.size(Geo_Sym_Arr)
          #sum = Geo_Sym_Arr
          Ply_Num_wo_Sym = Ply_Num/2
          #return render_template('app.html', Geo_Sym_Arr = Geo_Sym_Arr)
          #return render_template('app.html', Geo_Sym_Arr=Geo_Sym_Arr)

        elif symmetric_check =='N':
          Geo_Sym_Arr = Geo_Str_Arr
          Ply_Num = np.size(Geo_Sym_Arr)
          Ply_Num_wo_Sym = Ply_Num
          #return render_template('app.html', Geo_Sym_Arr=Geo_Sym_Arr)
        
        Ply_Num_wo_Sym = int(Ply_Num_wo_Sym)
        S11 = np.zeros(Ply_Num_wo_Sym)
        S22 = np.zeros(Ply_Num_wo_Sym)
        S12 = np.zeros(Ply_Num_wo_Sym)
        S66 = np.zeros(Ply_Num_wo_Sym)
        S16 = np.zeros(Ply_Num_wo_Sym)
        S26 = np.zeros(Ply_Num_wo_Sym)
        Q11 = np.zeros(Ply_Num_wo_Sym)
        Q22 = np.zeros(Ply_Num_wo_Sym)
        Q12 = np.zeros(Ply_Num_wo_Sym)
        Q66 = np.zeros(Ply_Num_wo_Sym)
        Q16 = np.zeros(Ply_Num_wo_Sym)
        Q26 = np.zeros(Ply_Num_wo_Sym)

        for x in range(Ply_Num_wo_Sym + 1):
            S11[x - 1] = U_1_S + U_2_S * ma.cos(2 * ma.radians(Geo_Sym_Arr[x - 1])) + U_3_S * ma.cos(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            S22[x - 1] = U_1_S - U_2_S * ma.cos(2 * ma.radians(Geo_Sym_Arr[x - 1])) + U_3_S * ma.cos(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            S12[x - 1] = U_4_S - U_3_S * ma.cos(4 * ma.radians(Geo_Sym_Arr[x - 1]))
            S66[x - 1] = U_5_S - 4 * U_3_S * ma.cos(4 * ma.radians(Geo_Sym_Arr[x - 1]))
            S16[x - 1] = U_2_S * ma.sin(2 * ma.radians(Geo_Sym_Arr[x - 1])) + 2 * U_3_S * ma.sin(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            S26[x - 1] = U_2_S * ma.sin(2 * ma.radians(Geo_Sym_Arr[x - 1])) - 2 * U_3_S * ma.sin(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            # print(S11)
            Q11[x - 1] = U_1_Q + U_2_Q * ma.cos(2 * ma.radians(Geo_Sym_Arr[x - 1])) + U_3_Q * ma.cos(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            Q22[x - 1] = U_1_Q - U_2_Q * ma.cos(2 * ma.radians(Geo_Sym_Arr[x - 1])) + U_3_Q * ma.cos(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            Q12[x - 1] = U_4_Q - U_3_Q * ma.cos(4 * ma.radians(Geo_Sym_Arr[x - 1]))
            Q66[x - 1] = U_5_Q - U_3_Q * ma.cos(4 * ma.radians(Geo_Sym_Arr[x - 1]))
            Q16[x - 1] = 1 / 2 * U_2_Q * ma.sin(2 * ma.radians(Geo_Sym_Arr[x - 1])) + U_3_Q * ma.sin(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))
            Q26[x - 1] = 1 / 2 * U_2_Q * ma.sin(2 * ma.radians(Geo_Sym_Arr[x - 1])) - U_3_Q * ma.sin(
                4 * ma.radians(Geo_Sym_Arr[x - 1]))

        # print(S11[1])
        for x in range(Ply_Num_wo_Sym):
            # print('Layer Number = ', x+1)
            # print('Ply Orientation = ', Geo_Sym_Arr[x],'[Degrees]' )
            Off_Axis_S_Matrix = np.array([[S11[x], S12[x], S16[x]], [S12[x], S22[x], S26[x]], [S16[x], S26[x], S66[x]]])
            # print('Off Axis S Matrix:[1/Mpa]', Off_Axis_S_Matrix)
            Off_Axis_Q_Matrix = np.array([[Q11[x], Q12[x], Q16[x]], [Q12[x], Q22[x], Q26[x]], [Q16[x], Q26[x], Q66[x]]])
            # print('Off Axis Q Matrix:[Mpa]', Off_Axis_Q_Matrix)

        """Input the off-axis stress vector (P20) to
        Calculate on axis stress & off axis strain & on axis strain
        """

        # Off_Axis_Stress_Input = input('Enter the off-axis stress, in [Mpa]: ')
        Off_Axis_Stress_list = Off_Axis_Stress_Input.split("/")
        Off_Axis_Stress_Arr = np.array(Off_Axis_Stress_list)
        Off_Axis_Stress_Arr = list(map(float, Off_Axis_Stress_Arr))

        Tau_1 = Off_Axis_Stress_Arr[0]
        Tau_2 = Off_Axis_Stress_Arr[1]
        Tau_6 = Off_Axis_Stress_Arr[2]

        p = 0.5 * (Tau_1 + Tau_2);
        q = 0.5 * (Tau_1 - Tau_2);
        r = Tau_6;
        #return render_template('app.html', r_html = r)
        # print(Tau_1,Tau_2,Tau_6)
        # print(Tau_1 + Tau_2)

        for x in range(Ply_Num_wo_Sym):
            # Table 3.2
            On_Axis_Stress_Arr = np.array(
                [p + q * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) + r * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])), \
                 p - q * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) - r * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])), \
                 -1 * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])) * q + r * ma.cos(2 * ma.radians(Geo_Sym_Arr[x]))])
            # print('On Axis Stress Matrix: ', '\u03C3_x = ', On_Axis_Stress_Arr[0],',', '\u03C3_y = ', On_Axis_Stress_Arr[1], ',', '\u03C3_s = ', On_Axis_Stress_Arr[2])
            # Table 3.10
            Off_Axis_Strain_Arr = np.array([Tau_1 * S11[x] + Tau_2 * S12[x] + Tau_6 * S16[x], \
                                            Tau_1 * S12[x] + Tau_2 * S22[x] + Tau_6 * S26[x], \
                                            Tau_1 * S16[x] + Tau_2 * S26[x] + Tau_6 * S66[x]])


            #print('Off Axis Strain Matrix: ', '\u03B5_1 = ', Off_Axis_Strain_Arr[0], ',', '\u03B5_2 = ', Off_Axis_Strain_Arr[1],',', '\u03B5_6 = ', Off_Axis_Strain_Arr[2])
            p_strain = 0.5 * (Off_Axis_Strain_Arr[0] + Off_Axis_Strain_Arr[1])
            q_strain = 0.5 * (Off_Axis_Strain_Arr[0] - Off_Axis_Strain_Arr[1])
            r_strain = 0.5 * Off_Axis_Strain_Arr[2]

            # Table 3.5
            On_Axis_Strain_Arr = np.array([p_strain + q_strain * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) + r_strain * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])), \
                 p_strain - q_strain * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) - r_strain * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])), \
                 -2 * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])) * q_strain + r_strain * 2 * ma.cos(2 * ma.radians(Geo_Sym_Arr[x]))])
            # print('On Axis Strain Matrix: ', '\u03B5_x = ', On_Axis_Strain_Arr[0], ',', '\u03B5_y = ', On_Axis_Strain_Arr[1], ',','\u03B5_s = ', On_Axis_Strain_Arr[2])

        """Include honeycomb thickness, calculating the overall in-plane modulus and in-plane compliance for a given laminate ([A] and [a] matrix)"""

        # honey_comb = input('Enter the Honeycomb Thickness Z_c: ')
        if symmetric_check == 'Y':
            Total_Ply_Thickness = Ply_Num_wo_Sym * h_o * 2
        else:
            Total_Ply_Thickness = Ply_Num_wo_Sym * h_o
        Z_c = float(honey_comb)

        Sum_of_Ply_Thickness = Total_Ply_Thickness + Z_c

        # Calculate [A] matrix
        v = 2 * h_o / Total_Ply_Thickness

        V1_Star = 0
        V2_Star = 0
        V3_Star = 0
        V4_Star = 0

        for x in range(Ply_Num_wo_Sym):
            V1_Star = V1_Star + ma.cos(2 * ma.radians(Geo_Sym_Arr[x]))
            V2_Star = V2_Star + ma.cos(4 * ma.radians(Geo_Sym_Arr[x]))
            V3_Star = V3_Star + ma.sin(2 * ma.radians(Geo_Sym_Arr[x]))
            V4_Star = V4_Star + ma.sin(4 * ma.radians(Geo_Sym_Arr[x]))

        V1_Star = V1_Star * v
        V2_Star = V2_Star * v
        V3_Star = V3_Star * v
        V4_Star = V4_Star * v
        #print(V1_Star, V2_Star, V3_Star, V4_Star)
        A11 = Total_Ply_Thickness * (U_1_Q + U_2_Q * V1_Star + V2_Star * U_3_Q)
        A22 = Total_Ply_Thickness * (U_1_Q - U_2_Q * V1_Star + V2_Star * U_3_Q)
        A12 = Total_Ply_Thickness * (U_4_Q - V2_Star * U_3_Q)
        A66 = Total_Ply_Thickness * (U_5_Q - V2_Star * U_3_Q)
        A16 = Total_Ply_Thickness * (0.5 * V3_Star * U_2_Q + V4_Star * U_3_Q)
        A26 = Total_Ply_Thickness * (0.5 * V3_Star * U_2_Q - V4_Star * U_3_Q)

        A_Arr = np.array([[A11, A12, A16], [A12, A22, A26], [A16, A26, A66]])
        #print('In plane Modulus [A],[MN/m]:', A_Arr, file = sys.stderr)
        a_Arr = np.linalg.inv(A_Arr)
        # print('Off-axis compliance [a],[m/MN]:', a_Arr)
        #return render_template('app.html', A_Arr = A_Arr)
        #return render_template('app.html', a_Arr = a_Arr)
    

        # Input Stress Resultant Vector
        # Stress_Res_Vec = input('Enter the applied stress vector N1/N2/N6, in [N/m]')

        Stress_Res_list = Stress_Res_Vec.split("/")
        Stress_Res_Arr = np.array(Stress_Res_list)
        Stress_Res_Arr = list(map(float, Stress_Res_Arr))

        N_1 = Stress_Res_Arr[0] / 1000000
        N_2 = Stress_Res_Arr[1] / 1000000
        N_6 = Stress_Res_Arr[2] / 1000000

        # Table 4.2
        Off_Axis_Strain_Res_Strain = np.array([[a_Arr[0, 0] * N_1 + a_Arr[0, 1] * N_2 + a_Arr[0, 2] * N_6], \
                                               [a_Arr[1, 0] * N_1 + a_Arr[1, 1] * N_2 + a_Arr[1, 2] * N_6], \
                                               [a_Arr[2, 0] * N_1 + a_Arr[2, 1] * N_2 + a_Arr[2, 2] * N_6]])
        # print('In-plane strain of symmetric laminates [m/N]:', Off_Axis_Strain_Res_Strain)
        # Equ 3.21 Strain Transformation

        Eps_1 = Off_Axis_Strain_Res_Strain[0, 0]
        Eps_2 = Off_Axis_Strain_Res_Strain[1, 0]
        Eps_6 = Off_Axis_Strain_Res_Strain[2, 0]

        p_strain = 0.5 * (Eps_1 + Eps_2)
        q_strain = 0.5 * (Eps_1 - Eps_2)
        r_strain = 0.5 * Eps_6

        # Equ 2.12
        Q_xx = m * E_x
        Q_yy = m * E_y
        Q_yx = m * nu_over_x * E_y
        Q_xy = m * nu_over_y * E_x
        Q_ss = m * Es

        for x in range(Ply_Num):
            # print('layer Number =', x + 1)
            # print('Ply Orientation = ',Geo_Sym_Arr[x],'[Degrees]' )
            On_Axis_Strain_2 = np.array(
                [[p_strain + q_strain * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) + r_strain * ma.sin(
                    2 * ma.radians(Geo_Sym_Arr[x]))], \
                 [p_strain - q_strain * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) - r_strain * ma.sin(
                     2 * ma.radians(Geo_Sym_Arr[x]))], \
                 [-2 * q_strain * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])) + 2 * r_strain * ma.cos(
                     2 * ma.radians(Geo_Sym_Arr[x]))]])
            # print('On Axis Ply Strain:', On_Axis_Strain_2)
            On_Axis_Stress_2 = np.array([[Q_xx * On_Axis_Strain_2[0, 0] + Q_xy * On_Axis_Strain_2[1, 0]],
                                         [Q_yx * On_Axis_Strain_2[0, 0] + Q_yy * On_Axis_Strain_2[1, 0]],
                                         [Q_ss * On_Axis_Strain_2[2, 0]]])
            # print('On Axis Ply Stress [Mpa]:', On_Axis_Stress_2)
        # print('Geo_Sym_Arr', Geo_Sym_Arr)
        # Calculate [D] matrix

        # Mom_Res_Input = input('Enter the applied moment resultant vector M1/M2/M6, in [N*m]: ')

        Mom_Res_list = Mom_Res_Input.split("/")
        Mom_Res_Arr = np.array(Mom_Res_list)
        Mom_Res_Arr = list(map(float, Mom_Res_Arr))
        # print(Mom_Res_Arr[0])

        Z_c_star = 2 * Z_c / (0.000125 * Ply_Num + 2 * Z_c)
        h_star = (h_o * Ply_Num + 2 * Z_c) ** 3 / 12 * (1 - Z_c_star ** 3)
        # print('Z_c_star:', Z_c_star,'h_star',h_star)
        V1 = 0
        V2 = 0
        V3 = 0
        V4 = 0

        for x in range(Ply_Num_wo_Sym):
            # x starts from 0 to 9

            Z_x = (x + 1) * h_o + Z_c
            Z_x_minus_1 = (x) * h_o + Z_c
            V1 = V1 + 2 / 3 * ma.cos(ma.radians(2 * Geo_Sym_Arr[Ply_Num_wo_Sym - (x + 1)])) * (
                        Z_x ** 3 - Z_x_minus_1 ** 3)
            # print(Geo_Sym_Arr[Ply_Num_wo_Sym - x - 1])
            V2 = V2 + 2 / 3 * ma.cos(ma.radians(4 * Geo_Sym_Arr[Ply_Num_wo_Sym - (x + 1)])) * (
                        Z_x ** 3 - Z_x_minus_1 ** 3)
            V3 = V3 + 2 / 3 * ma.sin(ma.radians(2 * Geo_Sym_Arr[Ply_Num_wo_Sym - (x + 1)])) * (
                        Z_x ** 3 - Z_x_minus_1 ** 3)
            V4 = V4 + 2 / 3 * ma.sin(ma.radians(4 * Geo_Sym_Arr[Ply_Num_wo_Sym - (x + 1)])) * (
                        Z_x ** 3 - Z_x_minus_1 ** 3)
            # print(Z_x,Z_x_minus_1, V1,V2,V3,V4)

        D11 = U_1_Q * h_star + U_2_Q * V1 + U_3_Q * V2
        D22 = U_1_Q * h_star - U_2_Q * V1 + U_3_Q * V2
        D12 = U_4_Q * h_star - U_3_Q * V2
        D66 = U_5_Q * h_star - U_3_Q * V2
        D16 = 0.5 * V3 * U_2_Q + V4 * U_3_Q
        D26 = 0.5 * V3 * U_2_Q - V4 * U_3_Q

        D_Arr = np.array([[D11, D12, D16], [D12, D22, D26], [D16, D26, D66]])

        # print('Flexural Modulus [D]: [M/m^2]:', D_Arr)
        d_Arr = np.linalg.inv(D_Arr)
        # print('Flexural Modulus Compliance [d]: [m^2/M]', d_Arr)

        K1 = (d_Arr[0, 0] * Mom_Res_Arr[0] + d_Arr[0, 1] * Mom_Res_Arr[1] + d_Arr[0, 2] * Mom_Res_Arr[2]) * ma.pow(10,
                                                                                                                   -6)
        K2 = (d_Arr[0, 1] * Mom_Res_Arr[0] + d_Arr[1, 1] * Mom_Res_Arr[1] + d_Arr[1, 2] * Mom_Res_Arr[2]) * ma.pow(10,
                                                                                                                   -6)
        K6 = (d_Arr[0, 2] * Mom_Res_Arr[0] + d_Arr[2, 1] * Mom_Res_Arr[1] + d_Arr[2, 2] * Mom_Res_Arr[2]) * ma.pow(10,
                                                                                                                   -6)

        K_Arr = np.array([K1, K2, K6])
        E_bot_1_upper = np.zeros(1)
        E_top_1_upper = np.zeros(1)
        E_bot_2_upper = np.zeros(1)
        E_top_2_upper = np.zeros(1)
        E_bot_6_upper = np.zeros(1)
        E_top_6_upper = np.zeros(1)

        E_bot_1_lower = np.zeros(1)
        E_top_1_lower = np.zeros(1)
        E_bot_2_lower = np.zeros(1)
        E_top_2_lower = np.zeros(1)
        E_bot_6_lower = np.zeros(1)
        E_top_6_lower = np.zeros(1)

        for x in range(Ply_Num_wo_Sym):
            Z_bot = Z_c + x * h_o
            Z_top = Z_c + (x + 1) * h_o

            Z_bot_lower = -Z_c - (x + 1) * h_o
            Z_top_lower = -Z_c - x * h_o

            # np.concatenate((arr1, arr2))

            E_bot_1_upper = np.append(E_bot_1_upper, K1 * Z_bot)
            E_top_1_upper = np.append(E_top_1_upper, K1 * Z_top)
            E_bot_2_upper = np.append(E_bot_2_upper, K2 * Z_bot)
            E_top_2_upper = np.append(E_top_2_upper, K2 * Z_top)
            E_bot_6_upper = np.append(E_bot_6_upper, K6 * Z_bot)
            E_top_6_upper = np.append(E_top_6_upper, K6 * Z_top)

            E_bot_1_lower = np.append(E_bot_1_lower, K1 * Z_bot_lower)
            E_top_1_lower = np.append(E_top_1_lower, K1 * Z_top_lower)
            E_bot_2_lower = np.append(E_bot_2_lower, K2 * Z_bot_lower)
            E_top_2_lower = np.append(E_top_2_lower, K2 * Z_top_lower)
            E_bot_6_lower = np.append(E_bot_6_lower, K6 * Z_bot_lower)
            E_top_6_lower = np.append(E_top_6_lower, K6 * Z_top_lower)

        E_bot_1_upper = E_bot_1_upper[1:]
        E_top_1_upper = E_top_1_upper[1:]
        E_bot_2_upper = E_bot_2_upper[1:]
        E_top_2_upper = E_top_2_upper[1:]
        E_bot_6_upper = E_bot_6_upper[1:]
        E_top_6_upper = E_top_6_upper[1:]

        E_bot_1_lower = E_bot_1_lower[1:]
        E_top_1_lower = E_top_1_lower[1:]
        E_bot_2_lower = E_bot_2_lower[1:]
        E_top_2_lower = E_top_2_lower[1:]
        E_bot_6_lower = E_bot_6_lower[1:]
        E_top_6_lower = E_top_6_lower[1:]

        #print(E_bot_1_upper)

        E_Matrix_bot_1 = np.append(E_bot_1_lower[::-1], E_bot_1_upper)
        E_Matrix_top_1 = np.append(E_top_1_lower[::-1], E_top_1_upper)
        E_Matrix_bot_2 = np.append(E_bot_2_lower[::-1], E_bot_2_upper)
        E_Matrix_top_2 = np.append(E_top_2_lower[::-1], E_bot_2_upper)
        E_Matrix_bot_6 = np.append(E_bot_6_lower[::-1], E_bot_6_upper)
        E_Matrix_top_6 = np.append(E_top_6_lower[::-1], E_top_6_upper)

        #print(E_Matrix_bot_6)
        # Use Table 3.5 to calculate strain transformation from off-axis strain and stress
        Epsilon_1_bot = Eps_1 + E_Matrix_bot_1
        Epsilon_1_top = Eps_1 + E_Matrix_top_1
        Epsilon_2_bot = Eps_2 + E_Matrix_bot_2
        Epsilon_2_top = Eps_2 + E_Matrix_top_2
        Epsilon_6_bot = Eps_6 + E_Matrix_bot_6
        Epsilon_6_top = Eps_6 + E_Matrix_top_6

        p_bot = 0.5 * (Epsilon_1_bot + Epsilon_2_bot)
        p_top = 0.5 * (Epsilon_1_top + Epsilon_2_top)
        q_bot = 0.5 * (Epsilon_1_bot - Epsilon_2_bot)
        q_top = 0.5 * (Epsilon_1_top - Epsilon_2_top)
        r_bot = 0.5 * Epsilon_6_bot
        r_top = 0.5 * Epsilon_6_top

        Qxx = m * E_x  # Equ(2.12)
        Qyy = m * E_y
        Qyx = m * nu_over_x * E_y
        Qxy = m * E_x * nu_over_y
        Qss = m * Es
        On_Axis_Stress_bot_Matrix = np.zeros([3, 1])
        On_Axis_Stress_top_Matrix = np.zeros([3, 1])
        for x in range(Ply_Num):
            On_Axis_Strain_bot = np.array([p_bot[x] + q_bot[x] * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) + r_bot[
                x] * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])),
                                           p_bot[x] - q_bot[x] * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) - r_bot[
                                               x] * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])),
                                           -2 * q_bot[x] * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])) + 2 * r_bot[
                                               x] * ma.cos(2 * ma.radians(Geo_Sym_Arr[x]))])
            On_Axis_Strain_top = np.array([p_top[x] + q_top[x] * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) + r_top[
                x] * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])),
                                           p_top[x] - q_top[x] * ma.cos(2 * ma.radians(Geo_Sym_Arr[x])) - r_top[
                                               x] * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])),
                                           -2 * q_top[x] * ma.sin(2 * ma.radians(Geo_Sym_Arr[x])) + 2 * r_top[
                                               x] * ma.cos(2 * ma.radians(Geo_Sym_Arr[x]))])
            On_Axis_Stress_bot = np.array([Q_xx * On_Axis_Strain_bot[0] + Q_xy * On_Axis_Strain_bot[1],
                                           Q_yx * On_Axis_Strain_bot[0] + Q_yy * On_Axis_Strain_bot[1],
                                           Q_ss * On_Axis_Strain_bot[2]])
            print("HElloooooo", On_Axis_Stress_bot)
            On_Axis_Stress_top = np.array([Q_xx * On_Axis_Strain_top[0] + Q_xy * On_Axis_Strain_top[1],
                                           Q_yx * On_Axis_Strain_top[0] + Q_yy * On_Axis_Strain_top[1],
                                           Q_ss * On_Axis_Strain_top[2]])
            On_Axis_Stress_bot_Matrix = np.append(On_Axis_Stress_bot_Matrix, On_Axis_Stress_bot)
            On_Axis_Stress_top_Matrix = np.append(On_Axis_Stress_top_Matrix, On_Axis_Stress_top)

            # print("On Axis Strain Bot:", On_Axis_Strain_bot)
            # print("On Axis Strain top:", On_Axis_Strain_top)
            # print("On Axis Stress Bot:", On_Axis_Stress_bot)
            # print("On Axis Stress top:", On_Axis_Stress_top)

        On_Axis_Stress_bot_Matrix = On_Axis_Stress_bot_Matrix[3:]
        On_Axis_Stress_bot_Matrix = np.reshape(On_Axis_Stress_bot_Matrix, (Ply_Num, 3))
        On_Axis_Stress_top_Matrix = On_Axis_Stress_top_Matrix[3:]
        On_Axis_Stress_top_Matrix = np.reshape(On_Axis_Stress_top_Matrix, (Ply_Num, 3))
        #print(" On_Axis_Stress_bot_Matrix:", On_Axis_Stress_bot_Matrix)
        #print(" On_Axis_Stress_top_Matrix:", On_Axis_Stress_top_Matrix)
        Table_1,max_list= max_failure(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,Stress_Res_Arr,Mom_Res_Arr)
        Table_2,quad_list = quad_failure(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,
                                Stress_Res_Arr, Mom_Res_Arr)
        Table_3,hashin_list = hashin_failure(X_t, X_c, Y_t, Y_c, S_c, Ply_Num, On_Axis_Stress_bot_Matrix, On_Axis_Stress_top_Matrix,
                                Stress_Res_Arr, Mom_Res_Arr)
        #return render_template('app.html', **locals())

        return render_template('app.html', Geo_Sym_Arr=Geo_Sym_Arr, K_Arr = K_Arr,S_on_axis=S_on_axis,Q_on_axis=Q_on_axis,A_Arr = A_Arr,a_Arr=a_Arr,D_Arr=D_Arr,d_Arr=d_Arr,
                               headings = headings, data_1 = data_1,Table_1 = [Table_1.to_html(classes='data',header="true")],Table_2 = [Table_2.to_html(classes='data',header="true")],
                               Table_3 = [Table_3.to_html(classes='data',header="true")],max_list=max_list,quad_list=quad_list,hashin_list=hashin_list)

        '''
        #return render_template('simple.html', tables=[.to_html(classes='data')], titles=df.columns.values)

        # print('Curvature k1, k2, k6, [1/m]:', K_Arr)
        # type(Geo_Str_Arr)
        '''




