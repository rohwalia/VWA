import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import median
from sklearn.linear_model import LinearRegression


N_0 = 2.3
N_1 = 2
c = 0.5
def trajectory(var, t, v_0):
    gl, g6, f6, fd, ga, da, pg13, pg3, pg2, pep, pyr, ac, atp, nadh = var
    v_hk = 14* ((atp*gl)/(0.2*0.1+0.2*gl+0.1*atp+atp*gl))
    v_pgk = 60* (((N_0-atp)*pg13)/(0.2*1.8e-3+0.2*pg13+1.8e-3*(N_0-atp)+(N_0-atp)*pg13))
    v_adh = 30* ((nadh*ac)/(0.01*0.78+0.01*ac+0.78*nadh+nadh*ac))

    v_pgi = 200*((g6-(f6/0.298))/(0.03+g6+(0.03/0.01)*f6))
    v_tim = 500*((da-(ga/22))/(0.4+da+(0.4/0.4)*ga))
    v_pgm = 100*((pg3-(pg2/5))/(0.1+pg3+(0.1/0.1)*pg2))
    v_en = 100*((pg2-(pep/6.3))/(0.1+pg2+(0.1/0.1)*pep))
    v_pdc = 100*(pyr/(30+pyr))

    v_ald = (36*(fd-(ga*da/8.1e-2)))/(0.3+fd+((2*36)/(8.1e-2*5*36))*ga+((2*36)/(8.1e-2*5*36))*da+(fd*ga)/10
                                      - ((36*da*ga)/(5*36*8.1e-2)))
    v_pfk = 30*(((f6/0.03)*((f6/0.03)+1)**3)/(250*(((atp/0.05)+1)/(((N_0-atp)*c/0.005)+1))**4+(1+(f6/0.03))**4))
    a_gapdh = ((N_1-nadh)/0.018)
    b_gapdh = nadh/0.24
    v_gapdh = 170*((a_gapdh*(1+a_gapdh+b_gapdh)**3+10*a_gapdh*(1+a_gapdh+b_gapdh)**3)/
                   ((1+a_gapdh+b_gapdh)**4+10*(1+a_gapdh+b_gapdh)**4))*(ga/(ga+0.01))
    v_pk = 60*(((pep/0.19)*((pep/0.19)+1)**3)/(250*(((atp/9.3)+1)/((fd/0.2)+1))**4+(1+(pep/0.19))**4))*((N_0-atp)/((N_0-atp)+0.3))
    v_atp = 24.2*(atp/(atp+0.05))

    dGldt = v_0-v_hk
    dG6dt = v_hk-v_pgi
    dF6dt = v_pgi-v_pfk
    dFDdt = v_pfk-v_ald
    dGAdt = v_ald-v_gapdh+v_tim
    dDAdt = v_ald-v_tim
    d13PGdt = v_gapdh-v_pgk
    d3PGdt = v_pgk-v_pgm
    d2PGdt = v_pgm-v_en
    dPEPdt = v_en-v_pk
    dPYRdt = v_pk-v_pdc
    dACdt = v_pdc-v_adh
    dATPdt = v_pk+v_pgk-v_hk-v_pfk-v_atp
    dNADHdt = v_gapdh-v_adh

    return [dGldt, dG6dt, dF6dt, dFDdt, dGAdt, dDAdt, d13PGdt, d3PGdt, d2PGdt, dPEPdt, dPYRdt, dACdt, dATPdt, dNADHdt]

factor = np.random.rand(10)*2*np.pi
t_max = 25
dt = 0.01
t = np.linspace(0, t_max, int(t_max/dt))
initial = [0 , 6 , 0.03 , 0.3 , 2 , 2 , 0 , 0 , 0 , 0.19 , 8.31 , 5 ,1.6,0.1] #[1.09,5.1,5.1,0.7,0.55,0.1,0.1,0.1,0.1,0,8.31,5,0.2,0.1]
chaos = []

flux = np.linspace(5, 50, 100)
for j in flux:
    c_val = []
    for c in factor:
        r = odeint(trajectory, initial, t, args=(j,))
        p_c=[]
        q_c=[]
        p_now = 0
        q_now = 0
        for i in range(len(r)):
            amp = r[:,12][i]
            p_now = p_now + amp * np.cos((i+1)*c)
            q_now = q_now + amp * np.sin((i+1)*c)
            p_c.append(p_now)
            q_c.append(q_now)
        #plt.plot(p_c, q_c)
        #plt.show()
        M = []
        for i in range(round(len(p_c)/10)):
            M_value = 0
            for el in range(len(p_c)-round(len(p_c)/10)):
                M_value = M_value + ((p_c[i+el]-p_c[el])**2 + (q_c[i+el]-q_c[el])**2)*dt**2
            M.append(M_value/(len(p_c)-round(len(p_c)/10)))
        M = np.array(M)
        model = LinearRegression().fit(np.log(np.array(range(len(M)))[1:]).reshape(-1, 1), np.log(M[1:]))
        c_val.append(model.coef_[0])
    chaos.append([j, median(c_val)])
print(chaos)
chaos = np.array(chaos)
plt.plot(chaos[:,0], chaos[:,1])
plt.xlabel("Glucose flow rate")
plt.ylabel("Growth rate K")
plt.show()
