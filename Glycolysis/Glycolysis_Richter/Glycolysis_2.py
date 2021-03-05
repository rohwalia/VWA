#Richter glycolytic model

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import fftpack
import pandas as pd

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

t_max = 10
t= np.linspace(0,t_max,300000)
dt = t_max/300000
up = 0.1
freq_main = []
amp_main = []
initial = [0 , 6 , 0.03 , 0.3 , 2 , 2 , 0 , 0 , 0 , 0.19 , 8.31 , 5 ,1.6,0.1]
#Working initials: [1.09,5.1,5.1,0.7,0.55,0.1,0.1,0.1,0.1,0,8.31,5,0.2,0.1],
# [0,0,5.1,0,0,0,0,0,0,0,8.31,4,2,0.1],
# [0.0789 , 0.2444 , 0.0481 , 0.4980 , 0.3842 , 0.0173 , 0.0001 , 0.2151 , 0.0303 , 0.02 , 1.2760 , 0.1392 , 1.1529 , 0.0759]
flux = [7] #np.linspace(12, 25, 50)
for i in flux:
    r= odeint(trajectory, initial, t, args=(i,))
    atp_series = np.array(r[:,12])
    limit_min = 0.25
    limit_max = 2
    atp_series = atp_series[int(limit_min/dt):int(limit_max/dt)]
    plt.plot(t[int(limit_min/dt):int(limit_max/dt)], atp_series)
    plt.show()
    fft = fftpack.fft(atp_series)
    freqs = fftpack.fftfreq(atp_series.size, d=dt)
    power = np.abs(fft)[np.where(freqs > up)]
    freqs = freqs[np.where(freqs > up)]
    freq_main.append(freqs[power.argmax()])
    amp_main.append(power[power.argmax()])
print(np.array(freq_main)/60)
print(amp_main)
output = pd.DataFrame(list(zip(flux, freq_main, amp_main)), columns=["Flow rate", "Frequency", "Amplitude"])
with pd.ExcelWriter('FFT.xlsx') as writer:
    sheetname = "Sheet 1"
    output.to_excel(writer, sheet_name=sheetname, index=False)
