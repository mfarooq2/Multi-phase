import numpy as np
def upwind_ENO3(data, F, dx):
    F = F*np.ones(data.shape)
    data_x = np.zeros(data.shape)

    # extrapolate the beginning and end points of data
    data[2] = 2*data[3]-data[4]
    data[1] = 2*data[2]-data[3]
    data[0] = 2*data[1]-data[2]
    data[-3] = 2*data[-4]-data[-5]
    data[-2] = 2*data[-3]-data[-4]
    data[-1] = 2*data[-2]-data[-3]

    #Generate the divided difference tables
    #ignoring division by dx for efficiency
    D1 = (data[1:]-data[:-1]); D2 = (D1[1:]-D1[:-1])/2
    absD2 = np.abs(D2); D3 = (D2[1:]-D2[:-1])/3
    absD3 = np.abs(D3)
    for i in range(len(data)-6):
        if F[i+3] > 0:  # use D-
            k = i-1
        elif F[i+3] < 0:  # use D+
            k = i
        else:
            continue
        Q1p = D1[k+3] #D1k_half
        if absD2[k+2] <= absD2[k+3]: #|D2k| <= |D2kp1|
            kstar = k-1
            c = D2[k+2]
        else:
            kstar = k
            c = D2[k+3]
        Q2p = c*(2*(i-k)-1)

        if absD3[kstar+2] <= absD3[kstar+3]: #|D3kstar_half| <= |D3kstar_1_half|
            cstar = D3[kstar+2] #D3kstar_half
        else:
            cstar = D3[kstar+3] #D3kstar_1_half
        Q3p = cstar*( 3*(i-kstar)*(i-kstar) - 6*(i-kstar) + 2 )
        data_x[i+3] = Q1p+Q2p+Q3p
        data_x[i+3] = F[i+3]*data_x[i+3]/dx
            
    return data_x