### IMPORT LIBRARIES #################################################
import xarray as xr
import numpy as np

### MAKE ISCCP HISTOGRAM #############################################
def isccp_hist(tau,ctp,time,savename):
    """
        Make ISCCP histogram from cloud top pressure and brightness temperature
        INPUT:
            tau - cloud optical depth
            ctp - cloud top pressure, hPa
            time - time, days
            savename - output data filename
        OUTPUT:
            saves data to file
        AUTHOR: Catherine L. Stauffer
    """
    (NT,NY,NX) = np.shape(tau)
    tbins = [1.175494e-38,0.3,1.3,3.6,9.4,23,60,100000]
    pbins = [50,180,310,440,560,680,800,100000]
    isccp_hist = np.zeros((NT,7,7))
    for t in range(NT):
        for pidx in range(7):
            for tidx in range(7):
                isccp_hist[t,pidx,tidx] = np.sum(((tau[t,:,:]>=tbins[tidx])&(tau[t,:,:]<tbins[tidx+1])&(ctp[t,:,:]/100>=pbins[pidx])&(ctp[t,:,:]/100<pbins[pidx+1])).astype(int))
    tbins_mp = [115., 245., 375., 500., 620., 740., 900.]
    pbins_mp = [0.15, 0.950, 2.60, 6.50, 16.2, 41.5,220.]
    ds = xr.Dataset(
        data_vars = {
            'time':(['time'],time,{'units':'day','long_name':'time'}),
            'ctp_mdpts':(['ctp_mdpts'],pbins_mp,{'units':'hPa','long_name':'cloud top pressure'}),
            'tau_mdpts':(['tau_mdpts'],tbins_mp,{'units':'','long_name':'cloud optical depth'}),
            'isccp':(['time','ctp_mdpts','tau_mdpts'],isccp_hist,{'units':'','long_name':'ISCCP histogram'}),
            'dhoriz':(['dhoriz'],[NY*NX],{'units':'','long_name':'number of horizontal grids'})
        }
    )
    ds.to_netcdf(savename)
    ds.close()

### END ##############################################################