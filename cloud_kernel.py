"""
    calculate kernel by calling the cloud_kernel function
    requires rrtmg_lw_rcemip.nc
    INPUT:
        savename - file name for kernel output
        sst - surface temperature, K
        tlev - level temperature, K
        plev - level pressure, hPa
        rho_lay - layer density, g/m3
        nlay - number of model layers
        nlev - number of model levels
        tlay - layer temperature, K
        play - layer pressure, hPa
        qvlay - water vapor, g/g
        dz - layer thickness, m
    OUTPUT:
        saves data to netCDF
    AUTHOR: Catherine L. Stauffer
"""

### import libraries #############################################
import numpy as np
import xarray as xr
from scipy import interpolate

### CAM RADII FUNCTION ###############################################
def calc_radii_cam(nz_lay,tlay_z):
    """
    calculates ice and liquid radii profiles
    INPUT:
        nz_lay - number of model layers
        tlay_z - layer temperature profile (K)
    OUTPUT:
        radice - ice effective radii (um)
        radliq - liquid effective radii (um)
    """
    rei_table = [5.92779,6.26422,6.61973,6.99539,7.39234,7.81177,8.25496,
                    8.72323,9.21800,9.74075,10.2930,10.8765,11.4929,12.1440,
                    12.8317,13.5581,14.2319,15.0351,15.8799,16.7674,17.6986,
                    18.6744,19.6955,20.7623,21.8757,23.0364,24.2452,25.5034,
                    26.8125,27.7895,28.6450,29.4167,30.1088,30.7306,31.2943,
                    31.8151,32.3077,32.7870,33.2657,33.7540,34.2601,34.7892,
                    35.3442,35.9255,36.5316,37.1602,37.8078,38.4720,39.1508,
                    39.8442,40.5552,41.2912,42.0635,42.8876,43.7863,44.7853,
                    45.9170,47.2165,48.7221,50.4710,52.4980,54.8315,57.4898,
                    60.4785,63.7898,65.5604,71.2885,75.4113,79.7368,84.2351,
                    88.8833,93.6658,98.5739,103.603,108.752,114.025,119.424,
                    124.954,130.630,136.457,142.446,148.608,154.956,161.503,
                    168.262,175.248,182.473,189.952,197.699,205.728,214.055,
                    222.694,231.661,240.971,250.639]
    Trei = np.arange(180,275,1)
    re_ice_T = np.ones(nz_lay)*5.0
    re_ice_T[(tlay_z>=180) & (tlay_z<=274)] = interpolate.interp1d(Trei, rei_table)(tlay_z[(tlay_z>=180) & (tlay_z<=274)])
    radice = np.maximum((np.ones(nz_lay)*5.0), np.minimum((np.ones(nz_lay)*140.0), re_ice_T))
    radliq = np.array([14]).repeat(nz_lay)
    return(radice,radliq)

### ICE AND LIQUID WATER CONTENT #####################################
def iwc_fu96(tau,dz,rho,Dge): # Fu 1996, Eqn. 3.12
    a0=-.291721e-4 # Fu 1996, Table 3a
    a1=2.51925 # Fu 1996, Table 3a
    iwc = tau/(dz*(a0 + (a1/Dge)))/rho # Fu 1996, Eqn. 3.9a
    return(iwc) # g/g

def lwc_slingo89(tau,dz,rho,Reff):
    ai=2.817e-2 # Slingo 1989, Table 1
    bi=1.305 # Slingo 1989, Table 1
    lwc = tau/(dz*(ai+(bi/Reff)))/rho # Slingo 1989, Eqn 1
    return(lwc) # g/g

def run_rrtmg(p_lay,p_lev,ta_lay,ta_lev,qv_lay,lwc_lay,iwc_lay,dz,rho,rice,rliq,tsfc):
    """
    runs rrtmg as interfaced by climlab (Rose, 2018)
    INPUT:
        p_lay: layer pressure, hPa
        p_lev: level pressure, hPa
        ta_lay: layer temperature, K
        ta_lev: level temperature, K
        qv_lay: watervapor, g/g
        lwc: liquid water content, g/g
        iwc: ice water content, g/g
        dz: layer thickness, m
        rho: density, g/m3
        rice: ice effective radii, um
        rliq: liquid effective radii, um
        tsfc: surface temperature, K
    OUTPUT:
        lwuflx: Wm-2
        lwdflx: Wm-2
        lwuflxc: Wm-2
        lwdflxc: Wm-2
        swuflx: Wm-2
        swdflx: Wm-2
        swuflxc: Wm-2
        swdflxc: Wm-2
        lwhr: K/s
        swhr: K/s
        lwhrc: K/s
        swhrc: K/s
    *** REQUIRES CLIMLAB AS DOWNLOADED FOLLOWING THESE INSTRUCTIONS
        https://climlab.readthedocs.io/en/latest/intro.html#installation
    """

    ### constants ####################################################
    ncol,icld,idrv = 1,3,0
    inflg,iceflg,liqflg = 2,3,1
    nbndlw,ngptlw,ngptsw = 16,140,112
    iaer,isolvar,solcycfrac = 0,0,0
    adjes = 1 # 551.85/1365.2#1
    dyofyr = 0
    scon = 551.58 # 1367
    coszen = np.cos(np.pi*42.05/180)
    adir = np.array(( 0.026/( coszen**1.7 + 0.065 ) ) + ( 0.15 * (coszen-0.10) * (coszen-0.50) * (coszen-1.00) ))
    adif = np.array(0.07)
    coszen = np.array(coszen)
    emis = .95 * np.ones((ncol,nbndlw))
    asdir,aldir = adir,adir
    asdif,aldif = adif,adif
    nlay = np.shape(ta_lay)[0]
    ### chemical profile #############################################
    h2ovmr = qv_lay * 28.966 / 18.01528
    h2ovmr = h2ovmr[np.newaxis, ...]
    def interpchem(mls):
        import xarray as xr
        ds = xr.open_dataset('rrtmg_lw_rcemip.nc')
        chm = ds['AbsorberAmountMLS'][:,mls].values
        pls = ds['Pressure'].values
        f = interpolate.interp1d(pls, chm)
        chemint = f(p_lay.squeeze())
        return(chemint[np.newaxis,...])
    co2vmr = interpchem(6)
    o3vmr = interpchem(7)
    n2ovmr = interpchem(8)
    ch4vmr = interpchem(10)
    o2vmr = interpchem(11)
    cfc11vmr = 0. * np.ones_like(o2vmr)
    cfc12vmr = 0. * np.ones_like(o2vmr)
    cfc22vmr = 0. * np.ones_like(o2vmr)
    ccl4vmr = 0. * np.ones_like(o2vmr)
    ### water paths ##################################################
    lwp = lwc_lay*rho*dz
    iwp = iwc_lay*rho*dz
    cldfrac = np.zeros(np.shape(lwp))
    cldfrac[np.where((lwp+iwp)>0)] = 1
    ### set profiles for rrtmg #######################################
    p_lev = p_lev[np.newaxis, ...]
    p_lay = p_lay[np.newaxis, ...]
    ta_lay = ta_lay[np.newaxis, ...]
    ta_lev = ta_lev[np.newaxis, ...]
    reicmcl = rice[np.newaxis,...]
    relqmcl = rliq[np.newaxis,...]
    cldfmcllw = np.repeat((cldfrac.reshape(1,cldfrac.shape[0]))[np.newaxis,:,:],ngptlw,axis=0)
    cldfmclsw = np.repeat((cldfrac.reshape(1,cldfrac.shape[0]))[np.newaxis,:,:],ngptsw,axis=0)
    ciwpmcllw = np.repeat((iwp.reshape(1,iwp.shape[0]))[np.newaxis,:,:],ngptlw,axis=0)
    ciwpmclsw = np.repeat((iwp.reshape(1,iwp.shape[0]))[np.newaxis,:,:],ngptsw,axis=0)
    clwpmcllw = np.repeat((lwp.reshape(1,lwp.shape[0]))[np.newaxis,:,:],ngptlw,axis=0)
    clwpmclsw = np.repeat((lwp.reshape(1,lwp.shape[0]))[np.newaxis,:,:],ngptsw,axis=0)
    taucmclsw = np.zeros((ngptsw,1,nlay))
    taucmcllw = np.zeros((ngptlw,1,nlay))
    tauaersw = np.zeros((1,nlay,14))
    tauaerlw = np.zeros((1,nlay,16))
    ssacmcl = np.zeros([112,ncol,nlay])
    asmcmcl = np.zeros([112,ncol,nlay])
    fsfcmcl = np.zeros([112,ncol,nlay])
    ssaaer = np.zeros((1,nlay,14))
    asmaer = np.zeros((1,nlay,14))
    ecaer = np.zeros((1,nlay,6))
    bndsolvar = np.zeros((14))
    indsolvar = np.zeros((2))
    ### run rrtmg ####################################################
    from climlab.radiation.rrtm._rrtmg_lw import _rrtmg_lw
    (lwuflx,lwdflx,lwhr,lwuflxc,lwdflxc,lwhrc,__,__) = \
        _rrtmg_lw.climlab_rrtmg_lw(ncol,nlay,icld,idrv,p_lay,p_lev,
                                ta_lay,ta_lev,tsfc,h2ovmr,o3vmr,
                                co2vmr,ch4vmr,n2ovmr,o2vmr,
                                cfc11vmr,cfc12vmr,cfc22vmr,
                                ccl4vmr,emis,inflg,iceflg,
                                liqflg,cldfmcllw,taucmcllw,ciwpmcllw,
                                clwpmcllw,reicmcl,relqmcl,tauaerlw)
    from climlab.radiation.rrtm._rrtmg_sw import _rrtmg_sw
    (swuflx,swdflx,swhr,swuflxc,swdflxc,swhrc) = \
        _rrtmg_sw.climlab_rrtmg_sw(ncol,nlay,icld,iaer,p_lay,p_lev,
                                ta_lay,ta_lev,tsfc,h2ovmr,o3vmr,
                                co2vmr,ch4vmr,n2ovmr,o2vmr,
                                asdir,asdif,aldir,aldif,coszen,
                                adjes,dyofyr,scon,isolvar,inflg,
                                iceflg,liqflg,cldfmclsw,taucmclsw,
                                ssacmcl,asmcmcl,fsfcmcl,ciwpmclsw,
                                clwpmclsw,reicmcl,relqmcl,tauaersw,
                                ssaaer,asmaer,ecaer,bndsolvar,
                                indsolvar,solcycfrac)

    return(lwuflx.squeeze(),lwdflx.squeeze(),lwuflxc.squeeze(),lwdflxc.squeeze(),\
           swuflx.squeeze(),swdflx.squeeze(),swuflxc.squeeze(),swdflxc.squeeze(),\
           lwhr.squeeze(),swhr.squeeze(),lwhrc.squeeze(),swhrc.squeeze())

def cloud_kernel(savename,sst,tlev,plev,rho_lay,nlay,nlev,tlay,play,qvlay,dz):
    """
        requires rrtmg_lw_rcemip.nc
        INPUT:
            savename - file name for kernel output
            sst - surface temperature, K
            tlev - level temperature, K
            plev - level pressure, hPa
            rho_lay - layer density, g/m3
            nlay - number of model layers
            nlev - number of model levels
            tlay - layer temperature, K
            play - layer pressure, hPa
            qvlay - water vapor, g/g
            dz - layer thickness, m
        OUTPUT:
            saves data to netCDF
        Author: Catherine L. Stauffer
    """
    ### ISCCP bins ###############################################
    tbins = [0.0,0.3,1.3,3.6,9.4,23,60,380]   # optical thickness bins
    pbins = [50,180,310,440,560,680,800,1000] # cloud top pressure bins

    ### calculate effective radii#### ############################
    reic,relq = calc_radii_cam(nlay,tlay) # effective radii

    ### calculate cloud water content using a given tau/ctp bin ##
    cldfrac = np.zeros((nlay,8,8)) # (-) . . . . cloud fraction
    clwc = np.zeros((nlay,8,8)) # (g/g) . . . cloud water
    ciwc = np.zeros((nlay,8,8)) # (g/g) . . . cloud ice
    for tidx,tau in enumerate(tbins): ## loop through tau bins
        for pidx,ctp in enumerate(pbins): ## loop through ctp bins
            cidx = np.argmin(np.abs(play.squeeze()[:-1] - ctp))
            cldfrac[cidx,pidx,tidx] = 1
            if (tlay[cidx] < 260): ciwc[cidx,pidx,tidx] = iwc_fu96(tau,dz,rho_lay[cidx],reic[cidx])
            if (tlay[cidx] >= 260): clwc[cidx,pidx,tidx] = lwc_slingo89(tau,dz,rho_lay[cidx],relq[cidx])

    ### run rrtmg ################################################
    lwu,lwd,lwuc,lwdc = np.zeros((nlev,8,8)),np.zeros((nlev,8,8)),np.zeros((nlev,8,8)),np.zeros((nlev,8,8))
    swu,swd,swuc,swdc = np.zeros((nlev,8,8)),np.zeros((nlev,8,8)),np.zeros((nlev,8,8)),np.zeros((nlev,8,8))
    qlw,qsw,qlwc,qswc = np.zeros((nlay,8,8)),np.zeros((nlay,8,8)),np.zeros((nlay,8,8)),np.zeros((nlay,8,8))
    for tidx,tau in enumerate(tbins): ## loop through tau bins
        for pidx,ctp in enumerate(pbins): ## loop through ctp bins
            lwu[:,pidx,tidx],lwd[:,pidx,tidx],lwuc[:,pidx,tidx],lwdc[:,pidx,tidx],\
                swu[:,pidx,tidx],swd[:,pidx,tidx],swuc[:,pidx,tidx],swdc[:,pidx,tidx],\
                    qlw[:,pidx,tidx],qsw[:,pidx,tidx],qlwc[:,pidx,tidx],qswc[:,pidx,tidx] = \
                        run_rrtmg(play,plev,tlay,tlev,qvlay,clwc[:,pidx,tidx],ciwc[:,pidx,tidx],reic,relq,sst)

    ### average kernel isccp bins ################################
    histlwoutas,histlwincas = np.zeros((7,7)),np.zeros((7,7))
    histlwoutcs,histlwinccs = np.zeros((7,7)),np.zeros((7,7))
    histswoutas,histswincas = np.zeros((7,7)),np.zeros((7,7))
    histswoutcs,histswinccs = np.zeros((7,7)),np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            histlwoutas[i][j] = np.sum([lwu[-1][i][j],lwu[-1][i][j+1],lwu[-1][i+1][j],lwu[-1][i+1][j+1]])/4/100
            histlwincas[i][j] = np.sum([lwd[-1][i][j],lwd[-1][i][j+1],lwd[-1][i+1][j],lwd[-1][i+1][j+1]])/4/100
            histlwoutcs[i][j] = np.sum([lwuc[-1][i][j],lwuc[-1][i][j+1],lwuc[-1][i+1][j],lwuc[-1][i+1][j+1]])/4/100
            histlwinccs[i][j] = np.sum([lwdc[-1][i][j],lwdc[-1][i][j+1],lwdc[-1][i+1][j],lwdc[-1][i+1][j+1]])/4/100
            histswoutas[i][j] = np.sum([swu[-1][i][j],swu[-1][i][j+1],swu[-1][i+1][j],swu[-1][i+1][j+1]])/4/100
            histswincas[i][j] = np.sum([swd[-1][i][j],swd[-1][i][j+1],swd[-1][i+1][j],swd[-1][i+1][j+1]])/4/100
            histswoutcs[i][j] = np.sum([swuc[-1][i][j],swuc[-1][i][j+1],swuc[-1][i+1][j],swuc[-1][i+1][j+1]])/4/100
            histswinccs[i][j] = np.sum([swdc[-1][i][j],swdc[-1][i][j+1],swdc[-1][i+1][j],swdc[-1][i+1][j+1]])/4/100
    lwfkernel = (histlwoutcs-histlwinccs-histlwoutas+histlwincas)
    swfkernel = (histswoutcs-histswinccs-histswoutas+histswincas)

    ### save data ################################################
    tbin_midpoints = (np.array(tbins)[:-1]+np.array(tbins)[1:])/2
    pbin_midpoints = (np.array(pbins)[:-1]+np.array(pbins)[1:])/2
    ds = xr.Dataset(
        data_vars = {
                    'ctp':( ['ctp'], pbin_midpoints, {'units':'hPa','long_name':'cloud top pressure midpoints'}),
                    'tau':( ['tau'], tbin_midpoints, {'units':'','long_name':'cloud optical depth midpoints'}),
                    'ctp_bounds':( ['ctp_bounds'], pbins, {'units':'hPa','long_name':'cloud top pressure bounds'}),
                    'tau_bounds':( ['tau_bounds'], tbins, {'units':'','long_name':'cloud optical depth bounds'}),
                    'rlut_as':( ['ctp','tau'], histlwoutas, {'units':'Wm-2%-1','long_name':'upwelling longwave all sky top of atmosphere flux'}),
                    'rldt_as':( ['ctp','tau'], histlwincas, {'units':'Wm-2%-1','long_name':'downwelling longwave all sky top of atmosphere flux'}),
                    'rlut_cs':( ['ctp','tau'], histlwoutcs, {'units':'Wm-2%-1','long_name':'upwelling longwave clear sky top of atmosphere flux'}),
                    'rldt_cs':( ['ctp','tau'], histlwinccs, {'units':'Wm-2%-1','long_name':'downwelling longwave clear sky top of atmosphere flux'}),
                    'rsut_as':( ['ctp','tau'], histswoutas, {'units':'Wm-2%-1','long_name':'upwelling shortwave all sky top of atmosphere flux'}),
                    'rsdt_as':( ['ctp','tau'], histswincas, {'units':'Wm-2%-1','long_name':'downwelling shortwave all sky top of atmosphere flux'}),
                    'rsut_cs':( ['ctp','tau'], histswoutcs, {'units':'Wm-2%-1','long_name':'upwelling shortwave clear sky top of atmosphere flux'}),
                    'rsdt_cs':( ['ctp','tau'], histswinccs, {'units':'Wm-2%-1','long_name':'downwelling shortwave clear sky top of atmosphere flux'}),
                    'LWKernel':( ['ctp','tau'], lwfkernel, {'units':'Wm-2%-1','long_name':'lwf cloud radiative kernel'}),
                    'SWKernel':( ['ctp','tau'], swfkernel, {'units':'Wm-2%-1','long_name':'swf cloud radiative kernel'}),
        }
    )
    ds.to_netcdf(savename)
    ds.close()

### end ##########################################################