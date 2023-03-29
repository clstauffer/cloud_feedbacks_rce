"""
    run the isccp simulator by calling the run_icarus function
    INPUT:
        sst - surface temperature, K
        time - time, day
        z - height, m
        ta - temperature, K
        qv - water vapor, g/g
        pa - pressure, hPa
        clw - cloud water, g/g
        cli - cloud ice, g/g
        icarustype - implement AIS or OIS (optional)
    OUTPUT:
        tau - clou doptical depth
        ctp - cloud top pressure, hPa
        ctb - cloud brightness temperature, K
    AUTHOR: Catherine L. Stauffer
"""

### IMPORT LIBARIES ##################################################
import numpy as np
from scipy import interpolate

### ICARUS PROCESSES #################################################
def ois_icarus(nlev,pfull,phalf,at,qv,dtau_in,dem_in,skt):
    """
        runs Icarus as implemented in COSP
        INPUT:
            nlev.......number of levels, unitless
            pfull......pressure of model layer, Pa
            at.........temperature of model layer, K
            dtau_in....input SW optical depth, unitless
            dem_in.....input LW emissivitiy, unitless
            skt........surface temperature, K
        OUTPUT:
            boxtau.....cloud optical depth, unitless
            boxctp.....cloud top pressure, Pa
            boxctb.....cloud brightness temperature, K
    """

    rec2p13,tauchk,output_missing_value,emsfc_lw = 1./2.13,-1.*np.log(0.9999999),-1.E+30,0.99

    ### TEMPERATURE AND TROPOPAUSE PROPERTIES ########################
    atmin,atmax,attrop    = 400,0,120
    itrop,ptrop,attropmin = 1,5000,400
    for ilev in range(nlev):
        if (pfull[ilev]<40000 and pfull[ilev]>5000 and at[ilev]<attropmin):
            ptrop     = pfull[ilev]
            attropmin = at[ilev]
            attrop    = attropmin
            itrop     = ilev
    for ilev in range(nlev):
        if ilev >= itrop:
            if (at[ilev] > atmax): atmax = at[ilev]
            if (at[ilev] < atmin): atmin = at[ilev]

    ### COMPUTE CLOUD OPTICAL DEPTH ##################################
    tau = np.sum(dtau_in)

    ### COMPUTE INFRARED BRIGHTNESS TEMPERUATRES #####################
    ###### AND CLOUD TOP TEMPERATURE SATELLITE SHOULD SEE ############
    dem_wv = np.zeros(nlev)
    fluxtop_clrsky = 0
    trans_layers_above_clrsky = 1
    ###### DO CLEAR SKY RADIANCE CALCULATION FIRST ###################
    wtmair = 28.9644 # dry air g/mol
    wtmh20 = 18.01534 # water g/mol
    Navo   = 6.023E+23 # 1/mol
    grav   = 9.806650E+02
    pstd   = 1.013250E+06
    t0     = 296
    for ilev in range(nlev):
        # press and dpress are dyne/cm2 = Pascals *10
        press  = pfull[ilev]*10
        dpress = (phalf[ilev+1]-phalf[ilev])*10
        atmden = dpress/grav # g/cm2 = kg/m2/10
        rvh20  = qv[ilev]*wtmair/wtmh20
        wk     = rvh20*Navo*atmden/wtmair
        rhoave = (press/pstd)*(t0/at[ilev])
        rh20s  = rvh20*rhoave
        rfrgn  = rhoave-rh20s
        tmpexp = np.exp(-0.02*(at[ilev]-t0))
        tauwv  = wk*1e-20*( (0.0224697*rh20s*tmpexp) + (3.41817e-7*rfrgn) )*0.98
        dem_wv[ilev] = 1 - np.exp( -1 * tauwv)
    for ilev in range(nlev):
        # Black body emission at temperature of the layer
        bb=1 / ( np.exp(1307.27/at[ilev]) - 1. )
        # increase TOA flux by flux emitted from layer
        # times total transmittance in layers above
        fluxtop_clrsky = fluxtop_clrsky + dem_wv[ilev]*bb*trans_layers_above_clrsky
        # update trans_layers_above with transmissivity
        # from this layer for next time around loop
        trans_layers_above_clrsky= trans_layers_above_clrsky*(1.-dem_wv[ilev])
    # add in surface emission
    bb=1/( np.exp(1307.27/skt) - 1. )
    fluxtop_clrsky = fluxtop_clrsky + emsfc_lw * bb * trans_layers_above_clrsky
    # clear sky brightness temperature
    meantbclr = 1307.27/(np.log(1.+(1./fluxtop_clrsky))) # mean clear-sky 10.5 micron brightness temperature
    ###### END OF CLEAR SKY CALCULATION ##############################

    ### COMPUTE INFRARED BRIGHTNESS TEMPERUATRES #####################
    ###### AND CLOUD TOP TEMPERATURE SATELLITE SHOULD SEE ############
    fluxtop = 0.
    trans_layers_above = 1.
    for ilev in range(nlev):
        bb = 1 / ( np.exp(1307.27/at[ilev]) - 1. )
        if dem_in[ilev] == 0: dem = dem_wv[ilev]
        else: dem= 1. - ( (1. - dem_wv[ilev]) * (1. -  dem_in[ilev]) )
        fluxtop = fluxtop + dem * bb * trans_layers_above
        trans_layers_above = trans_layers_above*(1.-dem)
    bb = 1/( np.exp(1307.27/skt) - 1. )
    fluxtop = fluxtop + (emsfc_lw * bb * trans_layers_above)
    meantb = 1307.27/(np.log(1. + (1./fluxtop))) # +meantb

    #################### ACCOUNT FOR ISCCP PROCEDURES ################
    btcmin = 1. /  ( np.exp(1307.27/(attrop-5.)) - 1. )
    transmax = (fluxtop-btcmin)/(fluxtop_clrsky-btcmin)
    tauir  = tau * rec2p13
    taumin = -1. * np.log(np.max([np.min([transmax,0.9999999]),0.001]))
    if (transmax>0.001 and transmax<=0.9999999):
        fluxtopinit = fluxtop
        tauir = tau *rec2p13
    for icycle in range(2):
        if (tau>(tauchk)):
            if (transmax>0.001 and transmax<=0.9999999):
                emcld   = 1. - np.exp(-1. * tauir  )
                fluxtop = fluxtopinit - ((1-emcld)*fluxtop_clrsky)
                fluxtop = np.max([1.E-06, (fluxtop/emcld)])
                tb= 1307.27/ (np.log(1. + (1./fluxtop)))
                if (tb>260.): tauir = tau / 2.56
    if (tau>(tauchk)): # cloudy box
        tb= 1307.27/ (np.log(1. + (1./fluxtop)))
        if tauir < taumin:
            if (tauir<taumin):
                tb = attrop - 5.
                tau = 2.13*taumin
    if (tau<=(tauchk)): # clearsky box
        tb = meantbclr

    #################### DETERMINE CLOUD TOP PRESSURE ################
    nmatch = 0
    match = np.zeros(nlev-1)
    for k1 in range(nlev-1): # (1,nlev)
        ilev = (nlev-2)-k1   # (nlev-1)-k1
        # if isccp_top_height_direction == 2: ilev = (nlev-2)-k1 # lowest pressure (heighest alitutde, default)
        # if isccp_top_height_direction != 2: ilev = k1 # heighest pressure (lowest altitdue)
        if (ilev>=itrop): # between tropopause and surface
            if ((at[ilev]>=tb and at[ilev+1]<=tb) or (at[ilev]<=tb and at[ilev+1]>=tb)):
                nmatch = nmatch+1
                match[nmatch] = ilev
    if (nmatch>=1):
        k1 = int(match[nmatch]) # match(j,nmatch(j))
        k2 = int(k1 + 1)
        logp1 = np.log(pfull[k1])
        logp2 = np.log(pfull[k2])
        atd = np.max([tauchk,np.abs(at[k2] - at[k1])])
        logp=logp1+(logp2-logp1)*np.abs(tb-at[k1])/atd
        ptop = np.exp(logp)
        if(np.abs(pfull[k1]-ptop) < np.abs(pfull[k2]-ptop)): levmatch=k1
        else: levmatch=k2
    else:
        if (tb<=attrop):
            ptop = ptrop
            levmatch = itrop
        if (tb>=atmax):
            ptop = pfull[nlev-1]
            levmatch = nlev-1
    if (tau<=(tauchk)):
        ptop=0.
        levmatch=0
    if tau > tauchk and ptop > 0:
        boxtau = tau
        boxctp = ptop
        boxtmp = tb
    else:
        boxtau = output_missing_value
        boxctp = output_missing_value
        boxtmp = output_missing_value
    return(boxtau,boxctp,boxtmp)
def ais_icarus(nlev,pfull,at,dtau_in,dem_in,skt):
    """
        Calculates cloud optical depth, brightness temperature, and 
            cloud optical depth as in COSP, excluding water vapor
            continuum; as used in Stauffer and Wing (submitted)
        INPUT:
            nlev.......number of levels, unitless
            pfull......pressure of model layer, Pa
            at.........temperature of model layer, K
            dtau_in....input SW optical depth, unitless
            dem_in.....input LW emissivitiy, unitless
            skt........surface temperature, K
        OUTPUT:
            boxtau.....cloud optical depth, unitless
            boxctp.....cloud top pressure, Pa
            boxctb.....cloud brightness temperature, K
    """

    rec2p13,tauchk,output_missing_value,emsfc_lw = 1./2.13,-1.*np.log(0.9999999),-1.E+30,0.99

    ### TEMPERATURE AND TROPOPAUSE PROPERTIES ########################
    atmin,atmax,attrop    = 400,0,120
    itrop,ptrop,attropmin = 1,5000,400
    for ilev in range(nlev):
        if (pfull[ilev]<40000 and pfull[ilev]>5000 and at[ilev]<attropmin):
            ptrop     = pfull[ilev]
            attropmin = at[ilev]
            attrop    = attropmin
            itrop     = ilev
    for ilev in range(nlev):
        if ilev >= itrop:
            if (at[ilev] > atmax): atmax = at[ilev]
            if (at[ilev] < atmin): atmin = at[ilev]

    ### COMPUTE CLOUD OPTICAL DEPTH ##################################
    tau = np.sum(dtau_in)

    ### COMPUTE INFRARED BRIGHTNESS TEMPERUATRES #####################
    ###### AND CLOUD TOP TEMPERATURE SATELLITE SHOULD SEE ############
    fluxtop = 0.
    trans_layers_above = 1.
    for ilev in range(nlev):
        bb = 1 / ( np.exp(1307.27/at[ilev]) - 1. )
        dem= 1. - (1. -  dem_in[ilev])
        fluxtop = fluxtop + (dem * bb * trans_layers_above)
        trans_layers_above = trans_layers_above*(1.-dem)
    bb = 1/( np.exp(1307.27/skt) - 1. )
    fluxtop = fluxtop + (emsfc_lw * bb * trans_layers_above)
    meantb = 1307.27/(np.log(1. + (1./fluxtop)))

    #################### ACCOUNT FOR ISCCP PROCEDURES ################
    btcmin = 1. /  ( np.exp(1307.27/(attrop-5.)) - 1. )
    tauir  = tau * rec2p13
    taumin = -1. * np.log(0.9999999)
    fluxtopinit = fluxtop
    if (tau>(tauchk)):
        emcld   = 1. - np.exp(-1. * tauir  )
        fluxtop = fluxtopinit - ((1-emcld)*(emsfc_lw * bb))
        fluxtop = np.max([1.E-06, (fluxtop/emcld)])
        tb= 1307.27/ (np.log(1. + (1./fluxtop)))
    if (tau>(tauchk)): # cloudy box
        tb= 1307.27/ (np.log(1. + (1./fluxtop)))
        if tauir < taumin: # and doisccpprocess == True:
            if (tauir<taumin):
                tb = attrop - 5.
                tau = 2.13*taumin
    if (tau<=(tauchk)): # clearsky box
        tb = meantb

    #################### DETERMINE CLOUD TOP PRESSURE ################
    nmatch = 0
    match = np.zeros(nlev-1)
    for k1 in range(nlev-1):
        ilev = (nlev-2)-k1 # lowest pressure (heighest alitutde, default)
        if (ilev>=itrop): # between tropopause and surface
            if ((at[ilev]>=tb and at[ilev+1]<=tb) or (at[ilev]<=tb and at[ilev+1]>=tb)):
                nmatch = nmatch+1
                match[nmatch] = ilev
    if (nmatch>=1):
        k1 = int(match[nmatch]) # match(j,nmatch(j))
        k2 = int(k1 + 1)
        logp1 = np.log(pfull[k1])
        logp2 = np.log(pfull[k2])
        atd = np.max([tauchk,np.abs(at[k2] - at[k1])])
        logp=logp1+(logp2-logp1)*np.abs(tb-at[k1])/atd
        ptop = np.exp(logp)
        if(np.abs(pfull[k1]-ptop) < np.abs(pfull[k2]-ptop)): levmatch=k1
        else: levmatch=k2
    else:
        if (tb<=attrop):
            ptop = ptrop
            levmatch = itrop
        if (tb>=atmax):
            ptop = pfull[-1]
            levmatch = nlev
    if (tau<=(tauchk)):
            ptop=0.
            levmatch=0
    if tau > tauchk and ptop > 0:
        boxtau = tau
        boxctp = ptop
        boxtmp = tb
    else:
        boxtau = output_missing_value
        boxctp = output_missing_value
        boxtmp = output_missing_value
    return(boxtau,boxctp,boxtmp)

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

### CLOUD OPTICAL DEPTH PARAMETERIZATIONS ############################
def fuICE(iwc,dz,rho,Dge): # Fu 1996, Eqn. 3.12
    # iwc gm-3, De um
    a0=-.291721e-4 # Fu 1996, Table 3a, m-1
    a1=2.51925 # Fu 1996, Table 3a, 
    tauice = iwc*rho*dz*(a0+(a1/Dge)) # Fu 1996, Eqn. 3.9a
    return(tauice)
def slingoLIQ(lwp,Reff):
    # g/m2
    ai=2.817e-2 # Slingo 1989, Table 1, m2/g
    bi=1.305 # Slingo 1989, Table 1, umm2/g
    tauliq = lwp*(ai+(bi/Reff)) # Slingo 1989, Eqn 1
    return(tauliq)
def calc_tau(lwp,iwc,dz,rho,relq,reic):
    tauliq = slingoLIQ(lwp,relq)
    tauice = fuICE(iwc,dz,rho,reic)
    tauALL = tauliq+tauice
    return(tauALL)

### RUN ICARUS #######################################################
def run_icarus(sst,nlay,ta_lay,qv_lay,pa_lay,cli_lay,ta_lev,pa_lev,rho_lay,lwp_lay,icarustype='AIS'):
    """
        INPUT:
            sst - surface temperature, K
            time - time, day
            z - height, m
            ta - temperature, K
            qv - water vapor, g/g
            pa - pressure, hPa
            clw - cloud water, g/g
            cli - cloud ice, g/g
            icarustype - implement AIS or OIS (optional)
        OUTPUT:
            tau - clou doptical depth
            ctp - cloud top pressure, hPa
            ctb - cloud brightness temperature, K
    """

    De,ae = calc_radii_cam(nlay,ta_lay) # units of um

    ### FLIP PROFILES: TOA -> SFC ####################################
    ta_layer = np.flip(ta_lay)
    qv_layer = np.flip(qv_lay)
    pa_layer = np.flip(pa_lay*100) # pressure of full model levels (Pascals)
    pi_layer = np.flip(pa_lev*100) # pressure of full model levels (Pascals)

    ### CALCULATE INPUT CLOUD OPTICAL DEPTH AND EMISSIVITY ###
    dz = 287*((ta_lev[:-1]+ta_lev[1:])/2)*np.log(pa_lev[:-1]/pa_lev[1:])/9.8
    tauin = calc_tau(lwp_lay,cli_lay,dz,rho_lay,ae,De)
    tauSW = np.flip(tauin)
    demLW  = 1. - np.exp(-tauSW/2)            

    ### RUN ICARUS ###################################################
    if icarustype == 'OIS': tau,ctp,ctb = ois_icarus(nlay,pa_layer,pi_layer,ta_layer,qv_layer,tauSW,demLW,int(sst))
    if icarustype == 'AIS': tau,ctp,ctb = ais_icarus(nlay,pa_layer,ta_layer,tauSW,demLW,int(sst))
    if tau<0: tau =np.nan
    if ctp<0: ctp =np.nan
    if ctb<0: ctb =np.nan

    return(tau,ctp,ctb)

### END ##############################################################