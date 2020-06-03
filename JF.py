import os
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from lmfit import Model, Parameters
from chamber_tools import calc_flux, calc_NRMSE, calc_RMSE, linear_fit, exponential_fit, feksp, feksp_taylor, fGPP

import JF_config as config


def read_licor(date):
    filename = config.folder_licor+"li840a_"+date+".csv"
    licor = pd.read_csv(filename, names=['start_datetime', 'celltemp_avg', 'celltemp_dev', 'celltemp_min',
                                         'celltemp_max', 'celltemp_med', 'cellpres_avg', 'cellpres_dev',
                                         'cellpres_min', 'cellpres_max', 'cellpres_med', 'co2_avg', 'co2_dev',
                                         'co2_min', 'co2_max', 'co2_med', 'co2abs_avg', 'co2abs_dev',
                                         'co2abs_min', 'co2abs_max', 'co2abs_med', 'h2o_avg', 'h2o_dev',
                                         'h2o_min', 'h2o_max', 'h2o_med', 'h2oabs_avg', 'h2oabs_dev',
                                         'h2oabs_min', 'h2oabs_max', 'h2oabs_med'])
    licor = licor[1:]
    licor["start_datetime"] = pd.to_datetime(
        licor["start_datetime"], format="%Y-%m-%d %H:%M:%S")
    licor.set_index("start_datetime", inplace=True)
    licor = licor.astype(float)
    return licor


def read_rpi(date):
    filename = config.folder_rpi+"RPi-Data_"+date+".csv"
    rpi = pd.read_csv(filename, names=["TIME", "PAR(umol/m2/s)", "T(C)",
                                       "P(hPa)", "RH(%)", "Latitude(Lat)", "Longitude(Lon)", "Altitude(m)"])
    rpi = rpi[1:]
    rpi["TIME"] = pd.to_datetime(
        rpi["TIME"], format="%Y-%m-%d %H:%M:%S")
    rpi.set_index('TIME', inplace=True)
    rpi = rpi.astype(float)
    return rpi


def find_good_meas(cur_licor, cur_rpi, drift_a, drift_b):
    secs = (cur_licor.index-cur_licor.index[0]).seconds
    co2 = drift_a*cur_licor.co2_avg+drift_b
    if config.use_fit == 'linear':
        slope, nrmse, rmse, co2_hat = linear_fit(co2, secs)
        tang = np.nan
    elif config.use_fit == 'exponential':
        slope, nrmse, rmse, co2_hat = exponential_fit(
            co2, secs)
        tang = co2_hat[0]+slope*(secs)
    else:
        print("use_fit must be linear/exponential, fix config-file")
        exit()

    if len(cur_licor) > 12:
        i = 1
        while np.std(cur_rpi["PAR(umol/m2/s)"]) > config.par_sd_limit or rmse > config.rmse_limit:
            cur_licor = cur_licor[:len(cur_licor)-i*2]
            cur_rpi = cur_rpi[:len(cur_rpi)-i]
            secs = (cur_licor.index-cur_licor.index[0]).seconds
            co2 = drift_a*cur_licor.co2_avg+drift_b
            if config.use_fit == 'linear':
                slope, nrmse, rmse, co2_hat = linear_fit(co2, secs)
                tang = np.nan
            else:
                slope, nrmse, rmse, co2_hat = exponential_fit(co2, secs)
                tang = co2_hat[0]+slope*(secs)

            i += 1
            if(len(cur_licor) <= 12):
                break

    return slope, nrmse, rmse, co2_hat, cur_licor, cur_rpi, tang


def plot_meas(fig, i, date, collar, orig_licor, orig_rpi, cur_licor, co2_hat, tang, cur_rpi, rmse, drift_a, drift_b):
    ax = fig.add_subplot(3, 2, i+1)
    ax.set_xlabel("t [s]", fontsize=6)
    ax.set_ylabel("CO$_2$ [ppm]", fontsize=6)
    ax.scatter(orig_licor.index.time, drift_a*orig_licor.co2_avg+drift_b,
               color="black", label="co2", s=3)
    if(rmse > config.rmse_limit or np.std(cur_rpi["PAR(umol/m2/s)"]) > config.par_sd_limit):
        c = "red"
    else:
        c = "black"
    for t in [cur_licor.index[0].time(), cur_licor.index[len(cur_licor)-1].time()]:
        ax.axvline(t, c=c)
    ax.plot(cur_licor.index.time, co2_hat, ls="--", c="k",
            label="Fit ("+config.use_fit+")")
    if config.use_fit == 'exponential':
        ax.plot(cur_licor.index.time, tang, c="green", label="Tangent")
    ax.set_title("RMSE: "+str(round(rmse, 2))+" PAR sd: " +
                 str(round(np.std(cur_rpi["PAR(umol/m2/s)"]), 2)), fontsize=6)
    ax2 = ax.twinx()

    ax2.plot(orig_rpi.index.time,
             orig_rpi["PAR(umol/m2/s)"], c="darkblue", label="PAR")
    ax2.set_ylabel("PAR [$\mu$mol m$^-$$^2$ s$^-$$^1$]", fontsize=6)
    ax2.set_ylim(0, 2000)
    ax.legend(loc="upper left", prop={'size': 6})
    ax2.legend(loc="upper right", prop={'size': 6})
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax2.tick_params(axis='both', which='major', labelsize=6)
    plt.suptitle("Date: "+str(date)+" Collar: " + str(collar), fontsize=10)

    return fig

# Light response fitting


def fit_LR(lr, lai, collar, experiment, date, plotting):
    # Define the fit model
    GPPmodel = Model(fGPP)

    # Use Parameter class for model params
    params = Parameters()
    params.add_many(("alpha", -0.001, True, config.alpha_bounds[0], config.alpha_bounds[1]), ("GPmax", -1,
                                                                                              True, config.GPmax_bounds[0], config.GPmax_bounds[1]))

    # Fit
    result = GPPmodel.fit(lr.GPP/lai, PAR=lr.PAR,
                          alpha=params["alpha"], GPmax=params["GPmax"], method="leastsq")

    alpha_fit = result.params['alpha'].value
    GPmax_fit = result.params['GPmax'].value
    alpha_se = result.params['alpha'].stderr
    GPmax_se = result.params['GPmax'].stderr

    GP1200 = lai*fGPP(1200, alpha_fit, GPmax_fit)

    if(result.covar is not None):
        GP1200_unc = result.eval_uncertainty(sigma=1.96, PAR=1200)[0]
    else:
        GP1200_unc = np.nan

    return alpha_fit, alpha_se, GPmax_fit, GPmax_se, GP1200, GP1200_unc, result


def plot_LR(result, alpha, GPmax, lr, lai, collar, date):
    fig, ax = plt.subplots()
    # Calculate the fitted function and 2-sigma uncertainty at arbitary x
    xp = np.linspace(0, 2000)
    GPP_fit = fGPP(xp, alpha, GPmax)
    if(result.covar is not None):
        GPP_unc = result.eval_uncertainty(sigma=1.96, PAR=xp)
        ax.fill_between(xp, GPP_fit-GPP_unc, GPP_fit +
                        GPP_unc, alpha=0.5, color="grey")
    ax.scatter(lr.PAR, lr.GPP/lai, c="black", s=6)
    ax.plot(xp, GPP_fit, c="black")
    ax.set_title("Collar " + collar)
    ax.set_xlabel("PAR [$\mu$mol m$^-$$^2$ s$^-$$^1$]")
    ax.set_ylabel("GPP [mg CO$_2$ m$^-$$^2$ s$^-$$^1$]")
    ax.grid()
    # plt.show()
    fig.savefig(config.results_path+config.experiment +
                "_LR_"+date+"_"+collar+".png")
    plt.close()


def main():
    # Read metadata
    metadata = pd.read_csv(config.metadata_path, delimiter=";")
    metadata.Date = pd.to_datetime(
        metadata.Date, format="%d.%m.%Y").dt.date
    cols = ["s1", "e1", "s2", "e2", "s3", "e3", "s4", "e4", "s5", "e5"]
    for col in cols:
        metadata[col] = pd.to_datetime(
            metadata[col], format="%H:%M:%S")
    starts = cols[::2]
    ends = cols[1:len(cols):2]

    # Constants for fixing gas analyzer drift
    driftfix = pd.read_excel(config.drift_path)
    driftfix.Date = pd.to_datetime(driftfix.Date, format="%d.%m.%Y")

    z = 0

    # Dataframe where all fluxes are stored (despite par sd and rmse)
    all_fluxes = pd.DataFrame(columns=["Date", "Experiment", "Collar", "Fit",
                                       "PAR_median", "PAR_sd", "NEE [mg CO2 m-2 s-1]", "NRMSE", "RMSE", "LAI", "Temp"])
    lightresponse_results = pd.DataFrame(
        columns=["Date", "Experiment", "Collar", "alpha", "alpha_se", "GPmax", "GPmax_se", "GP1200 [mg CO2 m-2 s-1]", "GP1200_95perc_CI ", "Reco [mg CO2 m-2 s-1]", "LAI", "Temp"])

    for date in metadata.Date.unique():
        # Constants for fixing drift on date
        drift_a = driftfix[driftfix.Date == pd.Timestamp(date)].a.values[0]
        drift_b = driftfix[driftfix.Date == pd.Timestamp(date)].b.values[0]

        # Read data
        licor = read_licor(str(date))
        rpi = read_rpi(str(date))

        # All measurements on date
        cur_metadata = metadata[metadata.Date == date]

        # One row = one collar
        for idx, row in cur_metadata.iterrows():
            print("Date: ", str(date), " Collar: ", str(row.Collar))
            lr = pd.DataFrame(columns=["PAR", "NEE"])

            # If use_LAI = False, GPP is scaled with LAI=1
            if config.use_LAI:
                LAI = row.LAI
            else:
                LAI = 1

            for i in range(len(starts)):
                if pd.isnull(row[starts[i]]):
                    continue
                # Original start and end of measurement
                s = row[starts[i]].time()
                e = row[ends[i]].time()
                orig_licor = licor.between_time(s, e)
                orig_rpi = rpi.between_time(s, e)

                # Remove 10s from the start of both files
                cur_licor = orig_licor[2:]
                cur_rpi = orig_rpi[1:]

                # Try to find a good (par sd < par_sd_limit & rmse < rmse_limit) part of the measurement. Only chopped from end.
                slope, nrmse, rmse, co2_hat, cur_licor, cur_rpi, tang = find_good_meas(
                    cur_licor, cur_rpi, drift_a, drift_b)

                # Temperature, pressure and PAR
                temp = np.median(cur_rpi["T(C)"]+273.15)
                pres = np.median(cur_rpi["P(hPa)"])
                par = np.median(cur_rpi["PAR(umol/m2/s)"])
                par_sd = np.std(cur_rpi["PAR(umol/m2/s)"])

                # If no temperature, air pressure or PAR, read it from metadata
                if np.isnan(temp):
                    temp = row["Temp (C)"]
                    if np.isnan(temp):
                        print("Temperature missing, fill in metadata. Date: " +
                              str(date)+" Collar: "+str(row.Collar))
                        exit()
                if np.isnan(pres):
                    pres = row["Air_pres (hPa)"]
                    if np.isnan(pres):
                        print("Air pressure missing, fill in metadata. Date: " +
                              str(date)+" Collar: "+str(row.Collar))
                        exit()
                if np.isnan(par):
                    par = row.PAR
                    if np.isnan(par):
                        print("PAR missing, fill in metadata. Date: " +
                              str(date)+" Collar: "+str(row.Collar))
                        exit()

                # System volume, collar-gap + chamber
                sysvol = config.collar_area * \
                    (float(row.Collar_height) -
                     config.gap)+row.Chamber_volume

                # Calculate flux
                flux = calc_flux(slope, pres, sysvol,
                                 temp, config.collar_area)

                # par < 5 -> NEE = Reco
                if par < 5:
                    par = 0
                    resp = flux

                # Save fluxes to dataframe
                all_fluxes.loc[z] = date, config.experiment, row.Collar, config.use_fit, par, par_sd, flux, nrmse, rmse, LAI, temp
                z += 1

                # If rmse < rmse_limit and par sd > par_sd_limit, measurement is used for light response fitting
                if(rmse < config.rmse_limit and par_sd < config.par_sd_limit):
                    lr.loc[i] = par, flux

                # Plot flux measurements
                if config.plotting:
                    if i == 0:
                        fig = plt.figure(figsize=(8.0, 5.0))
                    fig = plot_meas(fig, i, date, row.Collar, orig_licor,
                                    orig_rpi, cur_licor, co2_hat, tang, cur_rpi, rmse, drift_a, drift_b)
            if config.plotting:
                fig.tight_layout()
                # plt.show()
                fig.savefig(config.results_path+config.experiment +
                            "_"+str(date)+"_"+str(row.Collar)+".png", dpi=100, bbox_inches='tight')
                plt.close()

            # Do light response fitting
            lr["GPP"] = lr["NEE"]-resp  # GPP = NEE-R
            if(len(lr) > 3):  # More than three points required for light response
                alpha, alpha_se, GPmax, GPmax_se, GP1200, GP1200unc, result = fit_LR(
                    lr, LAI, str(row.Collar), config.experiment, str(date), config.plotting)
                lightresponse_results.loc[idx] = date, config.experiment, row.Collar, alpha, alpha_se, GPmax, GPmax_se, GP1200, GP1200unc, resp, LAI, temp

                # Plot light response
                if config.plotting:
                    plot_LR(result, alpha, GPmax, lr, LAI,
                            str(row.Collar), str(date))

    # Save results as csv-files
    lightresponse_results.to_csv(
        config.results_path+"LR_"+config.experiment+".csv", index=False)
    all_fluxes.to_csv(config.results_path+"Fluxes_"+config.experiment +
                      ".csv", index=False)


if __name__ == "__main__":
    main()
