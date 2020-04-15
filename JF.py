import pandas as pd
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import math
from scipy import stats
from lmfit import Model, Parameters
import matplotlib
import JF_config as config
#from JF_wrongtime_functions import get_co2_par2 as get_co2_par


class Experiment:
    def __init__(self, name, collar_area, gap):
        self.name = name
        self.collar_area = collar_area
        self.gap = gap


def read_licor(date):
    filename = config.folder_licor+"li840a_"+date+".csv"
    licor = pd.read_csv(filename, names=['start_datetime', 'celltemp_avg', 'celltemp_dev', 'celltemp_min',
                                         'celltemp_max', 'celltemp_med', 'cellpres_avg', 'cellpres_dev',
                                         'cellpres_min', 'cellpres_max', 'cellpres_med', 'co2_avg', 'co2_dev',
                                         'co2_min', 'co2_max', 'co2_med', 'co2abs_avg', 'co2abs_dev',
                                         'co2abs_min', 'co2abs_max', 'co2abs_med', 'h2o_avg', 'h2o_dev',
                                         'h2o_min', 'h2o_max', 'h2o_med', 'h2oabs_avg', 'h2oabs_dev',
                                         'h2oabs_min', 'h2oabs_max', 'h2oabs_med'])
    licor = licor[["start_datetime", "co2_avg", "h2o_avg"]]
    licor = licor[1:]
    licor.index = pd.to_datetime(
        licor["start_datetime"], format="%Y-%m-%d %H:%M:%S")
    return licor


def read_rpi(date):
    filename = config.folder_rpi+"RPi-Data_"+date+".csv"
    rpi = pd.read_csv(filename, names=["TIME", "PAR(umol/m2/s)", "T(C)",
                                       "P(hPa)", "RH(%)", "Latitude(Lat)", "Longitude(Lon)", "Altitude(m)"])
    rpi = rpi[1:]
    rpi.index = pd.to_datetime(
        rpi["TIME"], format="%Y-%m-%d %H:%M:%S")
    return rpi


# Returns lists of co2 and par between s(tart) and e(nd), timestaps (as in 13:08:16, 13:06:21..) amd seconds (0, 5, 10..)
def get_co2_par(licor, rpi, s, e, a, b):
    s = s.time()
    e = e.time()
    co2 = licor.between_time(s, e).co2_avg.values.astype(float)
    co2 = a*co2+b  # Fix drift in gas analyzer
    secs_lic = (licor.between_time(
        s, e).index-licor.between_time(s, e).index[0]).seconds
    timestamps_lic = licor.between_time(s, e).index.time
    par = rpi.between_time(s, e)["PAR(umol/m2/s)"].values.astype(float)
    secs_rpi = (rpi.between_time(
        s, e).index-rpi.between_time(s, e).index[0]).seconds
    timestamps_rpi = rpi.between_time(s, e).index.time

    return co2, secs_lic, timestamps_lic, par, secs_rpi, timestamps_rpi


# Chops 10s from the start of each measurement
# While par_sd>config.par_sd_limit OR rmse>config.rmse_limit, chops 5s from the end of the measurement to find a "good" part of the measurement
# Chopped measurement must be at least 60s long
# Returns new start and end timestamps
def find_good_meas(licor, rpi, s, e, drift_a, drift_b):
    new_s = s+dt.timedelta(seconds=10)
    new_e = e
    co2, secs_lic, _, par, _, _ = get_co2_par(
        licor, rpi, new_s, new_e, drift_a, drift_b)
    # Uses rmse of exponential fit, can be changed here
    if config.use_fit == 'exponential':
        slope, nrmse, rmse, _ = exponential_fit(co2, secs_lic)
    else:
        slope, nrmse, rmse, _ = linear_fit(
            co2, secs_lic)

    while(np.std(par) > config.par_sd_limit or rmse > config.rmse_limit):
        new_e = new_e-dt.timedelta(seconds=5)
        if((new_e-new_s).total_seconds() < 60):
            break
        co2, secs_lic, ts_lic, par, secs_rpi, ts_rpi = get_co2_par(licor, rpi,
                                                                   new_s, new_e, drift_a, drift_b)

        if config.use_fit == 'exponential':
            slope, nrmse, rmse, _ = exponential_fit(co2, secs_lic)
        else:
            slope, nrmse, rmse, _ = linear_fit(
                co2, secs_lic)

    return new_s, new_e


def calc_NRMSE(x, y, y_hat):
    y_min = min(y)
    y_max = max(y)
    sum = 0
    for i in range(len(y)):
        sum += (y_hat[i]-y[i])**2
    return math.sqrt(1/(len(y))*sum)/(y_max-y_min)


def calc_RMSE(x, y, y_hat):
    sum = 0
    for i in range(len(y)):
        sum += (y_hat[i]-y[i])**2
    return math.sqrt(1/(len(y))*sum)


# Does linear fitting, returns slope, nrmse, rmse and predicted co2
def linear_fit(co2, secs_lic):
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        secs_lic, co2)
    co2_hat = intercept+slope*secs_lic
    nrmse = calc_NRMSE(secs_lic, co2, co2_hat)
    rmse = calc_RMSE(secs_lic, co2, co2_hat)
    return slope, nrmse, rmse, co2_hat


# Does exponential fitting first with 17th order taylor series to get initial guesses
# Returns tangent's slope at s=0, nrmse, rmse and predicted co2
def exponential_fit(co2, secs_lic):
    co2_ = np.asarray(co2).reshape(-1, 1)
    t = np.asarray(secs_lic).reshape(-1, 1)

    # Fit Taylor series
    fluxmodel_taylor = Model(feksp_taylor)
    params_T = Parameters()
    params_T.add_many(("a_T", 300), ("b_T", 300), ("c_T", -0.000002))
    result_T = fluxmodel_taylor.fit(
        co2_, secs=t, a=params_T["a_T"], b=params_T["b_T"], c=params_T["c_T"])

    # Fit exponential model
    fluxmodel = Model(feksp)
    params = Parameters()
    params.add_many(("a", result_T.params["a"].value),
                    ("b", result_T.params["b"].value), ("c", result_T.params["c"].value))
    result = fluxmodel.fit(co2_, secs=t, a=params["a"],
                           b=params["b"], c=params["c"])
    # print("Exponential fit result: ", result.fit_report())

    a_fit = result.params["a"].value
    b_fit = result.params["b"].value
    c_fit = result.params["c"].value
    slope = (b_fit-a_fit)*c_fit
    co2_hat = feksp(secs_lic, a_fit, b_fit, c_fit)
    nrmse = calc_NRMSE(secs_lic, co2, co2_hat)
    rmse = calc_RMSE(secs_lic, co2, co2_hat)

    return slope, nrmse, rmse, co2_hat


def feksp(secs, a, b, c):
    return a+(b-a)*np.exp(c*secs)


def feksp_taylor(secs, a, b, c):
    def ftaylorseries(c, x):
        series = 0
        for i in range(17):
            series += (c**i*x**i)/math.factorial(i)
        return series
    return a+(b-a)*ftaylorseries(c, secs)


def calc_flux(slope, pres, sysvol, temp, collar_area):
    return slope * 44.01*pres*100 * sysvol/(8.31446*temp*collar_area)*0.001


def fGPP(PAR, alpha, GPmax):
    return alpha*GPmax*PAR/(alpha*PAR + GPmax)


# Light response fitting
def fit_LR(lr, lai, collar, experiment, date):
    # Define the fit model
    GPPmodel = Model(fGPP)

    # Use Parameter class for model params; this makes it possible to set
    # initial, max & min values for the param fit, or keep a fixed value
    params = Parameters()
    params.add_many(("alpha", -0.001, True, -0.1, 0.000000001), ("GPmax", -1,
                                                                 True, -30, -0.000001))

    # Fit
    result = GPPmodel.fit(lr.GPP/lai, PAR=lr.PAR,
                          alpha=params["alpha"], GPmax=params["GPmax"], method="leastsq")
    # fig, ax = plt.subplots()
    # ax.scatter(lr.PAR, lr.GPP/lai)
    # plt.show()
    # breakpoint()
    #print("LR fit result: ", result.fit_report())

    alpha_fit = result.params['alpha'].value
    GPmax_fit = result.params['GPmax'].value
    alpha_se = result.params['alpha'].stderr
    GPmax_se = result.params['GPmax'].stderr

    # Calculate the fitted function and 2-sigma uncertainty at arbitary x
    xp = np.linspace(0, 2000)
    GPP_fit = fGPP(xp, alpha_fit, GPmax_fit)

    if(result.covar is not None):
        GPP_unc = result.eval_uncertainty(sigma=1.96, PAR=xp)
        GP1200 = lai*fGPP(1200, alpha_fit, GPmax_fit)
        GP1200_unc = result.eval_uncertainty(sigma=1.96, PAR=1200)[0]
    else:
        GP1200, GP1200_unc = np.nan, np.nan

    # Plot
    if config.plotting:
        fig, ax = plt.subplots()
        if(result.covar is not None):
            ax.fill_between(xp, GPP_fit-GPP_unc, GPP_fit +
                            GPP_unc, alpha=0.5, color="grey")
        ax.scatter(lr.PAR, lr.GPP/lai, c="black", s=6)
        ax.plot(xp, GPP_fit, c="black")
        ax.set_title("Collar " + collar)
        ax.set_xlabel("PAR [$\mu$mol m$^-$$^2$ s$^-$$^1$]")
        ax.set_ylabel("GPP [mg CO$_2$ m$^-$$^2$ s$^-$$^1$]")
        ax.grid()
        fig.savefig(config.results_path+experiment +
                    "_LR_"+date+"_"+collar+".png")
        plt.close()

    return alpha_fit, alpha_se, GPmax_fit, GPmax_se, GP1200, GP1200_unc


def plot_meas(date, collar, fig, i, or_ts_lic, or_co2, new_s, new_e, ts_lic, co2_hat, slope, secs_lic, rmse, or_ts_rpi, or_par, par):
    ax = fig.add_subplot(3, 2, i+1)

    ax.set_xlabel("t [s]", fontsize=6)
    ax.set_ylabel("CO$_2$ [ppm]", fontsize=6)
    ax.scatter(or_ts_lic, or_co2,
               color="black", label="co2", s=3)
    if(rmse > config.rmse_limit or np.std(par) > config.par_sd_limit):
        c = "red"
    else:
        c = "black"
    for t in [new_s.time(), new_e.time()]:
        ax.axvline(t, c=c)
    ax.plot(ts_lic, co2_hat, ls="--", c="red",
            label="Fit ("+config.use_fit+")")
    if config.use_fit == 'exponential':
        tang = co2_hat[0]+slope*(secs_lic)
        ax.plot(ts_lic, tang, c="green", label="Tangent")
    ax.set_title("RMSE: "+str(round(rmse, 2))+" PAR sd: " +
                 str(round(np.std(par), 2)), fontsize=6)
    ax2 = ax.twinx()
    ax2.plot(or_ts_rpi, or_par, c="darkblue", label="PAR")

    ax2.set_ylabel("PAR [$\mu$mol m$^-$$^2$ s$^-$$^1$]", fontsize=6)
    ax2.set_ylim(0, 2000)
    ax.legend(loc="upper left", prop={'size': 6})
    ax2.legend(loc="upper right", prop={'size': 6})
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.suptitle("Date: "+str(date)+" Collar: " + collar)


def main():
    lai = pd.read_csv(config.lai_path, delimiter=",")
    lai.Date = pd.to_datetime(lai.Date, format="%Y-%m-%d")
    lai.set_index("Date", inplace=True)

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

    # Class for each experiment with name, collar area, gap in collar (not necessary if collars identical)
    nurmi = Experiment("nurmi", 0.348, 0.025)
    vilja = Experiment("vilja", 0.348, 0.025)
    lajisto = Experiment("lajisto", 0.297, 0.035)
    leikkuu = Experiment("leikkuu", 0.297, 0.035)

    # Experiments to go through #, vilja, lajisto, leikkuu
    experiments = [nurmi]
    z = 0
    for experiment in experiments:
        # Dataframe where all fluxes are stored (despite par sd and rmse)
        all_fluxes = pd.DataFrame(columns=["Date", "Experiment", "Collar",
                                           "PAR", "PAR_sd", "NEE", "NRMSE", "RMSE", "LAI", "Air_temp"])
        lightresponse_results = pd.DataFrame(
            columns=["Date", "Experiment", "Collar", "Alpha_fit", "Alpha_SE", "GPmax_fit", "GPmax_se", "GP1200", "GP1200unc", "Reco", "LAI", "Air_Temp"])
        metadata_exp = metadata[metadata.Experiment == experiment.name]
        dates = metadata_exp.Date.unique()
        for date in dates:

            if config.use_LAI:
                LAI = lai.loc[date, 'LAI_scaled']
            else:
                LAI = 1
            cur_metadata = metadata_exp[metadata_exp.Date == date]
            drift_a = driftfix[driftfix.Date == pd.Timestamp(date)].a.values[0]
            drift_b = driftfix[driftfix.Date == pd.Timestamp(date)].b.values[0]
            licor = read_licor(str(date))
            rpi = read_rpi(str(date))

            for idx, row in cur_metadata.iterrows():
                print("Date: ", str(date), " Collar: ", row.Collar)

                lr = pd.DataFrame(columns=["PAR", "NEE", "NRMSE", "PAR_sd"])
                for i in range(len(starts)):
                    if pd.isnull(row[starts[i]]):
                        continue

                    # Original start and end of measurement
                    s = row[starts[i]]
                    e = row[ends[i]]

                    or_co2, or_secs_lic, or_ts_lic, or_par, or_secs_rpi, or_ts_rpi = get_co2_par(licor, rpi,
                                                                                                 s, e, drift_a, drift_b)

                    # Chopped "good" measurement
                    new_s, new_e = find_good_meas(
                        licor, rpi, s, e, drift_a, drift_b)

                    # breakpoint()
                    co2, secs_lic, ts_lic, par, secs_rpi, ts_rpi = get_co2_par(licor, rpi,
                                                                               new_s, new_e, drift_a, drift_b)
                    # breakpoint()
                    if config.kellovaarassa == True:
                        print("JOUTUVAT LEIKKURIIN")
                        licor_ix = (licor.index.time ==
                                    ts_lic[len(ts_lic)-1]).argmax()
                        licor = licor.iloc[licor_ix:, ]
                        # breakpoint()
                        rpi_ix = (rpi.index.time ==
                                  ts_rpi[len(ts_rpi)-1]).argmax()
                        rpi = rpi.iloc[rpi_ix:, ]
                        # print(rpi)
                        # print("JUHUU")
                        # print(licor)
                        # breakpoint()
                    # breakpoint()
                    temp = np.mean(rpi.between_time(new_s.time(), new_e.time())[
                                   "T(C)"].values.astype(float)+273.15)
                    pres = np.mean(rpi.between_time(new_s.time(), new_e.time())[
                                   "P(hPa)"].values.astype(float))
                    par_median = np.median(par)

                    # If no temperature or pressure data
                    if np.isnan(temp):  # FIX THIS
                        temp = 293
                    if np.isnan(pres):  # FIX THIS
                        pres = 1000

                    # System volume, collar-gap + chamber
                    sysvol = experiment.collar_area * \
                        (float(row.Collar_height)*0.01 -
                         experiment.gap)+config.chamber_volume

                    if config.use_fit == 'linear':
                        # Linear fit
                        slope, nrmse, rmse, co2_hat = linear_fit(
                            co2, secs_lic)
                        flux = calc_flux(slope, pres, sysvol,
                                         temp, experiment.collar_area)

                    elif config.use_fit == 'exponential':
                        # Exponential fit
                        slope, nrmse, rmse, co2_hat = exponential_fit(
                            co2, secs_lic)
                        flux = calc_flux(slope, pres, sysvol,
                                         temp, experiment.collar_area)
                    else:
                        print(
                            "Check use_fit in config file! Must be exponential or linear.")
                        break

                    if par_median < 5:
                        resp = flux

                    # Save fluxes
                    all_fluxes.loc[z] = date, row.Experiment, row.Collar, par_median, np.std(
                        par), flux, nrmse, rmse, row.LAI, temp
                    z += 1

                    if(rmse < config.rmse_limit and np.std(par) < config.par_sd_limit):
                        lr.loc[i] = par_median, flux, nrmse, np.std(par)

                    # Plot
                    if config.plotting:
                        if i == 0:
                            fig = plt.figure(figsize=(8.0, 5.0))
                        plot_meas(date, row.Collar, fig, i, or_ts_lic, or_co2, new_s, new_e,
                                  ts_lic, co2_hat, slope, secs_lic, rmse, or_ts_rpi, or_par, par)
                if config.plotting:
                    fig.tight_layout()
                    fig.savefig(config.results_path+experiment.name +
                                "_"+str(date)+"_"+row.Collar+".png", dpi=100, bbox_inches='tight')
                    plt.close()
                lr["GPP"] = lr["NEE"]-resp  # GPP = NEE-R

                if(len(lr) > 3):  # More than three points required for light response
                    Alpha_fit, Alpha_se, GPmax_fit, GPmax_se, GP1200, GP1200unc = fit_LR(
                        lr, LAI, str(row.Collar), experiment.name, str(date))

                    # Save parameters
                    lightresponse_results.loc[idx] = date, experiment.name, row.Collar, Alpha_fit, Alpha_se, GPmax_fit, GPmax_se, GP1200, GP1200unc, resp, LAI, temp

        print(lightresponse_results)
        lightresponse_results.to_csv(
            config.results_path+"LR_"+experiment.name+".csv", index=False)
        all_fluxes.to_csv(config.results_path+experiment.name +
                          ".csv", index=False)


if __name__ == "__main__":
    main()
