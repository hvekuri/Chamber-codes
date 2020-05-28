import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
from chamber_tools import calc_NRMSE, calc_RMSE, linear_fit, exponential_fit, feksp, feksp_taylor

import gasmet_config as config


def calc_flux(slope, pres, sysvol, temp, collar_area, MM):
    return slope * MM*pres*100 * sysvol/(8.31446*temp*collar_area)*0.001


def get_temperature(date, starttime, endtime):
    hobo = pd.read_csv(
        config.hobo+date+".csv", skiprows=[0])
    hobo = hobo[['Date Time, GMT+03:00',
                 'Temp, °C (LGR S/N: 20145739, SEN S/N: 20145739)']]
    hobo.columns = ["DateTime", "Temp"]
    hobo["DateTime"] = pd.to_datetime(
        hobo['DateTime'], format="%m/%d/%y %I:%M:%S %p")
    hobo["Time"] = hobo["DateTime"].dt.time
    hobo.set_index("DateTime", inplace=True)
    hobo = hobo.between_time(starttime, endtime)
    return np.mean(hobo["Temp"])


def plot_meas(date, collar, gas, secs, gas_ppm, lin_co2_hat, exp_co2_hat, tang):
    fig, ax = plt.subplots()
    ax.scatter(secs, gas_ppm, c="black")
    ax.plot(secs, lin_co2_hat, c="k", label="Linear fit")
    ax.plot(secs, exp_co2_hat, c="blue", label="Exponential fit")
    ax.plot(secs, tang, c="green", label="Tangent")
    ax.set_ylabel("ppm")
    ax.set_xlabel("s")
    plt.legend(loc='best')
    plt.title("Collar " + str(collar) + " " + gas)
    # plt.show()
    fig.savefig(config.results +
                str(date)+"_"+str(collar)+"_"+gas+".png")
    plt.close()


if __name__ == "__main__":
    metadata = pd.read_csv(
        config.metadata, delimiter=";")
    for p in ["Start", "End"]:
        metadata[p] = pd.to_datetime(
            metadata[p], format="%H:%M:%S")
    metadata.Date = pd.to_datetime(
        metadata.Date, format="%d.%m.%Y").dt.date
    MM = [44.013, 44.01, 16.04]
    loc = 0
    fluxes = pd.DataFrame(columns=["Date", "Experiment", "Collar", "Gas", "Flux_lin [mg m-2 s-1]",
                                   "NRMSE_lin", "RMSE_lin", "Flux_exp [mg m-2 s-1]", "NRMSE_exp", "RMSE_exp", "Temp"])
    dates = metadata.Date.unique()
    for date in dates:
        print("DATE", date)
        data = pd.read_csv(
            config.data + date.strftime("%Y%m%d")+".txt", delimiter="\t")
        data = data[data["Time"] != "Time"]
        data.Date = pd.to_datetime(data.Date, format="%Y-%m-%d")
        data.set_index(pd.to_datetime(
            data.Time, format="%H:%M:%S"), inplace=True)
        cur_metadata = metadata[(metadata.Date == date)]
        for idx, row in cur_metadata.iterrows():
            S = (row.Start+dt.timedelta(seconds=20)).time()
            E = row.End.time()
            cur_data = data.between_time(S, E)
            for g, gas in enumerate(["Nitrous oxide N2O", "Carbon dioxide CO2", "Methane CH4"]):
                gas_ppm = np.asarray(cur_data[gas], dtype=float)
                time = cur_data.index
                secs = (cur_data.index-cur_data.index[0]).seconds
                try:
                    temp = get_temperature(
                        date.strftime("%Y%m%d"), S, E)+273.15
                except OSError:
                    temp = row.Temp+273.15
                pres = row.Air_pres

                if np.isnan(pres) or np.isnan(temp):
                    print("Temperature or pressure missing. Fill in metadata. Date: " +
                          str(date)+" Collar: "+str(row.Collar))
                    exit()

                sysvol = config.collar_area * \
                    (float(row.Collar_height) -
                     config.gap)+row.Chamber_volume
                lin_slope, lin_nrmse, lin_rmse, lin_co2_hat = linear_fit(
                    gas_ppm, secs)
                lin_flux = calc_flux(lin_slope, pres, sysvol,
                                     temp, config.collar_area, MM[g])
                exp_slope, exp_nrmse, exp_rmse, exp_co2_hat = exponential_fit(
                    gas_ppm, secs)
                exp_flux = calc_flux(exp_slope, pres, sysvol,
                                     temp, config.collar_area, MM[g])
                tang = exp_co2_hat[0]+exp_slope*(secs)
                fluxes.loc[loc] = date, config.experiment, row.Collar, gas, lin_flux, lin_nrmse, lin_rmse, exp_flux, exp_nrmse, exp_rmse, temp
                loc += 1
                if config.plotting:
                    plot_meas(date, row.Collar, gas, secs, gas_ppm, lin_co2_hat,
                              exp_co2_hat, tang)

    fluxes.to_csv(
        config.results+"Fluxes_"+config.experiment+".csv", index=False)
