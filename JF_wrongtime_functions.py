import datetime as dt


def find_data(df, diff_must_be, s_e):
    i = 0
    while True:
        diff = (dt.datetime.combine(dt.date(
            1, 1, 1), s_e)-dt.datetime.combine(dt.date(1, 1, 1), df.index[i].time())).total_seconds()
        if abs(diff) < diff_must_be:
            break
        else:
            i += 1

    return i

# Returns lists of co2 and par between s(tart) and e(nd), timestaps (as in 13:08:16, 13:06:21..) amd seconds (0, 5, 10..)


def get_co2_par2(licor, rpi, s, e, a, b):
    s = s.time()
    e = e.time()
    start = find_data(licor, 5, s)
    end = find_data(licor, 5, e)
    cur_licor = licor.iloc[start:end, ]
    start_rpi = find_data(rpi, 10, s)
    end_rpi = find_data(rpi, 10, e)
    cur_rpi = rpi.iloc[start_rpi:end_rpi, ]
    co2 = cur_licor.co2_avg.values.astype(float)
    co2 = a*co2+b  # Fix drift in gas analyzer
    secs_lic = (cur_licor.index-cur_licor.index[0]).seconds
    timestamps_lic = cur_licor.index.time
    par = cur_rpi["PAR(umol/m2/s)"].values.astype(float)
    secs_rpi = (cur_rpi.index-cur_rpi.index[0]).seconds
    timestamps_rpi = cur_rpi.index.time

    return co2, secs_lic, timestamps_lic, par, secs_rpi, timestamps_rpi
