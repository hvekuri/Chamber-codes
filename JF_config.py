# Folder(s) where licor and rpi files are
folder_licor = "C:/Users/vekuri/Desktop/JIHUU/Licor/"
#folder_licor = "C:/Users/vekuri/Documents/Ruukki 2019/Juusoflux/Kellovaarassa/Licor/"

folder_rpi = "C:/Users/vekuri/Desktop/JIHUU/RPi/"
#folder_rpi = "C:/Users/vekuri/Documents/Ruukki 2019/Juusoflux/Kellovaarassa/RPi/"

# Meta data with experiment, collar name, start and end times for each measurement, collar heights (and LAI if used for GPP)
metadata_path = "C:/Users/vekuri/Documents/Ruukki 2019/Juusoflux/VV_meta.csv"
#metadata_path = "C:/Users/vekuri/Documents/Ruukki 2019/Juusoflux/Kellovaarassa/kellovaarassa_aloitus2.csv"
# Constants for fixing gas analyzer drift
drift_path = "C:/Users/vekuri/Documents/Ruukki 2019/Juusoflux/DriftFix.xlsx"

kellovaarassa = False

# LAI
lai_path = "C:/Users/vekuri/Desktop/LAINURMI.csv"
use_LAI = False
# Quality control
# If PAR sd is greater than this, flux is not used for light response fitting, FEEL FREE TO CHANGE THE LIMITS
par_sd_limit = 150
rmse_limit = 1  # If RMSE is greater than this, flux is not used for LR fitting, FEEL FREE TO CHANGE THE LIMITS
#nrmse_limit = 0.05

# Plotting
plotting = True  # If False, no plots are shown

# Use exponential or linear fits for light response curves, "exponential"/"linear"
use_fit = "linear"

# Where to put results
results_path = "C:/Users/vekuri/Documents/Ruukki 2019/Juusoflux/LAI/"

# Chamber information
chamber_volume = 0.278
