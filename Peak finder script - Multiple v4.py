import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

#Array of filenames
scan_rates = [10, 20, 50, 100, 200]
file_names = []
for i in range(len(scan_rates)):
    file_names.append(str(scan_rates[i])+'mVs.txt')

#Make empty dataframes to add values into during the loop
ip_db = pd.DataFrame(columns = ['nu', 'sqrt_nu', 'ip_raw', 'ip', 'Ep'])
CV_db = pd.DataFrame()
background_db = pd.DataFrame()

#Make array of background values - linear regression will use this range to find the background for peak currents
background_E_l = [-0.39,-0.2,-0.2,-0.2,-0.2]
background_E_h = [-0.45,-0.3,-0.3,-0.3,-0.3]

#Define potential range where the peak is
peak_E_l = -0.5 #lower potential in the range where the peak is seen
peak_E_h = -0.9 #higher potential "    "                 "     "

#Define a function that will find the nearest value in a numpy array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #return array[idx] ##This return would give the actual value
    return idx ##This return gives the index of the actual value

for i in range(len(file_names)):#look at each file one at a time
    #Import data
    data = pd.read_csv(file_names[i], sep=';')

    #Remove spaces from column titles
    data.columns = data.columns.str.replace(' ', '_')

    #define columns
    cols = data.columns

    #Find column titles
    column_headers = list(data.columns.values)

    #Find nth CV
    CV_no = str(1)
    CV = data.query('Scan=='+CV_no)

    #Define potential and current
    potential = CV['Potential_applied_(V)']
    current = CV['WE(1).Current_(A)']

    #Select linear range for baseline

    background_i_l = find_nearest(potential[0:int(len(potential)/2)], value=background_E_l[i]) #Get the index for the lower background potential using the find nearest function
    background_i_h = find_nearest(potential[0:int(len(potential)/2)], value=background_E_h[i]) #Get the index for the higher background potential using the find nearest function
    print(background_i_l)
    print(background_i_h)
    base_potential = potential[background_i_l:background_i_h]
    base_current = current[background_i_l:background_i_h]

    #Create arrays ready for the regression
    base_potential = np.array(base_potential)
    base_potential_r = np.array(base_potential).reshape((-1,1))
    base_current = np.array(base_current)
    
    #Create model for the regression
    model = LinearRegression().fit(base_potential_r, base_current)
    intercept = model.intercept_
    slope = model.coef_

    #Create baseline for plot
    base_x = np.arange(np.min(base_potential)*2, np.max(base_potential), 0.1)
    base_y = (base_x*slope)+intercept

    #Find the peak in the CV
    ip_i_l = find_nearest(potential[0:int(len(potential)/2)], value=peak_E_l) #use the find nearest function to find the index for the start of the peak
    ip_i_h = find_nearest(potential[0:int(len(potential)/2)], value=peak_E_h) #"                      "                 "               end of the peak
    ip_range = current[ip_i_l:ip_i_h] #select i range where peak should be 
    Ep_range = potential[ip_i_l:ip_i_h]

    ip_raw = np.min(ip_range) # ip will be the minimum current in this range
    ip_index = np.where(ip_range==ip_raw)[0][0] #find the index corresponding to the minimum of i
    Ep = Ep_range.reset_index(drop=True)[ip_index] # Ep_range needs to have the index reset before the nth item can be called - slicing the xth - yth section of i means the index of irange starts at x! 

    #Subtract the background from ip using the baseline
    i_background = (slope*Ep) + intercept
    ip = ip_raw - i_background

    #Add ip values into the master datafiles for exporting later
    nu = np.divide(scan_rates[i], 1000.0)#convert mV into V
    sqrt_nu = np.power(nu, 0.5)
    new_row = pd.Series({'nu':nu, 'sqrt_nu':sqrt_nu, 'ip_raw':ip_raw, 'ip':ip[0], 'Ep':Ep}) #make a series with the new ip and Ep data
    ip_db = pd.concat([ip_db,new_row.to_frame().T], ignore_index=True) #concat transposes the series, then adds to the bottom of the database as a new row

    #Make a database for plotting all CVs later on
    CV_db = pd.concat([CV_db,potential.rename('E_'+str(scan_rates[i]))], axis=1)#rename the series as they are added as new names to the database
    CV_db = pd.concat([CV_db,current.rename('i_'+str(scan_rates[i]))], axis=1)

    #Add potential and current to the background database for all of the baselines
    base_x = pd.DataFrame(base_x, columns = ['E_'+str(scan_rates[i])])#still use mV in the titles for scan rates
    background_db = pd.concat([background_db, base_x], axis=1)
    base_y = pd.DataFrame(base_y, columns = ['i_'+str(scan_rates[i])])
    background_db = pd.concat([background_db, base_y], axis=1)
    
#reset the ip database index - they will all be 0 after the concat
ip_db = ip_db.reset_index(drop=True)
print(ip_db)

#Regression on ip vs nu1/2 to find D

all_sqrt = np.array(ip_db['sqrt_nu'])
all_sqrt_r = np.array(all_sqrt).reshape((-1,1))
all_ip = np.array(ip_db['ip'])

model = LinearRegression().fit(all_sqrt_r, all_ip)
intercept = model.intercept_
slope = model.coef_

ip_fit = all_sqrt*slope + intercept

#Find diffusion coefficient

#First import constants
radius = 0.1 #cm
F_const = 96485.0 #C mol-1
area = 3.1415*radius*radius #cm2
R_const = 8.314 #J K-1 mol-1
temp = 80+273.15 #K
conc = 5e-6 #mol/cm-3
n_const = 4.0

#Define Randles Sevcik --> ip = 0.4463 n F A c (n F nu D / R T)^1/2

RS_one = 0.4463*n_const*F_const*area*conc
RS_two = np.power((n_const*F_const/(R_const*temp)), 0.5)
sqrt_D = slope / (RS_one*RS_two)
D_const = np.power(sqrt_D, 2)

print('D = '+str(D_const)+' cm2 s-1')


#Plot section

#Build list of things to plot
potential_list = []
current_list = []
for i in range(len(scan_rates)):
    potential_list.append('E_'+str(scan_rates[i]))
    current_list.append('i_'+str(scan_rates[i]))

#build plots
nrow = int(np.power(len(scan_rates), 0.5))
ncol = math.ceil(len(scan_rates)/nrow)

#scale terms
scale_y = 1e-6

#multi plot for all CVs
fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
for i, scan_rates in enumerate(scan_rates):
    ax = plt.subplot(nrow, ncol, i+1)
    ax.plot(CV_db[potential_list[i]], CV_db[current_list[i]])
    ax.plot(background_db[potential_list[i]], background_db[current_list[i]])
    ax.plot(ip_db.iloc[i]['Ep'], ip_db.iloc[i]['ip_raw'], 'go')#retun the ith values from the database of peak currents and potentials 
    ax.set(xlabel='$E$ vs Pt / V', ylabel='i / $\mu$A')
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
#plt.tight_layout()
plt.show()

#Linear fit plot for ip vs nu1/2
fig, ax = plt.subplots()
ax.plot(all_sqrt, ip_fit, 'b--')
ax.plot(all_sqrt, all_ip, 'rx')
ax.set(xlabel=r'$\nu^{1/2}$ / V$^{1/2}$ s$^{-1/2}$', ylabel='$i_p$')

plt.show()
