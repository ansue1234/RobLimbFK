import numpy as np
#import pandas as pd

#Simulated Data Values
BX = -1.4
BY = 8.9
SMA1 = 100
SMA2 = 120
SMA3 = 0
SMA4 = 90
TIME = 1.00

BX1 = None
BY1 = 4.1
BX2 = 10.1
BY2 = 0.4

#check = [BX,BY,SMA1,SMA2,SMA3,SMA4,TIME]
check_list = [[BX,BY,SMA1,SMA2,SMA3,SMA4,TIME],[BX1,BY1,SMA1,SMA2,SMA3,SMA4,TIME],[BX2,BY2,SMA1,SMA2,SMA3,SMA4,TIME]]

maxValue = 150 
minValue = -150

clean_data_iter = []
clean_data_matrix = []

#The clean_bend_data function cleans data from the capacitive bend sensor before being saved for an iteration
def clean_bend_data(value):
    try:
        value = float(value)
    except Exception:
        print(value)
        value = 1
        return value 

    if (value < minValue) or (value > maxValue) or (value == None) or (abs(value) < 0.001) or (value == []):
        print(value)
        value = 1
        return value
    else:
        return value

#The fix_iter function is used to clean data between iterations before being stored in the np_clean_data
def fix_iter(iteration):
    iteration[0] = clean_bend_data(iteration[0])
    iteration[1] = clean_bend_data(iteration[1])
    return iteration

for iteration in check_list:
    clean_data_matrix.append(fix_iter(iteration))


np_clean_data = np.array(clean_data_matrix)

#save data as CSV
np.savetxt('check_list.csv', np_clean_data, delimiter= ',')

#Turn Data into Pandas DataFrame
#df = pd.DataFrame(np_clean_data, columns=['Bend X','Bend Y','SMA 1','SMA 2', 'SMA 3', 'SMA 4', 'Time'])

print(np_clean_data)