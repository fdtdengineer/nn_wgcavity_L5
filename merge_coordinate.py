#%%
import numpy as np
import pandas as pd

# Config
filepath_coordinates = 'input\\coordinates\\'
filename_coordinates = 'randn_mat.csv'

#num_params = 13
#num_dims = 2
#num_samples = 1000


# Data Loadings
df_coordinates = pd.read_csv(filepath_coordinates + filename_coordinates, header=None)
npr_coordinates = df_coordinates.to_numpy()
#npr_x = npr_coordinates.reshape(num_samples, num_params,num_dims)
npr_x = npr_coordinates.reshape(-1,1,14,2)
filepath_npy = 'npy\\'
np.save(filepath_npy + 'dataset.npy', npr_x)


''' # working space
# npr_x: (1000, 1, 13, 2)
# x_to_conv: (1000,2,5,13)

x_raw = npr_x
num_data = x_raw.shape[0]
xq = np.concatenate([np.zeros([num_data,1,1,2]), x_raw], axis=2)
xq2 = xq.reshape(num_data, 2,-1,2).transpose(0,1,3,2)
x_half = np.concatenate([xq2[:,:,:,::-1][:,:,:,:-1],xq2], axis=3)
x_all = np.concatenate([x_half[:,:,::-1], np.zeros([num_data,2,1,13]), x_half], axis=2)

'''


print("Done.")

