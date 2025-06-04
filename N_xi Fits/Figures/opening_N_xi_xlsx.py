import pandas as pd
import matplotlib.pyplot as plt


# Edit these as needed
rho=0.7
containing_folder = f'N_xi Fits\Figures' 

# Load dataframe
N_xi_fitted_parameters_fn = f'N_xi Fitted Parameters (rho={rho})'
df = pd.read_excel(rf'{containing_folder}\{N_xi_fitted_parameters_fn}.xlsx')
# Here's all the columns in the df
print("Columns in dataframe:")
for col in df.columns:
    print(col)
# Get peak frequency and corresponding T value arrays for each spcecies
anole_freqs = df[df['Species'] == 'Anole']['Frequency'].values
anole_Ts = df[df['Species'] == 'Anole']['T'].values
owl_freqs = df[df['Species'] == 'Owl']['Frequency'].values
owl_Ts = df[df['Species'] == 'Owl']['T'].values
human_freqs = df[df['Species'] == 'Human']['Frequency'].values
human_Ts = df[df['Species'] == 'Human']['T'].values

# Plot anole
plt.scatter(anole_freqs, anole_Ts)
plt.xlabel('Frequency (Hz)')
plt.ylabel('N_xi')
plt.show()




