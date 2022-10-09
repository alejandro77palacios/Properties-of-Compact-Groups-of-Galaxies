import itertools
import numpy as np
import pandas as pd
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

morpho = pd.read_csv('data/galaxies_morphology.tsv', sep='\t')
mean_group = morpho.groupby('Group').agg(mean_n=('n', 'mean'), mean_T=('T', 'mean'))
groups = pd.read_csv('data/groups.tsv', sep='\t')
combined = mean_group.merge(groups, on='Group')


my_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
coordinates = pd.read_csv('data/galaxies_coordinates.tsv', sep='\t')
angular = groups.groupby('Group')['z'].median().apply(my_cosmo.angular_diameter_distance).apply(lambda x: x.to(u.kpc))

sep_group = {}
for g in coordinates['Group'].unique():
    df = coordinates[coordinates['Group'] == g]
    sep_comb = []
    d = angular[g].value
    for comb in itertools.combinations(df.loc[:, 'Name'], 2):
        df1 = df.loc[df['Name'] == comb[0]]
        df2 = df.loc[df['Name'] == comb[1]]
        p1 = SkyCoord(ra=df1['RA'] * u.degree, dec=df1['DEC'] * u.degree, frame="fk5")
        p2 = SkyCoord(ra=df2['RA'] * u.degree, dec=df2['DEC'] * u.degree, frame="fk5")
        theta = p1.separation(p2).to(u.radian).value[0]
        r = theta * d
        sep_comb.append(r)
    sep_group[g] = np.median(np.array(sep_comb))
mu = groups.set_index('Group')['mean_mu'].dropna()
projected = pd.Series(sep_group)[mu.index]

plt.scatter(projected, mu)
plt.show()

nor_proj = stats.shapiro(projected).pvalue
nor_mu = stats.shapiro(mu).pvalue
pearson_test = stats.pearsonr(projected, mu).pvalue
results = [projected['HCG 2'], nor_proj, nor_mu, pearson_test]
print(*results)
