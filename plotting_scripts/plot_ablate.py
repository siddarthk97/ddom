import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# sns.set('seaborn-colorblind')
sns.set()
sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
sns.set_context('paper')
sns.axes_style('ticks')
colors = sns.color_palette('colorblind').as_hex()

plt.rcParams['text.usetex'] = True

super_conds="""
  79.828,5.961 
  82.797,2.786 
  85.168,3.695 
  85.776,3.786 
  84.733,2.803 
  88.248,0.708 
  96.416,6.796 
  99.438,6.776 
  102.610,6.702 
  99.703,6.114 
  97.450,2.765 
  101.991,7.845
"""

ant_conds="""
  487.211,20.645 
  506.525,16.267 
  491.595,23.217 
  499.904,12.410 
  515.524,17.348 
  528.546,30.903 
  522.224,14.875 
  528.380,13.779 
  519.079,28.980 
  500.730,35.503 
  462.046,45.198 
  349.572,94.961
"""

dkitty_conds="""
  235.239,12.334 
  238.045,10.530 
  243.158,10.717 
  251.157,11.801 
  244.195,14.143 
  221.949,18.446 
  171.125,36.261 
  90.543,63.475 
  76.233,102.441 
  5.109,17.990 
  107.847,113.972 
  76.505,63.260
"""

ant_gamma="""
  526.637,16.314 
  526.792,11.528 
  535.457,17.483 
  527.373,12.502 
  530.158,15.462 
  503.827,31.069 
  513.439,34.378 
  509.671,13.976 
  471.123,42.162
"""

super_gamma="""
  91.209,8.477 
  93.789,2.260 
  93.627,6.420 
  100.769,8.477 
  100.141,8.351 
  99.850,6.768 
  97.320,5.837 
  102.678,6.877 
  102.780,6.841
"""

super_steps="""
  86.703,3.644 
  89.024,9.116 
  86.832,3.099 
  89.683,4.713 
  88.248,0.708 
  87.528,1.655 
  88.032,5.829 
  91.200,5.444 
  90.736,5.492 
  88.663,4.736 
  91.219,6.395
"""

ant_steps="""
  509.289,23.658 
  522.775,27.019 
  525.988,9.368 
  504.796,13.498 
  528.546,30.903 
  523.275,16.661 
  518.067,17.447 
  516.279,25.941 
  521.214,14.100 
  512.841,19.038 
  511.025,17.913
"""

dkitty_steps="""
  221.568,20.873 
  231.846,15.442 
  237.661,11.137 
  222.524,18.245 
  221.949,18.446 
  236.747,21.391 
  227.389,17.789 
  233.873,11.746 
  239.779,18.334 
  239.685,13.231 
  228.555,12.848
"""

steps = "10 50 100 500 1000 2000 5000 10000 20000 50000 100000"
steps = steps.split(" ")
steps = [int(x) for x in steps]

conds = "0.0 0.1 0.5 1.0 1.5 2.0 3.0 4.0 5.0 6.0 8.0 10.0" 
conds = conds.split(" ")
conds = [float(x) for x in conds]

gamma = "0.0 0.5 1.0 2.0 3.0 5.0 6.0 8.0 10.0"
gamma = gamma.split(" ")
gamma = [float(x) for x in gamma]

x = gamma
# data = dkitty_gamma.split()
# y = [float(x.split(',')[0]) for x in data]
# plt.plot(x, y)
# plt.savefig('plots/dkitty_gamma.png')
# plt.close()

data = ant_gamma.split()
y = [float(x.split(',')[0]) for x in data]
plt.plot(x, y)
plt.savefig('plots/ant_gamma.png')
plt.close()

data = super_gamma.split()
y = [float(x.split(',')[0]) for x in data]
plt.plot(x, y)
plt.savefig('plots/super_gamma.png')
plt.close()

x = conds
data = dkitty_conds.split()
y = [float(x.split(',')[0]) for x in data]
plt.plot(x, y)
plt.savefig('plots/dkitty_conds.png')
plt.close()

data = ant_conds.split()
y = [float(x.split(',')[0]) for x in data]
plt.plot(x, y)
plt.savefig('plots/ant_conds.png')
plt.close()

data = super_conds.split()
y = [float(x.split(',')[0]) for x in data]
plt.plot(x, y)
plt.savefig('plots/super_conds.png')
plt.close()

x = steps
data = dkitty_steps.split()
y = [float(x.split(',')[0]) for x in data]
plt.xscale("log")
plt.plot(x, y)
plt.savefig('plots/dkitty_steps.png')
plt.close()

data = ant_steps.split()
y = [float(x.split(',')[0]) for x in data]
plt.xscale("log")
plt.plot(x, y)
plt.savefig('plots/ant_steps.png')
plt.close()

data = super_steps.split()
y = [float(x.split(',')[0]) for x in data]
plt.xscale("log")
plt.plot(x, y)
plt.savefig('plots/super_steps.png')
plt.close()
