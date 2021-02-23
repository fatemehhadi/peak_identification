import numpy as np
import matplotlib.pyplot as plt

def Gauss(x, mu, sigma, intensity = 1):
    # x is an array
    # mu is the expected value
    # sigma is the square root of the variance
    # intensity is a multiplication factor
    # This def returns the Gaussian function of x
    gauss_distribution = intensity/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)
    return gauss_distribution

# X-axis (Wavelengths)
first_wavelength = 100
last_wavelength = 200
num_wavelength = 1000
wavelength_range =  np.linspace(first_wavelength, last_wavelength, num_wavelength)

# Four different components
# Component A
mu_a1 = 130
sigma_a1 = 2
intensity_a1 = 1
mu_a2 = 185
sigma_a2 = 2
intensity_a2 = 0.4
mu_a3 = 170
sigma_a3 = 1
intensity_a3 = 1
gauss_a =  Gauss(wavelength_range, mu_a1, sigma_a1, intensity_a1) + Gauss(wavelength_range, mu_a2, sigma_a2, intensity_a2) #+ Gauss(wavelength_range, mu_a3, sigma_a3, intensity_a3)

# Component B
mu_b = 150
sigma_b = 15
intensity_b = 1
gauss_b = Gauss(wavelength_range, mu_b, sigma_b, intensity_b)

# Component C
mu_c1 = 120
sigma_c1 = 2
intensity_c1 = 0.15
mu_c2 = 165
sigma_c2 = 8
intensity_c2 = 1
gauss_c = Gauss(wavelength_range, mu_c1, sigma_c1, intensity_c1) + Gauss(wavelength_range, mu_c2, sigma_c2, intensity_c2)

# Component D
mu_d1 = 115
sigma_d1 = 5
intensity_d1 = 1
mu_d2 = 140
sigma_d2 = 5
intensity_d2 = 0.85
gauss_d = Gauss(wavelength_range, mu_d1, sigma_d1, intensity_d1) + Gauss(wavelength_range, mu_d2, sigma_d2, intensity_d2)

# Spectra normalization:
component_a = gauss_a/np.max(gauss_a)
component_b = gauss_b/np.max(gauss_b)
component_c = gauss_c/np.max(gauss_c)
component_d = gauss_d/np.max(gauss_d)

# Generate the components matrix
components = np.array([component_a, component_b, component_c, component_d])

# Rename the library spectra
X = components
X = X.T
X_size = np.shape(X)
m = X_size[0]
X = np.c_[np.ones(m), X] # Add intercept term to X
X_size = np.shape(X)
n = X_size[1]
#print("np.shape(X)",np.shape(X))

# What concentrations we want them to have in our query spectrum:
c_a = 0.25
c_b = 0.7
c_c = -0.1
c_d = 0.35

# Let's build the spectrum to be studied: The query spectrum
query_spectrum = c_a * component_a + c_b * component_b + c_c *component_c + c_d *component_d

# Let's add it some noise for a bit of realism:
query_spectrum = query_spectrum +  np.random.normal(0, 0.01, len(wavelength_range))
query_spectrum = np.reshape(query_spectrum, (m, 1))

np.savetxt('Data/query_spectrum.txt',query_spectrum)
np.savetxt('Data/wavelength_range.txt',wavelength_range)

i = query_spectrum
w = wavelength_range

plt.plot(w, i)
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (counts)')
plt.savefig('Images/raman-1.png')
plt.show()

num_window = 10
length_window = (last_wavelength-first_wavelength)/num_window
print("window = ", length_window)
print("wavelength_range [0] = ", wavelength_range[0])

num_roots = 0
roots = np.zeros((30,2))
roots2 = np.zeros((30,2))
roots_tuple = []
min_tuple = []
max_tuple = []
first_w = wavelength_range[0] - length_window/2
for iw in range(1, num_window*2-1):
    print('\n', iw)
    first_w = first_w + length_window/2
    last_w = first_w + length_window
    print('range',first_w,last_w)
    ind = (w > first_w) & (w < last_w)
    #ind = (w > 180) & (w < 190)
    w1 = w[ind]
    i1 = i[ind]

    plt.plot(w1, i1, 'b. ')
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Intensity (counts)')
    plt.savefig('Images/'+str(iw)+'raman-2.png')
    #plt.show()

    from scipy.interpolate import UnivariateSpline

    # s is a "smoothing" factor
    sp = UnivariateSpline(w1, i1, k=4, s=2000)

    plt.plot(w1, i1, 'b. ')
    plt.plot(w1, sp(w1), 'r-')
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Intensity (counts)')
    plt.savefig('Images/'+str(iw)+'raman-3.png')
    #plt.show()

    # get the first derivative evaluated at all the points
    d1s = sp.derivative()

    d1 = d1s(w1)

    # we can get the roots directly here, which correspond to minima and
    # maxima.
    minmax = sp.derivative().roots()
    print('Roots = {}'.format(minmax))
    #print(np.shape(minmax))

    if len(minmax)!=0:
        threshold = abs(minmax[0] - first_w)/length_window
        if threshold < 0.1:
            print('threshold',threshold)
            minmax = np.delete(minmax,[0])
            print('minmax',minmax)

    if len(minmax)!=0:
        threshold = abs(minmax[-1] - last_w)/length_window 
        if threshold < 0.1:
            print('threshold',threshold)
            minmax = np.delete(minmax,[-1])
            print('minmax',minmax)
    

    plt.clf()
    plt.plot(w1, d1, label='first derivative')
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('First derivative')
    plt.grid()

    plt.plot(minmax, d1s(minmax), 'ro ', label='zeros')
    plt.legend(loc='best')

    plt.plot(w1, i1, 'b. ')
    plt.plot(w1, sp(w1), 'r-')
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Intensity (counts)')
    plt.plot(minmax, sp(minmax), 'ro ')
    plt.savefig('Images/'+str(iw)+'raman-4.png')
    #plt.show()

    sp_minmax = sp(minmax)
    #print('minmax',np.shape(minmax),'sp_minmax',np.shape(sp_minmax))
    if len(minmax)!=0:
        #print('minmax',minmax)
        #print('sp_minmax',sp_minmax)
        for ir in range(len(minmax)):
            #print('ir',ir)
            #print('minmax[ir]',minmax[ir])
            roots[num_roots,0] = minmax[ir]
            roots[num_roots,1] = sp_minmax[ir]
            num_roots = num_roots + 1
        print('roots',roots)

    # get the second derivative evaluated at all the points
    d2s = d1s.derivative()
    print('Second derivative at roots = {}'.format(d2s(minmax)))

    # identify local minimum
    ind1 = (d2s(minmax) > 0)
    local_minimum = minmax[ind1]
    local_spmin = sp_minmax[ind1]
    print('Local minimum = {}'.format(local_minimum))
    if len(local_minimum)!=0:
        for imin in range(len(local_minimum)):
            min_tuple.append([local_minimum[imin],local_spmin[imin],'mimimum'])

    # identify local  maximum
    ind2 = (d2s(minmax) < 0)
    local_maximum = minmax[ind2]
    local_spmax = sp_minmax[ind2]
    print('Local maximum = {}'.format(local_maximum))
    if len(local_maximum)!=0:
        for imax in range(len(local_maximum)):
            max_tuple.append([local_maximum[imax],local_spmax[imax],'maximum)'])


roots_tuple = sorted(min_tuple + max_tuple)

print("\n roots_tuple",roots_tuple)

print("roots",roots)

lroots_tuple = len(roots_tuple)
iroots_tuple = 0
print('len',lroots_tuple)

while lroots_tuple > iroots_tuple + 1:
    print("\n")
    print(iroots_tuple)
    print(lroots_tuple)
    print(roots_tuple[iroots_tuple][0],roots_tuple[iroots_tuple+1][0])
    threshold = abs(roots_tuple[iroots_tuple][0] - roots_tuple[iroots_tuple+1][0])/length_window
    if threshold < 0.15:
        if roots_tuple[iroots_tuple][2] == roots_tuple[iroots_tuple+1][2]:
            roots_tuple[iroots_tuple][0] = (roots_tuple[iroots_tuple][0]+roots_tuple[iroots_tuple+1][0])/2
            roots_tuple[iroots_tuple][1] = (roots_tuple[iroots_tuple][1]+roots_tuple[iroots_tuple+1][1])/2
            roots_tuple.pop(iroots_tuple + 1)
            lroots_tuple = lroots_tuple - 1
            
        else:
            roots_tuple.pop(iroots_tuple)
            roots_tuple.pop(iroots_tuple)
            lroots_tuple = lroots_tuple - 2
            iroots_tuple = iroots_tuple + 1
        print("roots_tuple",roots_tuple)
    iroots_tuple = iroots_tuple + 1


lroots_tuple = len(roots_tuple)
iroots_tuple = 0
print('len',lroots_tuple)

while lroots_tuple > iroots_tuple + 1:
    if roots_tuple[iroots_tuple][2] == roots_tuple[iroots_tuple+1][2]:
        roots_tuple.pop(iroots_tuple)
        lroots_tuple = lroots_tuple - 1
    iroots_tuple = iroots_tuple + 1
    

for iroots_tuple in range(len(roots_tuple)):
     roots2[iroots_tuple,0] = roots_tuple[iroots_tuple][0]
     roots2[iroots_tuple,1] = roots_tuple[iroots_tuple][1]

print("roots",roots2)

plt.clf()
plt.plot(w, i, 'b. ')
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (counts)')
#plt.plot(roots[0:num_roots,0], roots[0:num_roots,1], 'ro ')
plt.plot(roots2[0:lroots_tuple,0], roots2[0:lroots_tuple,1], 'ro ')
plt.grid()
plt.savefig('Images/raman-5.png')
plt.show()