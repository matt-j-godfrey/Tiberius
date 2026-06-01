import pickle
import numpy as np

# Load background spectra
sky_spec = pickle.load(open('background_avg_star1.pickle', 'rb'))
sky_spec2 = pickle.load(open('background_avg_star2.pickle', 'rb'))

# Sum along axis 1 to get sky background
sky = sky_spec.sum(axis=1)
sky2 = sky_spec2.sum(axis=1)

# Save sky backgrounds as pickle files
#with open('sky1.pickle', 'wb') as handle:
#    pickle.dump(sky, handle)
#with open('sky2.pickle', 'wb') as handle:
#    pickle.dump(sky2, handle)

with open('sky1.pickle', 'wb') as handle:
    pickle.dump(sky_spec, handle)
with open('sky2.pickle', 'wb') as handle:
    pickle.dump(sky_spec2, handle)

# Load airmass and save as text
airmass = pickle.load(open('airmass.pickle', 'rb'))
np.savetxt('airmass.txt', airmass)

# Load exposure times and save as time.pickle
exposure_times = pickle.load(open('obs_time_array.pickle', 'rb'))
with open('mjd_time.pickle', 'wb') as handle:
    pickle.dump(exposure_times, handle)

# Save as time.txt
np.savetxt('mjd_time.txt', exposure_times)
