import numpy as np
import pandas as pd

from siapy.entities import Pixels

# Create from pandas DataFrame
pixels1 = Pixels(pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]}))

# Create from numpy array
pixels2 = Pixels.from_iterable(np.array([[10, 40], [20, 50], [30, 60]]))

# Create from list of coordinates
pixels3 = Pixels.from_iterable([(10, 40), (20, 50), (30, 60)])

# Should be the same
assert pixels1 == pixels2 == pixels3
