# --8<-- [start:long]
import pandas as pd
from rich import print

from siapy.entities import Pixels, Signatures
from siapy.entities.signatures import Signals

# Option 1: Step-by-step initialization
# Initialize Pixels object from a DataFrame
pixels_df = pd.DataFrame({"x": [10, 30], "y": [20, 40]})
pixels = Pixels(pixels_df)

# Initialize Signals object from a DataFrame
signals_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
signals = Signals(signals_df)

# Create Signatures object from Pixels and Signals objects
signatures1 = Signatures(pixels, signals)
# --8<-- [end:long]

# --8<-- [start:short]
# Option 2: Direct initialization with raw data
# Initialize Signatures directly from raw signals and coordinates data
signatures2 = Signatures.from_signals_and_pixels(
    signals=[[1, 2, 3], [4, 5, 6]],
    pixels=[[10, 20], [30, 40]],
)
# --8<-- [end:short]

# --8<-- [start:assert]
# Verify that both approaches produce equivalent results
assert signatures1 == signatures2

# Print the DataFrame representation of Pixels and Signals
df_multi = signatures2.to_dataframe_multiindex()
print(f"MultiIndex DataFrame:\n{df_multi}")
print(f"Signals DataFrame:\n{signatures2.signals.df}")
print(f"Pixels DataFrame:\n{signatures2.pixels.df}")
# --8<-- [end:assert]
