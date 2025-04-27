from siapy.entities import Signatures

signatures = Signatures.from_signals_and_pixels(
    signals=[[1, 2, 3], [4, 5, 6]],
    pixels=[[10, 20], [30, 40]],
)

df_multi = signatures.to_dataframe_multiindex()
print(f"MultiIndex DataFrame:\n{df_multi}")
print(f"Signals DataFrame:\n{signatures.signals.df}")
print(f"Pixels DataFrame:\n{signatures.pixels.df}")
