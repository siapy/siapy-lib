import pandas as pd

def average_signatures(area_of_signatures):
    x_center = area_of_signatures.x.min() + (area_of_signatures.x.max()
                                          - area_of_signatures.x.min()) / 2
    y_center = area_of_signatures.y.min() + (area_of_signatures.y.max()
                                            - area_of_signatures.y.min()) / 2
    signatures_mean = [list(area_of_signatures.signature.mean())]

    data = {"x": int(x_center), "y": int(y_center), "signature": signatures_mean}
    return pd.DataFrame(data, columns=["x", "y", "signature"])

