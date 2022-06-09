import pandas as pd


def average_signatures(area_of_signatures):
    if area_of_signatures is not None:
        x_center = area_of_signatures.x.min() + (area_of_signatures.x.max()
                                            - area_of_signatures.x.min()) / 2
        y_center = area_of_signatures.y.min() + (area_of_signatures.y.max()
                                                - area_of_signatures.y.min()) / 2
        signatures_mean = [list(area_of_signatures.signature.mean())]

        data = {"x": int(x_center), "y": int(y_center), "signature": signatures_mean}
        return pd.DataFrame(data, columns=["x", "y", "signature"])
    else:
        return list()

def limit_to_bounds(image_shape):
    y_max = image_shape[0]
    x_max = image_shape[1]
    def _limit(points):
        points = points[(points.x >= 0) &
                        (points.y >= 0) &
                        (points.x < x_max) &
                        (points.y < y_max)]
        return points
    return _limit



