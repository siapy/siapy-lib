This section provides an overview of key concepts and functionalities of the SiaPy library. You can find code snippets [here](https://github.com/siapy/siapy-lib/tree/main/docs/examples/src).

To follow along with the examples, please download the example data from [Zenodo](https://zenodo.org/records/14534998) and install the library by following the [installation instructions](https://siapy.github.io/siapy-lib/install/).

## 📄 Data used in examples

The hyperspectral images were acquired using Hyspex push-broom cameras from Norsk Elektro Optikk (Oslo, Norway). Two spectral regions were covered:

- VNIR-1600: Visible to near-infrared (400–988 nm) with 160 bands and a bandwidth of 3.6 nm.
- SWIR-384: Short-wave infrared (950–2500 nm) with 288 bands and a bandwidth of 5.4 nm.

The acquired hyperspectral data, expressed as reflectance values, were already radiometrically calibrated to ensure accuracy and reliability for subsequent analysis.

**Description of file names**

| File Name                          | Description                                      |
|------------------------------------|--------------------------------------------------|
| **L1_L2_L3__test__ID_CAM.img**     | Image file                                       |
| **L1_L2_L3__test__ID_CAM.hdr**     | Header file corresponding to the image file      |

| Component | Description                                      |
|-----------|--------------------------------------------------|
| **L**     | Labels of plants in the image                    |
| **ID**    | Random ID assigned when the image is acquired    |
| **CAM**   | Camera type; `corr` for VNIR, `corr2` for SWIR   |

| Labels    | Example       |
|-----------|---------------|
| **V-T-N** | KK-K-04       |

| Component | Description                                      |
|-----------|--------------------------------------------------|
| **V**     | Variety                                          |
|           | `KK`: KIS Krka                                   |
|           | `KS`: KIS Savinja                                |
| **T**     | Treatment                                        |
|           | `K`: Control                                     |
|           | `S`: Drought                                     |
| **N**     | Index of a particular plant                      |

## 🚀 Validation of setup

To verify that the data and the SiaPy installation meet the expected requirements, run the following code snippet. It should print a message indicating whether the loading was successful.

```python
--8<-- "docs/examples/src/spectral_imageset_load_01.py"
```

/// Warning

If the `SiaPy` library is not installed or the environment is not activated, an error will be thrown at the import statement. Additionally, if the data images are not present, the code will generate an empty image set, resulting in the script printing that the loading was not successful. Make sure to change `data_dir` to point to the directory where the images are stored.

///
