This section walks you through some use cases to help you clarify the concepts and functionalities of the SiaPy library. You can find code snippets [here](https://github.com/siapy/siapy-lib/tree/main/docs/examples/src).

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

## Examples

### Spectral image

This example demonstrates how to load and inspect a spectral image using the SiaPy library:

- Load a spectral image.
- Access various properties and metadata.

```python
--8<-- "docs/examples/src/spectral_image_01.py"
```

**Source: `spectral_image_01.py`**

---

This example demonstrates how to perform various operations on a spectral image:

- Convert a spectral image to a NumPy array.
- Calculate the mean value per band.
- Create a `Pixels` object from pixel coordinates.
- Extract spectral signatures and subarrays.
- Display the spectral image.

```python
--8<-- "docs/examples/src/spectral_image_02.py"
```

**Source: `spectral_image_02.py`**

### Spectral Image Set

This example demonstrates how to perform various operations on a spectral image set:

- Load a set of spectral images.
- Sort the spectral images.
- Get the number of images in the set.
- Retrieve the camera IDs.
- Iterate over images and print their shapes.
- Get images by camera ID.

```python
--8<-- "docs/examples/src/spectral_imageset_01.py"
```

**Source: `spectral_imageset_01.py`**

### Pixels

This example shows how to select individual pixels in an image.

```python
--8<-- "docs/examples/src/visualization_01.py"
```

**Source: `visualization_01.py`**

The selected pixels are highlighted in the image below.

![Selected pixels](images/pixels_selection.png)

---

### Areas

This example shows how to select areas within an image.

```python
--8<-- "docs/examples/src/visualization_02.py"
```

**Source: `visualization_02.py`**

The selected areas are highlighted in the image below.

![Selected areas](images/areas_selection.png)

### Transformations

This example demonstrates how to perform various transformations on spectral images:

- Select corresponding pixels in VNIR and SWIR images.
- Calculate the transformation matrix.

```python
--8<-- "docs/examples/src/transformations_01.py"
```

**Source: `transformations_01.py`**

---

This example demonstrates how to apply the calculated transformation matrix to spectral images:

- Select areas on VNIR image.
- Apply the transformation matrix to the selected areas.
- Transform the selected pixels from VNIR to SWIR space.
- Display both images with areas.

```python
--8<-- "docs/examples/src/transformations_02.py"
```

**Source: `transformations_02.py`**

---

This example demonstrates how to apply various image transformations:

- Add Gaussian noise to the image.
- Perform random cropping.
- Apply random mirroring.
- Rotate the image randomly.
- Rescale the image.
- Normalize the image area.

```python
--8<-- "docs/examples/src/transformations_03.py"
```

**Source: `transformations_03.py`**
