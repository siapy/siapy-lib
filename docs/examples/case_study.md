This section offers an overview of a case study based on real-world data.

/// Note
> 💡 **All code snippets** used in this case study are available in the [GitHub repository](https://github.com/siapy/siapy-lib/tree/main/docs/examples/src).
///

To follow along:

1. **Download sample data**: Access the hyperspectral image dataset from [Zenodo](https://zenodo.org/records/14534998)
2. **Install SiaPy**: Follow our [installation instructions](https://siapy.github.io/siapy-lib/latest/install/)

### 📋 Data Overview

The hyperspectral dataset used in this case study was acquired using Hyspex push-broom cameras from Norsk Elektro Optikk (Oslo, Norway), covering two spectral regions:

| Camera     | Spectral Range                     | Bands | Bandwidth  |
|------------|-----------------------------------|-------|------------|
| VNIR-1600  | Visible to near-infrared (400–988 nm) | 160   | 3.6 nm     |
| SWIR-384   | Short-wave infrared (950–2500 nm)    | 288   | 5.4 nm     |

/// Note
All hyperspectral data is provided as calibrated reflectance values, ensuring accuracy and reliability for your analysis.
///

### 📁 File Naming Convention

Understanding the file naming convention is crucial for working with the dataset:

```
L1_L2_L3__test__ID_CAM.img   ->  Image file
L1_L2_L3__test__ID_CAM.hdr   ->  Header file corresponding to the image file
```

The file names encode important metadata:

| Component | Description | Example |
|-----------|-------------|---------|
| **V-T-N** (Labels; **L**) | Plant identification | `KK-K-04` |
| **V** (Variety) | Plant variety | `KK`: KIS Krka, `KS`: KIS Savinja |
| **T** (Treatment) | Experimental treatment | `K`: Control, `S`: Drought |
| **N** (Index) | Plant identifier | `04`: Plant #4 |
| **ID** | Random acquisition ID | `18T102331` |
| **CAM** | Camera type | `corr`: VNIR, `corr2`: SWIR |

## 🚀 Validation Setup

Before diving into the examples, verify that your SiaPy installation and data are correctly configured:

```python
--8<-- "docs/examples/src/spectral_imageset_load_01.py"
```

/// Warning
If you encounter issues:

- Check that SiaPy is properly installed and your environment is activated
- Verify that you've downloaded the example data
- Ensure the `data_dir` variable points to the correct location of your dataset
- Make sure both `.img` and `.hdr` files are present in your data directory
///

## 🧪 Source Code Examples

### Working with SpectralImage Objects

**Example:**

```python
--8<-- "docs/examples/src/spectral_image_01.py"
```

**Source: `spectral_image_01.py`**

**Key Concepts:**

- **Multiple Loading Methods**: The script demonstrates two ways to load a hyperspectral image:
  - Using the SpectralPython library directly and wrapping with SiaPy
  - Using SiaPy's simplified `spy_open()` method
- **Property Access**: Shows how to access fundamental image properties through the SpectralImage interface:
  - Shape, dimensions (rows, columns), and spectral bands
  - Wavelength information for each band
  - File metadata and path information
  - Camera identification and associated geometric data

??? note "Implementation Details"
    This script uses decorators for read-only access to image attributes, following the library's design pattern for consistent data access.

**Example:**

```python
--8<-- "docs/examples/src/spectral_image_02.py"
```

**Source: `spectral_image_02.py`**

**Key Concepts:**

- **Array Conversion**: Converting a SpectralImage to NumPy arrays for numerical processing
- **Statistical Analysis**: Computing band-wise statistics like mean intensity values
- **Pixel Handling**: Creating and manipulating Pixels objects from coordinate data
- **Data Extraction**: Methods for extracting spectral signatures and subarrays from specific image regions
- **Visualization**: Converting hyperspectral data to displayable RGB images with optional histogram equalization

??? note "Implementation Details"
    Notice how methods follow SiaPy's naming convention: converters use `to_` prefix (e.g., `to_numpy()`, `to_signatures()`, `to_display()`), while calculation methods use descriptive verbs.

### Managing Image Collections

**Example:**

```python
--8<-- "docs/examples/src/spectral_imageset_01.py"
```

**Source: `spectral_imageset_01.py`**

**Key Concepts:**

- **Batch Loading**: Loading multiple spectral images into a unified SpectralImageSet container
- **Collection Management**: Organizing and sorting images within the set
- **Metadata Access**: Efficient access to collection properties (size, camera IDs)
- **Iteration Patterns**: Iterating through the image collection with standard Python iteration
- **Filtering and Selection**: Selecting images by specific criteria (e.g., camera ID)

??? note "Implementation Details"
    `SpectralImageSet` implements standard Python container interfaces, making it behave like a familiar collection type with additional hyperspectral-specific functionality.

### Interactive Pixel and Area Selection

**Example:**

```python
--8<-- "docs/examples/src/visualization_01.py"
```

**Source: `visualization_01.py`**

The selected pixels are highlighted in the image below.

![Selected pixels](images/pixels_selection.png)

**Key Concepts:**

- **Interactive Selection**: Using SiaPy's interactive tools to select specific pixels from displayed images
- **Point-Based Analysis**: Selecting individual points for detailed spectral analysis
- **User Interaction**: Simple keyboard-based interaction model (press Enter to finish selection)
- **Results Access**: Accessing the resulting Pixels object and its DataFrame representation

??? note "Implementation Details"
    The `pixels_select_click` function handles the display, interaction, and collection of pixel coordinates in a single operation, simplifying user interaction code.

**Example:**

```python
--8<-- "docs/examples/src/visualization_02.py"
```

**Source: `visualization_02.py`**

The selected areas are highlighted in the image below.

![Selected areas](images/areas_selection.png)

**Key Concepts:**

- **Region Selection**: Using SiaPy's lasso tool to define irregular regions of interest
- **Multiple Areas**: Creating and managing multiple selected areas within a single image
- **Polygon-Based Selection**: Defining complex shapes for region-based analysis
- **Selection Management**: Organized representation of selected areas for further processing

??? note "Implementation Details"
    Selected areas are returned as a list of `Pixels` objects, each representing a distinct region that can be separately analyzed or processed.

### Image Transformation and Processing

**Example:**

```python
--8<-- "docs/examples/src/transformations_01.py"
```

**Source: `transformations_01.py`**

**Key Concepts:**

- **Control Point Selection**: Interactive selection of corresponding points in VNIR and SWIR images
- **Coordinate Mapping**: Establishing relationships between points in different spectral spaces
- **Transformation Calculation**: Computing the mathematical transformation between coordinate systems
- **Spatial Alignment**: Creating a foundation for aligning multi-sensor hyperspectral data

??? note "Implementation Details"
    The `corregistrator.align()` function computes a transformation matrix that can transform coordinates from one image space to another, essential for multi-sensor data fusion.

**Example:**

```python
--8<-- "docs/examples/src/transformations_02.py"
```

**Source: `transformations_02.py`**

**Key Concepts:**

- **Area Selection**: Using the lasso tool to select regions in one spectral range
- **Coordinate Transformation**: Applying the transformation matrix to map selected areas between images
- **Cross-Spectral Analysis**: Enabling analysis of the same physical regions across different spectral data
- **Visual Verification**: Displaying both images with highlighted areas to verify correct transformation

??? note "Implementation Details"
    The transformation is applied to the `Pixels` objects directly, allowing selected regions to be mapped between different spectral ranges while preserving their shape relationships.

**Example:**

```python
--8<-- "docs/examples/src/transformations_03.py"
```

**Source: `transformations_03.py`**

**Key Concepts:**

- **Noise Injection**: Adding controlled Gaussian noise for robustness testing or data augmentation
- **Spatial Transformations**: Applying geometric operations including
- **Normalization**: Area-based normalization for standardizing image intensity distributions
- **Data Augmentation**: Creating modified versions of images for machine learning training

??? note "Implementation Details"
    All transformation functions follow a consistent input/output pattern, taking NumPy arrays as input and returning the transformed arrays, making them easily composable for complex processing pipelines.

---

*This case study was created with SiaPy latest version. If you encounter any issues, please check for updates or [report them on GitHub](https://github.com/siapy/siapy-lib/issues/new/choose).*
