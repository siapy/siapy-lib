<p align="center">
  <img src="https://github.com/siapy/siapy-lib/blob/main/docs/images/logo-text.svg?raw=true" alt="Sublime's custom image" width="500"/>
</p>

<p align="center">
    <em>Spectral imaging analysis for Python (SiaPy) is a tool for efficient processing of spectral images</em>
</p>
<p align="center">
<a href="https://github.com/siapy/siapy-lib/actions?query=workflow%3ATest+event%3Apull_request+branch%3Amain" target="_blank">
    <img src="https://github.com/siapy/siapy-lib/actions/workflows/test.yml/badge.svg?branch=main" alt="Test">
</a>
<a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/siapy/siapy-lib" target="_blank">
    <img src="https://coverage-badge.samuelcolvin.workers.dev/siapy/siapy-lib.svg" alt="Coverage">
</a>
<a href="https://pypi.org/project/siapy" target="_blank">
    <img src="https://img.shields.io/pypi/v/siapy?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://zenodo.org/doi/10.5281/zenodo.7409193"><img src="https://zenodo.org/badge/491829141.svg" alt="DOI"></a>
<a href="https://pypi.org/project/siapy" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/siapy.svg?color=%2334D058" alt="Supported Python versions">
</a>
</p>

---

__Source Code__: <https://github.com/siapy/siapy-lib>

__Bug Report / Feature Request__: <https://github.com/siapy/siapy-lib/issues/new/choose>

<!-- **Tutorials**: <a href="https://github.com/Agricultural-institute/SiaPy/tree/master/tutorials" target="_blank">https://github.com/Agricultural-institute/SiaPy/tree/master/tutorials</a> -->

__Documentation__: <https://siapy.github.io/siapy-lib/>

---

## 📚 Overview

__SiaPy__ is a versatile Python library designed for processing and analyzing spectral images. It is particularly useful for scientific and academic purposes, but it also serves well for quick prototyping.

### Key Features

- __Image Processing__: Easily read, display, and manipulate spectral image data.
- __Data Analysis__: Perform in-depth analysis of spectral signatures using advanced analytical techniques.
- __Machine Learning Integration__: Select image regions for training models and segment images using pre-trained models.
- __Camera Co-registration__: Align multiple cameras and compute transformations across different camera spaces.
- __Radiometric Conversion__: Convert radiance to reflectance using reference panels.

To make some of the functionality more easily accessible, a command line interface (CLI) is also provided. See [siapy-cli](https://github.com/siapy/siapy-cli). However, the full functionality can be exploited by using the library directly.

## 💡 Installation

To install the siapy library, use the following command:

``` bash
pip install siapy
```

For detailed information and additional options, please refer to the [instructions](https://siapy.github.io/siapy-lib/latest/install/).

## 💻 Examples

``` python
from pathlib import Path
from siapy.entities import SpectralImageSet

data_dir = "~/data"

header_paths = sorted(Path(data_dir).rglob("*.hdr"))
image_paths = sorted(Path(data_dir).rglob("*.img"))

imageset = SpectralImageSet.spy_open(
    header_paths=header_paths,
    image_paths=image_paths,
)
print(imageset)
```

For an overview of the key concepts and functionalities of the SiaPy library, please refer to the [documentation](https://siapy.github.io/siapy-lib/latest/concepts/overview/). Additionally, explore the use cases that demonstrate the library's capabilities [here](https://siapy.github.io/siapy-lib/latest/examples/case_study/).

## 🔍 Contribution guidelines

We always welcome small improvements or fixes. If you’re considering making more significant contributions to the source code, please contact us via email.

Contributing to SiaPy isn’t limited to coding. You can also:

- Help us manage and resolve issues, both new and existing.
- Create tutorials, presentations, and other educational resources.
- Propose new features.

Not sure where to start or how your skills might fit in? Don’t hesitate to reach out! You can contact us via email, or connect with us directly on GitHub by opening a new issue or commenting on an existing one.

If you’re new to open-source contributions, check out our [guide](https://siapy.github.io/siapy-lib/latest/contributing/) for helpful tips on getting started.

## 🕐 Issues and new features

Encountered a problem with the library? Please report it by creating an issue on GitHub.

Interested in fixing an issue or enhancing the library’s functionality? Fork the repository, make your changes, and submit a pull request on GitHub.

Have a question? First, ensure that the setup process was completed successfully and resolve any related issues. If you’ve pulled in newer code, you might need to delete and recreate your SiaPy environment to ensure all the necessary packages are correctly installed.

## 🤝 License

This project is licensed under the MIT License. See [LICENSE](https://siapy.github.io/siapy-lib/latest/permit/) for more details.
