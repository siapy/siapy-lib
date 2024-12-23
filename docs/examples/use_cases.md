The functionality of the SiaPy library has been implemented in various use cases, demonstrating its capabilities and potential applications. The library's functionality is not limited to these examples and can be extended to other applications as well.

---

### 🚀 SiaPy command line tool (CLI)

To facilitate the use of some of the SiaPy library's functionality, a command line interface (CLI) has been implemented.

The CLI currently supports the following commands:

- Display images from two cameras.
- Co-register cameras and compute the transformation from one camera's space to another.
- Select regions in images for training machine learning (ML) models.
- Perform image segmentation using a pre-trained ML model.
- Convert radiance images to reflectance by utilizing a reference panel.
- Display spectral signatures for in-depth analysis.

/// Info

💻 [Code Repository](https://github.com/siapy/siapy-cli)

///

---

### 🚀 Hyperspectral data utilization in research

This use case demonstrates how to utilize extracted data from hyperspectral images in research.
The project integrates a machine learning (ML) pipeline workflow with the SiaPy library to classify spectral signatures.

Key features:

- Provides a structured approach to train and test models.
- Features an integrated modular architecture for easy modification of models and data.
- Includes an optimization process with hyperparameter tuning.
- Utilizes Explainable AI techniques to understand the model, the data on which the model is trained, and the most relevant spectral bands (important features) for the model.
- Covers the entire process with visualization of results.

/// Info

💻 [Code Repository](https://github.com/Manuscripts-code/Potato-plants-nemdetect--PP-2025)

///

---

### 🥣 Additional information

The SiaPy library is designed to be flexible and extendable, making it suitable for a wide range of applications in hyperspectral imaging and analysis. Whether you are working on research projects, developing machine learning models, or performing detailed spectral analysis, SiaPy provides the tools and functionality needed to achieve your goals.

For more details, see [API](https://siapy.github.io/siapy-lib/api/core/configs/) documentation.
