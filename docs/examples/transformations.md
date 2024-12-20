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
