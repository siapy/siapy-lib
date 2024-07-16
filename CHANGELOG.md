# Changelog

## [0.3.3](https://github.com/siapy/siapy-lib/compare/v0.3.2...v0.3.3) (2024-07-16)


### Bug Fixes

* Add shape_type property to Shape class; fix tests ([0fccc9a](https://github.com/siapy/siapy-lib/commit/0fccc9ae75adaafaf4629c97d8d7187a25520a6e))

## [0.3.2](https://github.com/siapy/siapy-lib/compare/v0.3.1...v0.3.2) (2024-07-08)


### Bug Fixes

* Update mypy configuration to ignore missing imports for sklearn module ([15ad789](https://github.com/siapy/siapy-lib/commit/15ad789c45c63c6ed67eb2c103018a7944e8b286))

## [0.3.1](https://github.com/siapy/siapy-lib/compare/v0.3.0...v0.3.1) (2024-07-08)


### Bug Fixes

* Update gitignore to include tests/data directory and ignore E501 in flake8 configuration ([3664bd1](https://github.com/siapy/siapy-lib/commit/3664bd1a1668ff7d4cedfd489961ee69e68baf00))

## [0.3.0](https://github.com/siapy/siapy-lib/compare/v0.2.4...v0.3.0) (2024-07-06)


### Features

* Add camera_id property and images_by_camera_id method to SpectralImageSet ([c7814dc](https://github.com/siapy/siapy-lib/commit/c7814dc001e4672caacf300dae2fff2b66c7ffb0))


### Bug Fixes

* add scope to fixtures. ([9e2befd](https://github.com/siapy/siapy-lib/commit/9e2befdeba037c51e3fc0757635e7ed4ec0daae8))
* Update indent_style and indent_size in .editorconfig ([a3288dc](https://github.com/siapy/siapy-lib/commit/a3288dc4d3d9a44a2af1e3a11e29619d3bae097e))

## [0.2.4](https://github.com/siapy/siapy-lib/compare/v0.2.3...v0.2.4) (2024-07-04)


### Documentation

* moved files from .github -&gt; repo root ([b1a3989](https://github.com/siapy/siapy-lib/commit/b1a3989e07d20825414713dddcbe859837f6e281))
* security put in place ([8585538](https://github.com/siapy/siapy-lib/commit/8585538064f0e3eee1c2a76e156a1400d94abbcc))

## [0.2.3](https://github.com/siapy/siapy-lib/compare/v0.2.2...v0.2.3) (2024-07-01)


### Documentation

* fixed annotations, added citation file ([baa3628](https://github.com/siapy/siapy-lib/commit/baa36280f0a95fb6b33006d53a1dcdd67b60bd28))

## [0.2.2](https://github.com/siapy/siapy-lib/compare/v0.2.1...v0.2.2) (2024-06-30)


### Bug Fixes

* adapt codespell settings to siapy project ([70649e4](https://github.com/siapy/siapy-lib/commit/70649e4daa3eeffa3a4947a039025c3e26022802))
* fixed test; linter add; dependencies upgrated ([a8e8e8b](https://github.com/siapy/siapy-lib/commit/a8e8e8b2fea95be54aeebbee3a77a8aafd7c61cd))
* pyptoject.toml rename and add additional files to source pdm build ([951f3f7](https://github.com/siapy/siapy-lib/commit/951f3f747b2ad5062ec266f3b79952b605e06bf5))
* success flag added back and run_id ([4e62181](https://github.com/siapy/siapy-lib/commit/4e62181acb6604258569e1ca95178da2b78c229f))
* Update codespell settings to exclude unnecessary directories ([383792b](https://github.com/siapy/siapy-lib/commit/383792be91f7aeba27febeb53a068fb44f29abcd))


### Documentation

* conventional commits type description ([d4f05bd](https://github.com/siapy/siapy-lib/commit/d4f05bd481e0853751baf2b792789f1506316082))
* remove initial contributing readme ([afd7e0b](https://github.com/siapy/siapy-lib/commit/afd7e0b8e23478f9aaff262c7f6fa09e7c5550e7))

## [0.2.1](https://github.com/siapy/siapy-lib/compare/v0.2.0...v0.2.1) (2024-06-29)


### Bug Fixes

* added missing docs files ([4cd628b](https://github.com/siapy/siapy-lib/commit/4cd628b4da921f736c6d2e9d67b7aa07314c62c3))

## [0.2.0](https://github.com/siapy/siapy-lib/compare/v0.1.1...v0.2.0) (2024-06-29)


### Features

* Add method to get a shape by name in GeometricShapes ([1583a8d](https://github.com/siapy/siapy-lib/commit/1583a8d64ff6a1fcbaac01b34989da0ebcff403f))
* Add pixel selection functionality and lasso tool for image plotting ([fabc43b](https://github.com/siapy/siapy-lib/commit/fabc43b1ce804984d6f6d63aba4c7530241b0b9e))
* Add scikit-image dependency for image processing and add image transformations. ([a464ef7](https://github.com/siapy/siapy-lib/commit/a464ef7c9c361b3161249fe70ca6711367ef32a3))
* Add Shape class for geometric shapes in SpectralImage ([e1456c6](https://github.com/siapy/siapy-lib/commit/e1456c620b4ecc6abe04414453db568669fb8282))
* Add support for saving images with different data types and metadata ([5d6d09c](https://github.com/siapy/siapy-lib/commit/5d6d09c8150435d39326fc3a3893f9ad259549b3))
* Add test for transformation function and affine matrix generation ([c9aa4e3](https://github.com/siapy/siapy-lib/commit/c9aa4e362c72024e894efb96a396a11454055b3e))
* Add u(), v(), and to_numpy() methods to Pixels entity ([e4aa9e2](https://github.com/siapy/siapy-lib/commit/e4aa9e2f1beb5886d4b09ea144112f72766ca6c9))
* Improve SpectralImage class by removing redundant code and optimizing image processing, test added ([ea859f7](https://github.com/siapy/siapy-lib/commit/ea859f74db97f4ea0cfb3e60804f95940c4611c9))
* Move Corregistrator class to its own file and update imports ([5286365](https://github.com/siapy/siapy-lib/commit/5286365f8377c5a632ee5f81a8b00e51dcc9db98))
* Refactor code to use preserve_range=True in image rotation and rescaling functions, implemented merge_by_spectral ([217032b](https://github.com/siapy/siapy-lib/commit/217032b2a8071b9ea37ba486183d7ff1049068a5))
* Refactor Corregistrator class and add Pixels entity ([59aafb0](https://github.com/siapy/siapy-lib/commit/59aafb068c2359a2a48c8ebb9e136431b5a4f855))
* Refactor GeometricShapes class to use a separate class for managing shapes, test added ([5f32e0e](https://github.com/siapy/siapy-lib/commit/5f32e0eabebf6898a8dd5e45dff5d793c1afe432))
* Refactor SignaturesFilter to allow filtering rows and columns ([7d9e03c](https://github.com/siapy/siapy-lib/commit/7d9e03c3c629613f61bad7c141d17758ea6e3fc3))
* Refactor SignaturesFilter to allow filtering rows and columns ([ae2be44](https://github.com/siapy/siapy-lib/commit/ae2be44fd58425c5b885a11fd9c7edd2cdd4785a))
* separated validator functons. implemented test for image transformation ([9fe2c1a](https://github.com/siapy/siapy-lib/commit/9fe2c1a721030b4eedd45fadace5ac7ca03d1c98))


### Bug Fixes

* Refactor code to use Path instead of ConvexHull for FreeDraw convex hull calculation, shapes tests added ([2ddd253](https://github.com/siapy/siapy-lib/commit/2ddd253ac769557f882c84396f8e99873c0637bb))
* rename plotting -&gt; plot ([ac9a906](https://github.com/siapy/siapy-lib/commit/ac9a906f568742525317d19bae77f1e179bc25e4))
* tests renamed to prepend the name of the directory ([f91fe45](https://github.com/siapy/siapy-lib/commit/f91fe4505458d766f967d11df91a663678cf72b5))


### Documentation

* templates ([89e0a48](https://github.com/siapy/siapy-lib/commit/89e0a48154a67fb822cae0695f3e0071900665c3))
