# Changelog

## [0.7.1](https://github.com/siapy/siapy-lib/compare/v0.7.0...v0.7.1) (2025-04-09)


### Bug Fixes

* update mkdocs.yml to enable version selector for mike plugin and correct inventories key ([44e8c85](https://github.com/siapy/siapy-lib/commit/44e8c853fef5559b2eddd7b4e834d21a92ef6d96))


### Documentation

* update documentation deployment commands to use 'mike' instead of 'mkdocs' ([1fa65b6](https://github.com/siapy/siapy-lib/commit/1fa65b6608ad60e8cf2cae163baaeb3225422490))

## [0.7.0](https://github.com/siapy/siapy-lib/compare/v0.6.0...v0.7.0) (2025-04-09)


### Features

* add to_xarray method to ImageBase and implementations in RasterioLibImage and SpectralLibImage; enhance tests for new functionality ([0663f75](https://github.com/siapy/siapy-lib/commit/0663f75afe6c30e4f140cda8056606e8d2fa9a22))
* enhance Pixels and Signatures classes with __repr__ methods; remove SignaturesFilter class and update related tests ([19d212f](https://github.com/siapy/siapy-lib/commit/19d212fad287ef838a8539e50eb313ac4875fdc5))
* enhance Shape class with get_pixels from convex hull ([fefeca2](https://github.com/siapy/siapy-lib/commit/fefeca2fe004b4d71e81fa667515ae1749fa13ae))
* introduce Shape class and ShapeGeometryEnum for geometric representations ([8e33634](https://github.com/siapy/siapy-lib/commit/8e3363446f13d0d06ac3f4ff6a0a933b76e3b025))
* refactor shape handling; replace Shapefile with Shape and add tests ([381028d](https://github.com/siapy/siapy-lib/commit/381028d8a24ea485fb957e36e6187be575b50979))
* shape handling and pixel coordinates ([f7d48c9](https://github.com/siapy/siapy-lib/commit/f7d48c9eb4e5607567ad257c012455ed4d066578))
* update pixel class; add validators for initialization ([2bd2710](https://github.com/siapy/siapy-lib/commit/2bd2710dbba2e1924a7946b5fc90b2a6efb99d21))
* update SpectralImageSet methods to use spy_open; add rasterio_o… ([ad3aaaf](https://github.com/siapy/siapy-lib/commit/ad3aaafa2b1cf0b11bb1b53b666a67092d2ec083))
* update SpectralImageSet methods to use spy_open; add rasterio_open method and corresponding tests ([b61c2e8](https://github.com/siapy/siapy-lib/commit/b61c2e8f2315f64d9c4e7588174b4bd9067b2d0e))


### Bug Fixes

* add missing XarrayType to the public API in types.py ([eab41ee](https://github.com/siapy/siapy-lib/commit/eab41eeb033c37bb1b76ec8ce5828a1cf96cb0f4))
* add row and column properties to RasterioLibImage class; enhance tests for new properties ([a871485](https://github.com/siapy/siapy-lib/commit/a871485f10fc368fa51c2ada85587b3d6316e9e5))
* implement rasterio_open method in SpectralImage class; add corresponding tests ([2b0e82f](https://github.com/siapy/siapy-lib/commit/2b0e82f4ad93534710e42a6d89b80527281a8fc5))
* update pre-commit installation command to include all hook types ([41faf02](https://github.com/siapy/siapy-lib/commit/41faf023aba8f1971afe9a4509cf3c3e8ad273dd))


### Dependencies

* update pdm.lock and add shapely dependency in pyproject.toml ([534e639](https://github.com/siapy/siapy-lib/commit/534e6396365d9ea0eeca65b5437da267418b3b15))


### Documentation

* enhance copilot instructions with additional naming conventions ([0d7600d](https://github.com/siapy/siapy-lib/commit/0d7600d5af2de6eb92bda46c6cc23439aaee15dc))
* update documentation structure by removing obsolete shapefiles and shapes entries, and adding shape details ([5677b4f](https://github.com/siapy/siapy-lib/commit/5677b4fdd1f7710eb01067a1db50459d7254ff16))

## [0.6.0](https://github.com/siapy/siapy-lib/compare/v0.5.9...v0.6.0) (2025-03-22)


### Features

* add compress-data target to Makefile and enhance compression script with versioning ([910f7e9](https://github.com/siapy/siapy-lib/commit/910f7e91f186e8acfdeff4fbbb5e81b3c1779892))
* add script to compress test data and generate checksum ([865ea02](https://github.com/siapy/siapy-lib/commit/865ea02c9a0a382e3749d3c18fb89685d819ac76))
* add Shapefile entity with loading and geometry handling capabilities; update exceptions for filepath validation ([20bd3fc](https://github.com/siapy/siapy-lib/commit/20bd3fc4f0d96744283af8eb25e7b8f858577993))
* add SpectralLibImage support and update image loading methods ([04e4b72](https://github.com/siapy/siapy-lib/commit/04e4b723e89aba328e19313c00a262967293f0ce))
* add TestDataManager class for managing test data downloads and verification; enhance compress-data script with git tagging ([d00b0da](https://github.com/siapy/siapy-lib/commit/d00b0da3d6fea6921f8d666523c91ff3a4992522))
* implement RasterioLibImage for raster file handling; update ImageBase interface and add tests ([a6df199](https://github.com/siapy/siapy-lib/commit/a6df199dcb6df081e71b14dc8da7860b05e70efa))
* implementation of spectral images base class ([d333c44](https://github.com/siapy/siapy-lib/commit/d333c44125b4ba9d2ad9a1ed7b1e6e03315f81fe))
* integrate xarray support in rasterio and spectral libraries ([37bac16](https://github.com/siapy/siapy-lib/commit/37bac16abe18232cdebded0f7276ea3acd14faf9))


### Bug Fixes

* correct spelling of "coordinate" in Coordinates class annotations ([bb26fef](https://github.com/siapy/siapy-lib/commit/bb26fef3780275edf677b8eb5f78f8adaf1d31d5))
* improve error handling in SpectralLibImage.open method ([4cec71e](https://github.com/siapy/siapy-lib/commit/4cec71e23d7ac14b8d516f943716847ca912c4c5))
* improve test data integrity verification and extraction process in data_manager.py ([9817bd2](https://github.com/siapy/siapy-lib/commit/9817bd2202cd14ffd9a81bbd47cd5fa6c7775adc))
* include docs/examples/src in linting and formatting scripts ([3889e81](https://github.com/siapy/siapy-lib/commit/3889e816ae028bf492174991f4bcee2e61fc7af9))
* optimize image processing in RasterioLibImage and update nan handling in SpectralLibImage ([2ffb19b](https://github.com/siapy/siapy-lib/commit/2ffb19b6d54b779c55fc01311fa479d755202bc2))
* remove Git LFS checkout option from workflow and integrate test data integrity verification in pytest configuration ([e3ad547](https://github.com/siapy/siapy-lib/commit/e3ad547b2493c1bdd286841ddabd121bfad9a07c))
* update pre-commit configuration to include tests directory for linting ([c26cdd0](https://github.com/siapy/siapy-lib/commit/c26cdd0e4a4ce23eeb1c401eb161a7f10d598b1e))
* update Python version in workflow configurations from 3.10 to 3.12 ([381ed36](https://github.com/siapy/siapy-lib/commit/381ed36e2054aefec58ae70e7a50878bcbdc3023))
* update Python version in workflow configurations from 3.10 to 3.12 ([3e4287c](https://github.com/siapy/siapy-lib/commit/3e4287c1f8180d2481a30e290d72dec116d91b9d))
* update Python version in workflow configurations from 3.10 to 3.12 ([db90263](https://github.com/siapy/siapy-lib/commit/db9026360258b2fcd998bb7e2d9b236bb61f0f0d))
* update Python version range in pdm.lock to support 3.10 and below 3.13 ([80187de](https://github.com/siapy/siapy-lib/commit/80187de9479e3a1566cc89b29cdec60122ed7808))
* update type hint for open method in ImageBase class; remove TypeVar ([4234dc5](https://github.com/siapy/siapy-lib/commit/4234dc58f028538edcffb2550786f27a6bff91e1))


### Dependencies

* add geopandas, rasterio, and xarray as dependencies in pyproject.toml ([1c7cfde](https://github.com/siapy/siapy-lib/commit/1c7cfde193c52d3347bf5ba60df215b4afe30f22))
* add types-shapely dependency for improved type checking in linting ([88f0070](https://github.com/siapy/siapy-lib/commit/88f00700ad4ca5550beed4d870c5f6d6394be12b))
* remove unused lint dependencies from pdm.lock and pyproject.toml ([53b00e1](https://github.com/siapy/siapy-lib/commit/53b00e11af62aa7a442953b581ab9ef031974dbb))
* update configuration files for improved performance -- pdm update ([4f060d4](https://github.com/siapy/siapy-lib/commit/4f060d484bd2807ed3c7d9e7a02b821604a408ac))
* update dependencies and add new testing packages in pdm.lock and pyproject.toml ([da716ed](https://github.com/siapy/siapy-lib/commit/da716ed3c319ffd59060dc544528dfaca21276ca))


### Documentation

* reorganize API documentation structure for images and shapefiles ([52e6446](https://github.com/siapy/siapy-lib/commit/52e64465bdce316f87d4afd083dc5b962e50d472))

## [0.5.9](https://github.com/siapy/siapy-lib/compare/v0.5.8...v0.5.9) (2025-01-15)


### Bug Fixes

* rename fit_params to params in Scorer and cross_validation functions ([4ce6a76](https://github.com/siapy/siapy-lib/commit/4ce6a7660e2a8504b8a5a4a14383bfabf7504b20))


### Dependencies

* update configuration files for improved linting and formatting ([ecdc54b](https://github.com/siapy/siapy-lib/commit/ecdc54b6d44416c8a2a73386d390cb635a484830))


### Documentation

* add overview and key features to README.md ([1b1fd70](https://github.com/siapy/siapy-lib/commit/1b1fd70ca237b5046fdf683209707fd8276073cf))

## [0.5.8](https://github.com/siapy/siapy-lib/compare/v0.5.7...v0.5.8) (2024-12-23)


### Documentation

* add use cases documentation for SiaPy library and update navigation ([333bb9d](https://github.com/siapy/siapy-lib/commit/333bb9d4314b8d80b8a9904a81d430948fd62fb9))
* enhance examples with visualization scripts and update navigation ([42fc4b0](https://github.com/siapy/siapy-lib/commit/42fc4b0a634e7b440446bf6ae07438243aea9502))
* update README with installation instructions and add example usage ([5180335](https://github.com/siapy/siapy-lib/commit/51803350783fc30377f5965b5515460ec9e3c7c6))
* update use cases documentation to clarify code repository links ([b205431](https://github.com/siapy/siapy-lib/commit/b2054311e9e24cf50a0a839111384b6173661a2b))

## [0.5.7](https://github.com/siapy/siapy-lib/compare/v0.5.6...v0.5.7) (2024-12-20)


### Documentation

* add example for loading and processing a spectral image set ([2564e88](https://github.com/siapy/siapy-lib/commit/2564e88fed8dc306b9833b509ea0e8e80ab65bbf))
* add example script for loading and using SpectralImage ([24d5a4b](https://github.com/siapy/siapy-lib/commit/24d5a4b629e2ce753f11dbab5af61d8ba720e051))
* add examples for loading and processing spectral images ([3261665](https://github.com/siapy/siapy-lib/commit/3261665bbcc69f90af8e530304af333582563bbb))
* enhance examples for loading and processing spectral images ([09600b0](https://github.com/siapy/siapy-lib/commit/09600b0a65099a89bab1ad5708ae83bffd7514b9))
* update data directory paths in example scripts and add transformations examples ([d2e924f](https://github.com/siapy/siapy-lib/commit/d2e924f30e00d3622856e50c9b36086404f6564e))
* update introduction.md with correct Zenodo link and improve formatting ([15349d6](https://github.com/siapy/siapy-lib/commit/15349d6d094e82358275e67436deedebf3fc3857))

## [0.5.6](https://github.com/siapy/siapy-lib/compare/v0.5.5...v0.5.6) (2024-12-20)


### Dependencies

* update dependencies and metadata in pdm.lock and pyproject.toml ([0ceea41](https://github.com/siapy/siapy-lib/commit/0ceea41a2eb34f26eac3ff190955eb1adaa44105))


### Documentation

* introduction.md ([a6faef3](https://github.com/siapy/siapy-lib/commit/a6faef3b20da728add41dda0666dd9c6240a4f55))

## [0.5.5](https://github.com/siapy/siapy-lib/compare/v0.5.4...v0.5.5) (2024-10-10)


### Bug Fixes

* Update n_jobs default value to use all processors ([e768b93](https://github.com/siapy/siapy-lib/commit/e768b9361ceb3cb49d3e43e970d7a7fc1b514166))

## [0.5.4](https://github.com/siapy/siapy-lib/compare/v0.5.3...v0.5.4) (2024-09-18)


### Bug Fixes

* Refactor, add and correct type annotations ([12ffa05](https://github.com/siapy/siapy-lib/commit/12ffa05a19053dc220c5dbac6c351d2c116716ee))


### Documentation

* Add core.exceptions API documentation and update mkdocs.yml ([ca37d04](https://github.com/siapy/siapy-lib/commit/ca37d04e7d9a914a1a5e7e4b86b2a9c3e43e9b42))

## [0.5.3](https://github.com/siapy/siapy-lib/compare/v0.5.2...v0.5.3) (2024-09-17)


### Dependencies

* pdm update ([5fcea60](https://github.com/siapy/siapy-lib/commit/5fcea604a0311c23af5a18a17d852a19d83d1909))
* To indicate that your library supports type checking ([72425aa](https://github.com/siapy/siapy-lib/commit/72425aaa828a53ed35ca9de10672680a21868678))

## [0.5.2](https://github.com/siapy/siapy-lib/compare/v0.5.1...v0.5.2) (2024-09-03)


### Performance Improvements

* Refactor code to improve performance in test_to_signatures_perf ([39cf7a2](https://github.com/siapy/siapy-lib/commit/39cf7a243171557c3d0c449420d8a2ce4797a88a))


### Documentation

* funding update ([bfdbc65](https://github.com/siapy/siapy-lib/commit/bfdbc65157fbafb4825a5347480d1c48b2a454e7))
* Update funding configuration file ([bfdbc65](https://github.com/siapy/siapy-lib/commit/bfdbc65157fbafb4825a5347480d1c48b2a454e7))

## [0.5.1](https://github.com/siapy/siapy-lib/compare/v0.5.0...v0.5.1) (2024-08-28)


### Bug Fixes

* Comment out cache-related code in docs.yml workflow --temporary fix ([51a8e11](https://github.com/siapy/siapy-lib/commit/51a8e1129964d07c2817ef9c297c665e0fd81d22))
* Remove unused siapy version file and update installation instructions ([8e966e3](https://github.com/siapy/siapy-lib/commit/8e966e3eda31eabbfc7bae96d9ef5a662b99a01b))

## [0.5.0](https://github.com/siapy/siapy-lib/compare/v0.4.9...v0.5.0) (2024-08-28)


### Features

* Add load_from_parquet and save_to_parquet methods to Pixels class ([9d106cb](https://github.com/siapy/siapy-lib/commit/9d106cb1065568b7ec833ff564b7c2b43e11490d))
* Add load_from_parquet and save_to_parquet methods to Signals class ([47ccb2c](https://github.com/siapy/siapy-lib/commit/47ccb2cc6ba582c6affeb4f5c06582a5666eca81))
* Add load_from_parquet and save_to_parquet methods to Signatures class ([ce30e67](https://github.com/siapy/siapy-lib/commit/ce30e676c05b3232c7603f88482d7d306c46c17b))


### Bug Fixes

* Update save_to_parquet method to include index in the saved file ([36981a5](https://github.com/siapy/siapy-lib/commit/36981a51a7e584938ad51593b219584d0a221ccf))
* Update typing annotations in from_paths ([c827862](https://github.com/siapy/siapy-lib/commit/c827862f39f1b704c873273ad56c61f7ebf3aa56))


### Documentation

* Update installation instructions with additional package managers ([bf238a3](https://github.com/siapy/siapy-lib/commit/bf238a390ceb80a02aae969eb1fdaa956092d685))

## [0.4.9](https://github.com/siapy/siapy-lib/compare/v0.4.8...v0.4.9) (2024-08-22)


### Documentation

* Update installation instructions and troubleshooting section ([f5378c1](https://github.com/siapy/siapy-lib/commit/f5378c14cdc83e6f5d96ef32656bb47718733301))

## [0.4.8](https://github.com/siapy/siapy-lib/compare/v0.4.7...v0.4.8) (2024-08-22)


### Reverts

* fix codespell config ([fd2854e](https://github.com/siapy/siapy-lib/commit/fd2854e4390722809ada866b4fbd442d96c97c8c))


### Documentation

* change path to logo image ([b316fc0](https://github.com/siapy/siapy-lib/commit/b316fc01e8d2c3df3a7a18706dbc10b25fd6e192))
* Update page title in mkdocs.yml ([aa1b221](https://github.com/siapy/siapy-lib/commit/aa1b221a0677c80fd2bde5e44106ca6a0433380a))

## [0.4.7](https://github.com/siapy/siapy-lib/compare/v0.4.6...v0.4.7) (2024-08-20)


### Bug Fixes

* Update license to MIT License in docs directly ([4ee1772](https://github.com/siapy/siapy-lib/commit/4ee1772b0820e74f43e6a3d18539244eddf4ff8b))

## [0.4.6](https://github.com/siapy/siapy-lib/compare/v0.4.5...v0.4.6) (2024-08-20)


### Documentation

* Update links in README and mkdocs.yml ([dead861](https://github.com/siapy/siapy-lib/commit/dead8618578a66185418fba087640782800b59ff))

## [0.4.5](https://github.com/siapy/siapy-lib/compare/v0.4.4...v0.4.5) (2024-08-20)


### Bug Fixes

* deploying MkDocs action fail fix ([35e5a7b](https://github.com/siapy/siapy-lib/commit/35e5a7b532c6e31ef2b4864716c6993d4c438abc))

## [0.4.4](https://github.com/siapy/siapy-lib/compare/v0.4.3...v0.4.4) (2024-08-20)


### Documentation

* Add new API documentation files ([19fe2cc](https://github.com/siapy/siapy-lib/commit/19fe2ccaddd3974b254f59fcc709aba0178c9338))
* index docs page update with default readme file ([eb6703b](https://github.com/siapy/siapy-lib/commit/eb6703b95f015bd18bd66ea506f5504d97a5489d))

## [0.4.3](https://github.com/siapy/siapy-lib/compare/v0.4.2...v0.4.3) (2024-08-20)


### Dependencies

* Update dependencies for documentation improvements ([8d98dc1](https://github.com/siapy/siapy-lib/commit/8d98dc13eda13ef8251d6606c2c72688294a7100))


### Documentation

* Make docs for dev ([#111](https://github.com/siapy/siapy-lib/issues/111)) ([8fc65f1](https://github.com/siapy/siapy-lib/commit/8fc65f1f252ef52109f1bfa0a0036fe82d1c9c2a))
* Update site_url in mkdocs.yml ([1fcb70d](https://github.com/siapy/siapy-lib/commit/1fcb70d380bafbf5dd605cc13968f88743c247cb))

## [0.4.2](https://github.com/siapy/siapy-lib/compare/v0.4.1...v0.4.2) (2024-08-18)


### Documentation

* logo image path changed ([76d8850](https://github.com/siapy/siapy-lib/commit/76d885020b73674db6f1b9df08bf969248d63293))

## [0.4.1](https://github.com/siapy/siapy-lib/compare/v0.4.0...v0.4.1) (2024-08-12)


### Bug Fixes

* mypy fix ([0b421fe](https://github.com/siapy/siapy-lib/commit/0b421fe7f91fc5f5dc7c0fc873e2ab2b1fd0f0d9))


### Dependencies

* pdm update ([a49ef58](https://github.com/siapy/siapy-lib/commit/a49ef585c1b2fee97594d2249ae022aa16154dc1))


### Documentation

* Update copyright year to 2024 ([672274f](https://github.com/siapy/siapy-lib/commit/672274fb132be46fde15696587f8485d7c625a07))

## [0.4.0](https://github.com/siapy/siapy-lib/compare/v0.3.4...v0.4.0) (2024-08-10)


### Features

* Add spectral indices computation and calculation functions ([3ef8c63](https://github.com/siapy/siapy-lib/commit/3ef8c636f1ec513819e14e85b3a72741516f5106))
* Features generation implemented - automatic and spectral indices ([16365e0](https://github.com/siapy/siapy-lib/commit/16365e0dd6661520242b32df9b9572960b850ade))


### Bug Fixes

* include ([6d7e9e5](https://github.com/siapy/siapy-lib/commit/6d7e9e5841abe51c413fbf1e7584e94c9dd9ccdc))
* Remove unnecessary dataclass decorator brackets ([28143f3](https://github.com/siapy/siapy-lib/commit/28143f3588670a767ed97e8312f98255cb2f7c72))
* stubs for spyndex, mlxtend, and autofeat ([f89a22a](https://github.com/siapy/siapy-lib/commit/f89a22a9a592cde4072b25ae179c3ce9b3210866))


### Dependencies

* Update dependencies to include spyndex, mlxtend, and autofeat ([edd4978](https://github.com/siapy/siapy-lib/commit/edd4978903cd49b7d2a095778170c5863767538e))

## [0.3.4](https://github.com/siapy/siapy-lib/compare/v0.3.3...v0.3.4) (2024-07-26)


### Bug Fixes

* Create Target base class ([cf7194d](https://github.com/siapy/siapy-lib/commit/cf7194d64f26a9a4ee8fbad5c0b0a3978da2b936))
* set default direction for optimization to 'minimize' ([5a402fd](https://github.com/siapy/siapy-lib/commit/5a402fdb52a8001243f910115801150ed80ba30d))
* Update evaluators.py with type annotations and error handling ([b1a4010](https://github.com/siapy/siapy-lib/commit/b1a40104eb23e094b86d434ca481b190a1772ced))
* Update study config defaults to set direction to 'minimize' ([347a2bb](https://github.com/siapy/siapy-lib/commit/347a2bb6af5807162bccb970a722746b193dbe3f))


### Dependencies

* Update pyproject.toml with optuna&gt;=3.6.1 dependency ([1faac14](https://github.com/siapy/siapy-lib/commit/1faac145ffab5b304f6ca2c23b7a6959d40e11b4))


### Documentation

* Update default branch name to 'main' and adjust merge instructions ([42b0e29](https://github.com/siapy/siapy-lib/commit/42b0e29c7e7ec0b37eba0cb69b3a84831f4614ee))

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
