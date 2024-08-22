.PHONY: .pdm  ## Check that PDM is installed
.pdm:
	@pdm -V || echo 'Please install PDM: https://pdm.fming.dev/latest/#installation'

.PHONY: .pre-commit  ## Check that pre-commit is installed
.pre-commit:
	@pdm run pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install  ## Install the package, dependencies, and pre-commit for local development
install:
	./scripts/install-dev.sh

.PHONY: refresh-lockfiles  ## Sync lockfiles with requirements files.
refresh-lockfiles: .pdm
	pdm update --update-reuse --group :all

.PHONY: rebuild-lockfiles  ## Rebuild lockfiles from scratch, updating all dependencies
rebuild-lockfiles: .pdm
	pdm update --update-eager --group :all

.PHONY: format  ## Auto-format python source files
format: .pdm
	./scripts/format.sh

.PHONY: lint  ## Lint python source files
lint: .pdm
	./scripts/lint.sh

.PHONY: test  ## Run all tests
test: .pdm
	./scripts/test.sh "Develop"

.PHONY: codespell  ## Use Codespell to do spellchecking
codespell: .pre-commit
	pdm run pre-commit run codespell --all-files

.PHONY: testcov  ## Run tests and generate a coverage report
testcov: test
	@echo "building coverage html"
	@pdm run coverage html
	@echo "building coverage lcov"
	@pdm run coverage lcov

.PHONY: update-branches  ## Update local git branches after successful PR to develop or main branches
update-branches:
	./scripts/update-branches.sh

.PHONY: clean  ## Clear local caches and build artifacts
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .pdm-build
	rm -rf .mypy_cache
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -rf site
	rm -rf docs/_build
	rm -rf coverage.xml

.PHONY: generate-docs  ## Generate the docs
generate-docs:
	pdm run mkdocs build --strict

.PHONY: serve-docs  ## Serve the docs
serve-docs:
	pdm run mkdocs serve

.PHONY: version  ## Check project version
version:
	python -c "import siapy; print(siapy.__version__)"

.PHONY: help  ## Display this message
help:
	@grep -E \
		'^.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'
