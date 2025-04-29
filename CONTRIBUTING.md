# Guidelines for contributing

## Design philosophy

`SiaPy` is built with several key design principles:

1. **Type Safety**: Comprehensive type hints throughout the codebase
2. **Modular Design**: Composable components that can be used independently
3. **Consistent APIs**: Uniform interface patterns across the library
4. **Pythonic Interfaces**: Following Python best practices and conventions
5. **Error Handling**: Structured exception hierarchy for clear error reporting

## Code standard

### Dependency management

This project uses [PDM](https://github.com/pdm-project/pdm) for dependency management and packaging.

#### Highlights

- **Automatic virtual environment management**: Automatically manages the application environment.
- **Dependency resolution**: Automatically resolve any dependency version conflicts using the `pip` dependency resolver.
- **Dependency separation**: Supports separate lists of optional dependencies in the _pyproject.toml_. Production installs can skip optional dependencies for speed.
- **Builds**: Features for easily building the project into a Python package and publishing the package to PyPI.

#### Key commands

| Command | Description |
|---------|-------------|
| `pdm init` | Initialize a new project |
| `pdm add PACKAGE_NAME` | Add a package to the project dependencies |
| `pdm install` | Install dependencies from pyproject.toml |
| `pdm list` | Show a list of installed packages |
| `pdm run COMMAND` | Run a command within the PDM environment |
| `pdm shell` | Activate the PDM environment, similar to activating a virtualenv |
| `pdm sync` | Synchronize the project's dependencies |

### Testing with pytest

Tests are located in the _tests_ directory. The project uses [pytest](https://docs.pytest.org/en/latest/) as its testing framework, with configuration stored in _pyproject.toml_.

#### Key features

| Feature | Description | Documentation |
|---------|-------------|---------------|
| `capfd` | Capture stdout/stderr output | [Capturing output](https://docs.pytest.org/en/latest/how-to/capture-stdout-stderr.html) |
| `fixtures` | Reusable test components | [Fixtures](https://docs.pytest.org/en/latest/how-to/fixtures.html) |
| `monkeypatch` | Modify behavior during tests | [Monkeypatching](https://docs.pytest.org/en/latest/how-to/monkeypatch.html) |
| `parametrize` | Run tests with different inputs | [Parametrization](https://docs.pytest.org/en/latest/how-to/parametrize.html) |
| `tmp_path`/`tmp_dir` | Create temporary test files | [Temporary directories](https://docs.pytest.org/en/latest/how-to/tmpdir.html) |

### Code quality

#### Style and format

Python code is formatted with [Ruff](https://docs.astral.sh/ruff/). Ruff configuration is stored in _pyproject.toml_.

#### Static type checking

To learn type annotation basics, see the [Python typing module docs](https://docs.python.org/3/library/typing.html) and [Python type annotations how-to](https://docs.python.org/3/howto/annotations.html).
Type annotations are not used at runtime. The standard library `typing` module includes a `TYPE_CHECKING` constant that is `False` at runtime, but `True` when conducting static type checking prior to runtime. Type imports are included under `if TYPE_CHECKING:` conditions so that they are not imported at runtime.
[Mypy](https://mypy.readthedocs.io/en/stable/) is used for type-checking and it's [configuration](https://mypy.readthedocs.io/en/stable/config_file.html) is included in _pyproject.toml_.

#### Spell check

Spell check is performed with [CSpell](https://cspell.org/). Configuration is stored in _pyproject.toml_.

## Scripts

The project provides several helpful commands through the `make` utility, which mostly calls scripts from the _scripts_ folder. You can run these from the project root directory.

To view all available commands with descriptions, run:

```sh
make help
```

Available commands include:

| Command | Description |
|---------|-------------|
| `make install` | Install the package, dependencies, and pre-commit for local development |
| `make format` | Auto-format python source files |
| `make lint` | Lint python source files |
| `make test` | Run all tests |
| `make flt` | Run format, lint, and test in sequence |
| `make testcov` | Run tests and generate a coverage report |
| `make codespell` | Use Codespell to do spellchecking |
| `make refresh-lockfiles` | Sync lockfiles with requirements files |
| `make rebuild-lockfiles` | Rebuild lockfiles from scratch, updating all dependencies |
| `make clean` | Clear local caches and build artifacts |
| `make generate-docs` | Generate the documentation |
| `make serve-docs` | Serve the documentation locally |
| `make serve-docs-mike` | Serve the documentation using mike |
| `make update-branches` | Update local git branches after successful PR |
| `make version` | Check the current project version |
| `make compress-data` | Compress the data files |

Most commands use PDM (Python package and dependency manager) internally, which will be checked for installation with `.pdm` and `.pre-commit` helper tasks.

## Development

### Git

_[Why use Git?](https://www.git-scm.com/about)_ Git enables creation of multiple versions of a code repository called branches, with the ability to track and undo changes in detail.

Contribute by following:

- Install Git by [downloading](https://www.git-scm.com/downloads) from the website, or with a package manager like [Homebrew](https://brew.sh/).
- [Configure Git to connect to GitHub with SSH](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).
- [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repo.
- Create a [branch](https://www.git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) in your fork.
- Commit your changes with a [properly-formatted Git commit message](https://chris.beams.io/posts/git-commit/).
- Create a [pull request (PR)](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to incorporate your changes into the upstream project you forked.

### Project organization around git

### Branch Structure

| Branch    | Purpose                     | Protection Rules                |
|-----------|-----------------------------|---------------------------------|
| `main`    | Stable production code      | • Signed commits required<br>• No force pushing<br>• Status checks required<br>• Admin included |
| `develop` | Integration branch          | • Signed commits required<br>• Force pushing allowed<br>• Admin included |

### Workflow Rules

- The default branch is `main`
- Feature branches must target `develop` for pull requests
- Only PRs from `develop` may be merged into `main`
- Status checks must pass on `develop` before merging to `main`

#### Commit structure

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#specification) specification for commit messages. This standardizes the commit history and facilitates automatic generation of the changelog.
Ensure your commit messages clearly describe the changes made and follow the format `type(scope?): subject`, where `scope` is optional.

_Type_ must be one of the following:

- `feat`: A new feature
- `fix`: A bug fix
- `perf`: A code change that improves performance
- `deps`: Dependency updates
- `revert`: Reverts a previous commit
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc), hidden
- `chore`: Miscellaneous chores, hidden
- `refactor`: A code change that neither fixes a bug nor adds a feature, hidden
- `test`: Adding missing tests or correcting existing tests, hidden
- `build`: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm), hidden
- `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs), hidden

## GitHub Actions workflows

[GitHub Actions](https://github.com/features/actions) is a continuous integration/continuous deployment (CI/CD) service that runs on GitHub repos. Actions are grouped into workflows and stored in _.github/workflows_.

### Releases

The CI pipeline automatically creates release based on conventional commit messages.

Release version numbers are determined by analyzing commit types:

- `feat`: Increments minor version (1.0.0 → 1.1.0)
- `fix`: Increments patch version (1.0.0 → 1.0.1)
- `feat` with `BREAKING CHANGE`: Increments major version (1.0.0 → 2.0.0)

The following guidelines are considered:

- [SemVer](https://semver.org/) guidelines when choosing a version number. Note that [PEP 440](https://peps.python.org/pep-0440/) Python version specifiers and SemVer version specifiers differ, particularly with regard to specifying prereleases. Use syntax compatible with both.
- The PEP 440 default (like `1.0.0a0`) is different from SemVer.
- An alternative form of the Python prerelease syntax permitted in PEP 440 (like `1.0.0-alpha.0`) is compatible with SemVer, and this form should be used when tagging releases.
- Examples of acceptable tag names: `1.0.0`, `1.0.0-alpha.0`, `1.0.0-beta.1`
- Push to `develop` and verify all CI checks pass.
- Fast-forward merge to `main`, push, and verify all CI checks pass.

## Summary

**PRs welcome!**

- **Consider starting a discussion to see if there's interest in what you want to do.**
- **Submit PRs from feature branches on forks to the `develop` branch.**
- **Ensure PRs pass all CI checks.**
- **Maintain high test coverage (>80%).**
