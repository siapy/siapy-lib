import subprocess
import sys

import pytest

from siapy.core.configs import BASE_DIR

EXAMPLES_DIR = BASE_DIR / "docs" / "examples" / "src"


def get_example_files():
    """Get all Python files from examples directory."""
    return sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.manual
@pytest.mark.parametrize("example_path", get_example_files())
def test_example_script(example_path):
    """Test that example script runs without errors."""
    try:
        # Run the example script in a subprocess
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )
        assert result.returncode == 0, (
            f"Example {example_path.name} failed with:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Example {example_path.name} timed out after 60 seconds")
