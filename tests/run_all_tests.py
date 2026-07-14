"""Script to run all tests with coverage report."""
import subprocess
import sys


def run_tests():
    """Run pytest with coverage and verbose output."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=app.services",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report",
        "-p", "no:warnings",
    ]

    print("=" * 60)
    print("Running all Smart Match service tests...")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\nAll tests passed!")
    else:
        print(f"\nTests failed with exit code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())