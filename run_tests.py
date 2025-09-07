#!/usr/bin/env python3
"""
Test runner script for AI Image Detector.

This script provides convenient ways to run different types of tests
and generate coverage reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install -r requirements-test.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run AI Image Detector tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast", "coverage"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage-html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--test-file",
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Determine test selection
    if args.test_file:
        cmd.append(args.test_file)
    elif args.type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "fast":
        cmd.extend(["-m", "not slow"])
    elif args.type == "coverage":
        cmd.extend([
            "--cov=ai_image_detector",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
        if args.coverage_html:
            cmd.append("--cov-report=html")
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if args.type == "coverage" and args.coverage_html and success:
        print(f"\n📊 HTML coverage report generated in htmlcov/index.html")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
