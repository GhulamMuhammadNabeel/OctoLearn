#!/usr/bin/env python
"""
Octolearn Pre-Release Verification Script
Verifies all components are ready for publication
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(filename, description):
    """Check if file exists"""
    if Path(filename).exists():
        print(f"✅ {description}: {filename}")
        return True
    else:
        print(f"❌ {description}: {filename} NOT FOUND")
        return False

def check_version():
    """Check if version is set correctly"""
    try:
        import octolearn
        version = octolearn.__version__
        if version == "0.7.4":
            print(f"✅ Version Check: {version}")
            return True
        else:
            print(f"⚠️  Version Check: {version} (expected 0.7.4)")
            return False
    except Exception as e:
        print(f"❌ Version Check Failed: {e}")
        return False

def check_import():
    """Check if package imports correctly"""
    try:
        from octolearn import AutoML
        print(f"✅ Import Check: AutoML imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import Check Failed: {e}")
        return False

def check_git():
    """Check if git is initialized"""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            commits = len(result.stdout.strip().split('\n'))
            print(f"✅ Git Initialization: {commits} commit(s)")
            return True
        else:
            print(f"❌ Git Check Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Git Check Failed: {e}")
        return False

def print_header(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    """Run all checks"""
    print_header("🚀 Octolearn PRE-RELEASE VERIFICATION")
    
    all_passed = True
    
    # Phase 1: File Structure
    print_header("PHASE 1: FILE STRUCTURE")
    checks = [
        ("setup.py", "Setup Configuration"),
        ("pyproject.toml", "Build Configuration"),
        ("README.md", "Documentation"),
        ("LICENSE", "License File"),
        (".gitignore", "Git Ignore"),
        ("ARCHITECTURE.md", "Architecture Docs"),
        ("RELEASE.md", "Release Guide"),
        ("octolearn/__init__.py", "Package Init"),
        ("octolearn/core.py", "Core Module"),
        ("octolearn/experiments/risk_scorer.py", "Risk Scorer Module"),
        ("octolearn/experiments/preprocessing_suggester.py", "Preprocessing Module"),
        ("octolearn/experiments/baseline_importance.py", "Importance Module"),
        ("octolearn/experiments/plot_generator.py", "Plot Generator"),
        ("octolearn/experiments/report_generator.py", "Report Generator"),
    ]
    
    for filename, description in checks:
        if not check_file_exists(filename, description):
            all_passed = False
    
    # Phase 2: Package Metadata
    print_header("PHASE 2: PACKAGE METADATA")
    
    if not check_version():
        all_passed = False
    
    if not check_import():
        all_passed = False
    
    # Phase 3: Version Control
    print_header("PHASE 3: VERSION CONTROL")
    
    if not check_git():
        all_passed = False
    
    # Phase 4: Summary
    print_header("RELEASE READINESS SUMMARY")
    
    if all_passed:
        print("""
✅ ALL CHECKS PASSED - READY FOR PUBLICATION

Next Steps:
1. GitHub Release
   - Create repository: https://github.com/YOUR_USERNAME/octolearn
   - Push: git push -u origin main
   
2. PyPI Release (Build)
   - Clean: rm -r dist/ build/ *.egg-info/
   - Build: python -m build
   - Test: twine upload --repository testpypi dist/*
   - Release: twine upload dist/*

3. Verification
   - Test Install: pip install octolearn
   - Import: python -c "from octolearn import AutoML"

See RELEASE.md for detailed instructions.
        """)
        return 0
    else:
        print("""
⚠️  SOME CHECKS FAILED - PLEASE FIX AND RE-RUN

Review the errors above and correct any issues before publishing.
        """)
        return 1

if __name__ == "__main__":
    sys.exit(main())
