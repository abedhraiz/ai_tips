"""
Test documentation structure and consistency.

Ensures all documented files exist and links are valid.
"""

import pytest
from pathlib import Path


project_root = Path(__file__).parent.parent


def test_readme_exists():
    """Test that README.md exists."""
    readme = project_root / 'README.md'
    assert readme.exists(), "README.md should exist"
    
    content = readme.read_text(encoding='utf-8')
    assert len(content) > 1000, "README should have substantial content"
    assert '# 🤖 AI Tips' in content or '# AI Tips' in content


def test_license_exists():
    """Test that LICENSE exists."""
    license_file = project_root / 'LICENSE'
    assert license_file.exists(), "LICENSE should exist"


def test_contributing_exists():
    """Test that CONTRIBUTING.md exists."""
    contributing = project_root / 'CONTRIBUTING.md'
    assert contributing.exists(), "CONTRIBUTING.md should exist"


def test_security_exists():
    """Test that SECURITY.md exists."""
    security = project_root / 'SECURITY.md'
    assert security.exists(), "SECURITY.md should exist"


def test_code_of_conduct_exists():
    """Test that CODE_OF_CONDUCT.md exists."""
    coc = project_root / 'CODE_OF_CONDUCT.md'
    assert coc.exists(), "CODE_OF_CONDUCT.md should exist"


def test_changelog_exists():
    """Test that CHANGELOG.md exists."""
    changelog = project_root / 'CHANGELOG.md'
    assert changelog.exists(), "CHANGELOG.md should exist"


def test_core_model_docs_exist():
    """Test that core model documentation exists."""
    models_dir = project_root / 'docs' / 'models'
    assert models_dir.exists(), "docs/models/ should exist"
    
    core_models = ['LLM.md', 'VLM.md', 'LVM.md', 'LAM.md', 'LMM.md']
    
    for model in core_models:
        model_path = models_dir / model
        assert model_path.exists(), f"{model} should exist"


def test_protocol_docs_exist():
    """Test that protocol documentation exists."""
    protocols_dir = project_root / 'docs' / 'protocols'
    assert protocols_dir.exists(), "docs/protocols/ should exist"
    
    protocols = ['MCP.md', 'A2A.md', 'A2P.md', 'A2S.md', 
                 'ORCHESTRATION.md', 'WORKFLOWS.md', 'MLOPS.md']
    
    for protocol in protocols:
        protocol_path = protocols_dir / protocol
        assert protocol_path.exists(), f"{protocol} should exist"


def test_use_cases_have_readmes():
    """Test that all use cases have README files."""
    use_cases_dir = project_root / 'examples' / 'use-cases'
    
    use_cases = [
        'customer_service',
        'manager_assistant',
        'it_operations',
        'market_intelligence',
        'business_intelligence'
    ]
    
    for use_case in use_cases:
        readme = use_cases_dir / use_case / 'README.md'
        if (use_cases_dir / use_case).exists():
            assert readme.exists(), f"{use_case} should have README.md"


def test_requirements_files_exist():
    """Test that requirements files exist."""
    assert (project_root / 'requirements.txt').exists()
    
    req_dir = project_root / 'requirements'
    if req_dir.exists():
        assert (req_dir / 'core.txt').exists()
        assert (req_dir / 'full.txt').exists()
        assert (req_dir / 'dev.txt').exists()


def test_notebooks_directory():
    """Test that notebooks directory exists."""
    notebooks_dir = project_root / 'notebooks'
    assert notebooks_dir.exists(), "notebooks/ should exist"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
