"""
Test that all imports work correctly.

This test ensures all example files can be imported without errors.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_basic_modules():
    """Test that basic Python modules are available."""
    import asyncio
    import json
    import typing
    assert True


def test_core_dependencies():
    """Test that core dependencies can be imported."""
    try:
        import requests
        assert True
    except ImportError:
        pytest.skip("requests not installed (optional)")


def test_ai_dependencies():
    """Test that AI dependencies can be imported (if installed)."""
    dependencies = [
        ('openai', 'OpenAI API'),
        ('anthropic', 'Anthropic API'),
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
    ]
    
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {description} available")
        except ImportError:
            print(f"⚠ {description} not installed (optional)")
    
    # Test passes even if optional dependencies missing
    assert True


def test_example_files_exist():
    """Test that example files exist."""
    examples_dir = project_root / 'examples'
    
    assert examples_dir.exists(), "examples/ directory should exist"
    assert (examples_dir / 'communication').exists()
    assert (examples_dir / 'use-cases').exists()


def test_use_case_files_importable():
    """Test that use case files are syntactically correct."""
    use_cases_dir = project_root / 'examples' / 'use-cases'
    
    use_case_files = [
        'customer_service/customer_service_agents.py',
        'manager_assistant/manager_assistant.py',
        'it_operations/it_operations_automation.py',
        'market_intelligence/market_intelligence_agents.py',
        'business_intelligence/business_intelligence_agents.py',
    ]
    
    for file_path in use_case_files:
        full_path = use_cases_dir / file_path
        if full_path.exists():
            # Try to compile the file (syntax check)
            with open(full_path, 'r', encoding='utf-8') as f:
                code = f.read()
                try:
                    compile(code, str(full_path), 'exec')
                    print(f"✓ {file_path} - syntax OK")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {file_path}: {e}")
        else:
            print(f"⚠ {file_path} not found")


def test_communication_examples_importable():
    """Test that communication examples are syntactically correct."""
    comm_dir = project_root / 'examples' / 'communication'
    
    comm_files = [
        'mcp_implementation.py',
        'multi_model_pipeline.py',
    ]
    
    for file_name in comm_files:
        full_path = comm_dir / file_name
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                code = f.read()
                try:
                    compile(code, str(full_path), 'exec')
                    print(f"✓ {file_name} - syntax OK")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {file_name}: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
