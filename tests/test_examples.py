"""
Test that examples run without errors (syntax and basic execution).

These are smoke tests - they verify examples don't crash on import/basic execution.
"""

import pytest
import asyncio
from pathlib import Path
import sys
import importlib.util

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None


@pytest.mark.asyncio
async def test_customer_service_agents_structure():
    """Test customer service agents basic structure."""
    # Import and verify classes exist (but don't execute full workflow)
    try:
        sys.path.insert(0, str(project_root / 'examples' / 'use-cases' / 'customer_service'))
        
        # Verify file is syntactically correct
        file_path = project_root / 'examples' / 'use-cases' / 'customer_service' / 'customer_service_agents.py'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            print("✓ Customer service agents - syntax OK")
        else:
            pytest.skip("Customer service example not found")
    except Exception as e:
        pytest.fail(f"Customer service agents failed: {e}")


@pytest.mark.asyncio
async def test_manager_assistant_structure():
    """Test manager assistant basic structure."""
    try:
        file_path = project_root / 'examples' / 'use-cases' / 'manager_assistant' / 'manager_assistant.py'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            print("✓ Manager assistant - syntax OK")
        else:
            pytest.skip("Manager assistant example not found")
    except Exception as e:
        pytest.fail(f"Manager assistant failed: {e}")


@pytest.mark.asyncio
async def test_it_operations_structure():
    """Test IT operations basic structure."""
    try:
        file_path = project_root / 'examples' / 'use-cases' / 'it_operations' / 'it_operations_automation.py'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            print("✓ IT operations - syntax OK")
        else:
            pytest.skip("IT operations example not found")
    except Exception as e:
        pytest.fail(f"IT operations failed: {e}")


@pytest.mark.asyncio
async def test_market_intelligence_structure():
    """Test market intelligence basic structure."""
    try:
        file_path = project_root / 'examples' / 'use-cases' / 'market_intelligence' / 'market_intelligence_agents.py'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            print("✓ Market intelligence - syntax OK")
        else:
            pytest.skip("Market intelligence example not found")
    except Exception as e:
        pytest.fail(f"Market intelligence failed: {e}")


@pytest.mark.asyncio
async def test_business_intelligence_structure():
    """Test business intelligence basic structure."""
    try:
        file_path = project_root / 'examples' / 'use-cases' / 'business_intelligence' / 'business_intelligence_agents.py'
        if file_path.exists():
            # Verify syntax
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            
            # Verify import
            module = import_module_from_path("business_intelligence_agents", file_path)
            assert module is not None
            assert hasattr(module, "BusinessIntelligenceOrchestrator")
            
            print("✓ Business intelligence - syntax and import OK")
        else:
            pytest.skip("Business intelligence example not found")
    except Exception as e:
        pytest.fail(f"Business intelligence failed: {e}")


def test_mcp_implementation_structure():
    """Test MCP implementation structure."""
    try:
        file_path = project_root / 'examples' / 'communication' / 'mcp_implementation.py'
        if file_path.exists():
            # Verify syntax
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            
            # Verify import
            module = import_module_from_path("mcp_implementation", file_path)
            assert module is not None
            assert hasattr(module, "MCPServer")
            assert hasattr(module, "MCPClient")
            
            print("✓ MCP implementation - syntax and import OK")
        else:
            pytest.skip("MCP implementation not found")
    except Exception as e:
        pytest.fail(f"MCP implementation failed: {e}")


def test_multi_model_pipeline_structure():
    """Test multi-model pipeline structure."""
    try:
        file_path = project_root / 'examples' / 'communication' / 'multi_model_pipeline.py'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, str(file_path), 'exec')
            print("✓ Multi-model pipeline - syntax OK")
        else:
            pytest.skip("Multi-model pipeline not found")
    except Exception as e:
        pytest.fail(f"Multi-model pipeline failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
