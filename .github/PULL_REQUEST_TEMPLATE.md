# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code quality improvement (refactoring, optimization)
- [ ] CI/CD improvement
- [ ] Test coverage improvement

## Related Issues
<!-- Link to related issues using #issue_number -->
Closes #
Related to #

## Changes Made
<!-- List the specific changes made in this PR -->

- 
- 
- 

## Motivation and Context
<!-- Why is this change required? What problem does it solve? -->

## How Has This Been Tested?
<!-- Describe the tests you ran to verify your changes -->

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing
- [ ] Test environment: [e.g., Python 3.11, Ubuntu 22.04]

**Test Configuration:**
- Python version:
- OS:
- Models tested:

**Test Output:**
```
Paste relevant test output here
```

## Screenshots (if applicable)
<!-- Add screenshots to demonstrate the changes -->

## Code Quality Checklist
<!-- Ensure your code meets quality standards -->

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Pre-commit Checks
<!-- These should pass before submitting -->

- [ ] `black` formatting passes
- [ ] `isort` import sorting passes
- [ ] `flake8` linting passes
- [ ] `mypy` type checking passes
- [ ] `pytest` tests pass
- [ ] `bandit` security scan passes

**Run locally:**
```bash
# Format and check
black examples/ tests/
isort examples/ tests/
flake8 examples/ tests/

# Type check
mypy examples/

# Run tests
pytest tests/ -v

# Security scan
bandit -r examples/
```

## Documentation
<!-- Have you updated the documentation? -->

- [ ] README.md updated (if needed)
- [ ] CHANGELOG.md updated
- [ ] Inline code documentation added/updated
- [ ] Example code added/updated (if applicable)
- [ ] API documentation updated (if applicable)

## Breaking Changes
<!-- If this PR contains breaking changes, describe them here -->

**Does this PR introduce breaking changes?**
- [ ] Yes
- [ ] No

**If yes, what breaks and what is the migration path?**

## Performance Impact
<!-- Describe any performance implications -->

- [ ] No performance impact
- [ ] Performance improvement (describe)
- [ ] Performance regression (explain why it's acceptable)

**Benchmarks (if applicable):**
```
Paste benchmark results here
```

## Dependencies
<!-- List any new dependencies added -->

- [ ] No new dependencies
- [ ] New dependencies added (list below)

**New dependencies:**
```
package-name==version  # reason for adding
```

## Deployment Notes
<!-- Any special deployment considerations? -->

- [ ] No special deployment steps required
- [ ] Requires configuration changes
- [ ] Requires database migration
- [ ] Requires environment variable updates

**Special instructions:**

## Additional Notes
<!-- Any additional information for reviewers -->

## Reviewer Checklist
<!-- For reviewers -->

- [ ] Code quality is acceptable
- [ ] Tests are adequate and passing
- [ ] Documentation is clear and complete
- [ ] No security concerns
- [ ] No performance concerns
- [ ] Breaking changes are justified and documented
