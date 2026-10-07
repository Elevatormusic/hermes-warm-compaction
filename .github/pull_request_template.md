## What and why

<!-- What this change does, and the problem it fixes. Link the issue. -->

## How it was tested

<!-- The new test that failed before the change, and the checks you ran. -->

- [ ] `python -m unittest discover -s tests -p "test_wc_*.py"` passes
- [ ] `ruff check .` passes
- [ ] The integration check passes on a clean Hermes checkout (when the change touches Hermes behavior)
- [ ] README Limits and CHANGELOG are updated (when behavior changes)
- [ ] No private conversation, key, or full request body in the change
