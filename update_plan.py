import re

with open('AGENT_PLAN.md', 'r') as f:
    content = f.read()

# Phase 2: mark 5.1-5.3 as done and clear the PAUSE note
content = re.sub(
    r'^(\s*\[\s*\]\s+)5\.1\s+→\s+5\.3.*\n?',
    r'\1[x] 5.1 → 5.3 (smoke test passed — 2 epochs, ACNE04 only)\n',
    content,
    flags=re.MULTILINE
)

# Also clear the PAUSE line under Phase 2
content = re.sub(
    r'↳ PAUSE: Mohit runs smoke test.*?Confirm OK before continuing\n?',
    '↳ SMOKE TEST PASSED — proceeding to Phase 3\n',
    content,
    flags=re.DOTALL
)

with open('AGENT_PLAN.md', 'w') as f:
    f.write(content)
print('AGENT_PLAN.md updated')