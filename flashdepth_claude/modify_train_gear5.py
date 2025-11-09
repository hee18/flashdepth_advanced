#!/usr/bin/env python3
"""
Helper script to modify train_gear5.py from train_gear3_upgrade.py
Automates all necessary changes for Gear5 two-stage training.
"""

import re


def modify_train_gear5():
    """Apply all necessary modifications to train_gear5.py"""

    with open('train_gear5.py', 'r') as f:
        content = f.read()

    # 1. Replace all 'self.phase' with 'self.step'
    content = re.sub(r'\bself\.phase\b', 'self.step', content)
    content = re.sub(r'\bphase\b(?=\s*[=:])', 'step', content)

    # 2. Replace 'gear3' project names with 'gear5'
    content = content.replace('flashdepth-gear3', 'flashdepth-gear5')
    content = content.replace('gear3_phase', 'gear5_step')
    content = content.replace('train_results/gear3', 'train_results/gear5')

    # 3. Update separation_method to always be 'multi_layer' (not used in Step 1, but needed for Step 2)
    # Keep the config reading for compatibility

    # 4. Replace checkpoint handling for Step 2
    content = re.sub(
        r"# Step 2: Load Gear-S checkpoint.*?checkpoint_path = self\.config\.get\('gear_checkpoint'\)",
        """# Step 2: Load Step 1 checkpoint
            checkpoint_path = self.config.get('step1_checkpoint')""",
        content,
        flags=re.DOTALL
    )

    # 5. Update logger messages about phase/step
    content = content.replace('Phase {self.step}', 'Step {self.step}')
    content = content.replace('Training Phase', 'Training Step')
    content = content.replace('phase {self.step}', 'step {self.step}')

    # 6. Update checkpoint name patterns
    content = content.replace('phase{self.step}', 'step{self.step}')
    content = content.replace('_phase{self.step}', '_step{self.step}')

    # 7. Update file paths and naming
    content = content.replace('gear3{phase_suffix}', 'gear5{step_suffix}')
    content = content.replace('phase_suffix', 'step_suffix')

    # 8. Save the modified content
    with open('train_gear5.py', 'w') as f:
        f.write(content)

    print("✅ Successfully modified train_gear5.py")
    print("Key changes:")
    print("  - Renamed phase → step throughout")
    print("  - Updated project names gear3 → gear5")
    print("  - Updated checkpoint handling for 2-stage training")
    print("  - Updated all logging messages")


if __name__ == '__main__':
    modify_train_gear5()
