#!/usr/bin/env python3
"""
Helper script to modify test_gear5.py from test_gear3_upgrade.py
Automates all necessary changes for Gear5 two-stage testing.
"""

import re


def modify_test_gear5():
    """Apply all necessary modifications to test_gear5.py"""

    with open('test_gear5.py', 'r') as f:
        content = f.read()

    # 1. Update docstring
    old_docstring = '''"""
Gear3 Upgrade Test Script: Enhanced FG/BG Separation Methods'''

    new_docstring = '''"""
Gear5 Test Script: Two-Stage Global + Foreground Modulation'''

    content = content.replace(old_docstring, new_docstring)

    # 2. Update imports
    content = content.replace(
        'from flashdepth.gear3_upgrade_modules import Gear3UpgradeMetricHead',
        '''from flashdepth.gear5_modules import (
    GlobalScalePredictorMultiLayer,
    ForegroundOnlyModulationHead,
    Gear5MetricHead
)'''
    )

    # 3. Replace class name
    content = content.replace('class Gear3UpgradeTester:', 'class Gear5Tester:')

    # 4. Update results directory
    content = content.replace("'test_results/gear3'", "'test_results/gear5'")
    content = content.replace("test_results/gear3", "test_results/gear5")

    # 5. Update config path reference
    content = content.replace('configs/gear3', 'configs/gear5')

    # 6. Update all self.phase references
    content = re.sub(r'\bself\.phase\b', 'self.step', content)

    # 7. Update initialization to use 'step' instead of 'phase'
    content = re.sub(
        r"self\.phase = config\.get\('phase', 1\)",
        "self.step = config.get('step', 1)",
        content
    )

    # 8. Update logging messages
    content = content.replace('Phase {self.', 'Step {self.')
    content = content.replace('Testing Phase', 'Testing Step')
    content = content.replace('phase {self.', 'step {self.')

    # 9. Update checkpoint loading config
    content = re.sub(
        r"checkpoint_path = self\.config\.get\('gear_checkpoint'\)",
        "checkpoint_path = self.config.get('step_checkpoint')",
        content
    )

    # 10. Update Gear3UpgradeMetricHead references
    content = content.replace('Gear3UpgradeMetricHead', 'Gear5MetricHead')
    content = content.replace('gear3_upgrade_head', 'gear5_metric_head')

    # 11. Update separation_method handling
    #     Gear5 always uses multi_layer [4,11,17,23] for Step 2
    #     Step 1 doesn't need separation, just global GSP

    # 12. Save the modified content
    with open('test_gear5.py', 'w') as f:
        f.write(content)

    print("✅ Successfully modified test_gear5.py")
    print("Key changes:")
    print("  - Updated docstring and class name")
    print("  - Changed imports to gear5_modules")
    print("  - Renamed phase → step throughout")
    print("  - Updated Gear3UpgradeMetricHead → Gear5MetricHead")
    print("  - Updated results directories")
    print("")
    print("⚠️  Manual review needed for:")
    print("  - Model setup for Step 1 vs Step 2")
    print("  - Forward pass logic (global GSP vs GSP+FG modulation)")
    print("  - Visualization of step-specific outputs")


if __name__ == '__main__':
    modify_test_gear5()
