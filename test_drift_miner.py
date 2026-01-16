#!/usr/bin/env python3
"""
Test script to validate drift_miner functionality without GitHub API.
"""

import json
from drift_miner import DriftMiner


def test_keyword_detection():
    """Test that drift-fixing commit detection works."""
    print("Testing drift-fixing commit keyword detection...")
    print("-" * 60)
    
    miner = DriftMiner()
    
    test_messages = [
        ("DOC: Update docs for linear algebra module", True),
        ("BUG: Fix segmentation fault", False),
        ("DOC: Fix formula in scipy.optimize", True),
        ("ENH: Add new feature", False),
        ("MAINT: sync comment with implementation", True),
        ("DOC: Correct documentation for numpy.array", True),
        ("TEST: Add more test cases", False),
        ("fix docstring for mean function", True),
        ("Update comment in variance calculation", True),
        ("Refactor code structure", False),
    ]
    
    passed = 0
    failed = 0
    
    for message, expected in test_messages:
        result = miner.is_drift_fixing_commit(message)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{message[:50]}...' -> {result} (expected: {expected})")
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_code_segment_extraction():
    """Test that code segment extraction works."""
    print("\nTesting code segment extraction...")
    print("-" * 60)
    
    miner = DriftMiner()
    
    # Sample Python code with docstrings
    sample_code = '''
def calculate_mean(values):
    """
    Calculate the arithmetic mean of a list of values.
    
    Parameters
    ----------
    values : list
        A list of numerical values
    
    Returns
    -------
    float
        The mean of the values
    """
    return sum(values) / len(values)

def calculate_variance(values):
    """Compute variance of values."""
    mean = calculate_mean(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

class Statistics:
    """A class for statistical calculations."""
    
    def __init__(self, data):
        """Initialize with data."""
        self.data = data
'''
    
    segments = miner.extract_code_segments(sample_code, 'stats.py')
    
    print(f"Extracted {len(segments)} code-documentation segments:")
    for i, segment in enumerate(segments, 1):
        print(f"\n  Segment {i}:")
        print(f"    File: {segment['filename']}")
        print(f"    Start line: {segment['start_line']}")
        print(f"    Code preview: {segment['code'].split(chr(10))[0][:60]}...")
        print(f"    Doc preview: {segment['documentation'].split(chr(10))[0][:60]}...")
    
    # We expect at least 3 segments (2 functions + 1 class)
    success = len(segments) >= 3
    print(f"\nResult: {'✓ PASS' if success else '✗ FAIL'} (found {len(segments)} segments, expected >= 3)")
    return success


def test_generate_summary():
    """Test summary generation."""
    print("\nTesting summary generation...")
    print("-" * 60)
    
    miner = DriftMiner()
    
    # Create mock drift events
    miner.drift_events = [
        {
            'repository': 'scipy/scipy',
            'commit_message': 'DOC: update docs for linear algebra',
            'file': 'scipy/linalg/basic.py',
        },
        {
            'repository': 'numpy/numpy',
            'commit_message': 'DOC: fix formula in numpy.mean',
            'file': 'numpy/core/fromnumeric.py',
        },
        {
            'repository': 'scipy/scipy',
            'commit_message': 'DOC: sync comment with code',
            'file': 'scipy/optimize/minimize.py',
        },
    ]
    
    summary = miner.generate_summary()
    
    print(f"Total drift events: {summary['total_drift_events']}")
    print(f"Repositories: {summary['repositories']}")
    print(f"Keywords: {summary['common_keywords']}")
    
    success = (
        summary['total_drift_events'] == 3 and
        len(summary['repositories']) == 2 and
        summary['repositories']['scipy/scipy'] == 2 and
        summary['repositories']['numpy/numpy'] == 1
    )
    
    print(f"\nResult: {'✓ PASS' if success else '✗ FAIL'}")
    return success


def main():
    """Run all tests."""
    print("="*60)
    print("Drift Miner Test Suite")
    print("="*60)
    print()
    
    tests = [
        test_keyword_detection,
        test_code_segment_extraction,
        test_generate_summary,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
