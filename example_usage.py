#!/usr/bin/env python3
"""
Example script demonstrating how to use the DriftMiner programmatically.
"""

import os
from drift_miner import DriftMiner


def main():
    """Example usage of the DriftMiner class."""
    
    # Initialize the miner with a GitHub token (optional but recommended)
    # Token can be passed directly or via GITHUB_TOKEN environment variable
    token = os.getenv('GITHUB_TOKEN')
    miner = DriftMiner(github_token=token)
    
    print("="*60)
    print("Documentation Drift Mining Example")
    print("="*60)
    print()
    
    # Example 1: Mine a small number of commits from NumPy
    print("Example 1: Mining NumPy repository (limited to 20 commits)...")
    print("-" * 60)
    numpy_events = miner.mine_repository('numpy/numpy', max_commits=20)
    print(f"Found {len(numpy_events)} drift events in NumPy\n")
    
    # Example 2: Mine SciPy
    print("Example 2: Mining SciPy repository (limited to 20 commits)...")
    print("-" * 60)
    scipy_events = miner.mine_repository('scipy/scipy', max_commits=20)
    print(f"Found {len(scipy_events)} drift events in SciPy\n")
    
    # Add all events to the miner
    miner.drift_events.extend(numpy_events)
    miner.drift_events.extend(scipy_events)
    
    # Save results
    output_file = 'example_drift_events.json'
    miner.save_results(output_file)
    
    # Generate summary
    summary = miner.generate_summary()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total drift events: {summary['total_drift_events']}")
    print(f"\nEvents by repository:")
    for repo, count in summary['repositories'].items():
        print(f"  - {repo}: {count} events")
    
    if summary['common_keywords']:
        print(f"\nMost common drift-fixing keywords:")
        sorted_keywords = sorted(
            summary['common_keywords'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for keyword, count in sorted_keywords:
            print(f"  - '{keyword}': {count} occurrences")
    
    print(f"\nResults saved to: {output_file}")
    print("\nExample complete!")


if __name__ == '__main__':
    main()
