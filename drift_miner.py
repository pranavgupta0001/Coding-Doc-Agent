#!/usr/bin/env python3
"""
Documentation Drift Miner

This tool mines GitHub repositories (SciPy, NumPy) for documentation drift events.
It identifies commits that fix documentation drift and extracts code-documentation pairs
labeled as "Consistent" (after fix) or "Drifted" (before fix).
"""

import os
import json
import re
import argparse
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from github import Github, GithubException
from dotenv import load_dotenv


class DriftMiner:
    """Mines documentation drift from GitHub repositories."""
    
    # Keywords that indicate drift-fixing commits
    DRIFT_KEYWORDS = [
        'update docs',
        'update documentation',
        'fix docs',
        'fix documentation',
        'fix formula',
        'fix docstring',
        'sync comment',
        'sync documentation',
        'correct docs',
        'correct documentation',
        'docs fix',
        'documentation fix',
        'update comment',
        'fix comment'
    ]
    
    # File extensions to analyze
    CODE_EXTENSIONS = ['.py', '.c', '.cpp', '.h', '.hpp', '.f', '.f90']
    DOC_EXTENSIONS = ['.rst', '.md', '.txt']
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize the drift miner with GitHub API access."""
        load_dotenv()
        token = github_token or os.getenv('GITHUB_TOKEN')
        if not token:
            print("Warning: No GitHub token provided. API rate limits will be restrictive.")
            self.github = Github()
        else:
            self.github = Github(token)
        
        self.drift_events = []
    
    def is_drift_fixing_commit(self, commit_message: str) -> bool:
        """Check if a commit message indicates a drift-fixing commit."""
        message_lower = commit_message.lower()
        return any(keyword in message_lower for keyword in self.DRIFT_KEYWORDS)
    
    def extract_code_segments(self, file_content: str, filename: str) -> List[Dict[str, str]]:
        """Extract code segments with their documentation from a file."""
        segments = []
        
        if not filename.endswith('.py'):
            # For now, focus on Python files which have clear docstrings
            return segments
        
        lines = file_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for function/class definitions
            if line.startswith('def ') or line.startswith('class '):
                start_line = i
                code_lines = [lines[i]]
                doc_lines = []
                
                # Move to next line to check for docstring
                i += 1
                if i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        quote = '"""' if next_line.startswith('"""') else "'''"
                        doc_lines.append(lines[i])
                        
                        # Check if it's a single-line docstring
                        if lines[i].count(quote) == 2:
                            # Single-line docstring (opening and closing on same line)
                            pass
                        else:
                            # Multi-line docstring
                            i += 1
                            # Collect docstring lines until closing quote
                            while i < len(lines) and quote not in lines[i]:
                                doc_lines.append(lines[i])
                                i += 1
                            
                            if i < len(lines):
                                doc_lines.append(lines[i])
                        
                        i += 1
                
                # Collect a few more lines of code for context
                context_lines = 0
                while i < len(lines) and context_lines < 10:
                    # Stop at next function/class definition
                    stripped = lines[i].strip()
                    if stripped.startswith('def ') or stripped.startswith('class '):
                        break
                    
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        code_lines.append(lines[i])
                        context_lines += 1
                    i += 1
                
                if doc_lines:
                    segments.append({
                        'filename': filename,
                        'start_line': start_line,
                        'code': '\n'.join(code_lines),
                        'documentation': '\n'.join(doc_lines)
                    })
                # Don't increment i here - it was already incremented in the loop
            else:
                i += 1
        
        return segments
    
    def mine_repository(self, repo_name: str, max_commits: int = 100) -> List[Dict]:
        """Mine a repository for drift-fixing commits."""
        print(f"Mining repository: {repo_name}")
        
        try:
            repo = self.github.get_repo(repo_name)
        except GithubException as e:
            print(f"Error accessing repository {repo_name}: {e}")
            print(f"Status: {e.status if hasattr(e, 'status') else 'Unknown'}")
            if hasattr(e, 'status') and e.status == 403:
                print("Note: This is likely due to API rate limiting. Please provide a GitHub token.")
                print("You can set GITHUB_TOKEN environment variable or use --token option.")
            return []
        except Exception as e:
            print(f"Unexpected error accessing repository {repo_name}: {e}")
            return []
        
        drift_events = []
        commits_checked = 0
        
        try:
            commits = repo.get_commits()
            
            for commit in commits:
                if commits_checked >= max_commits:
                    break
                
                commits_checked += 1
                
                if commits_checked % 10 == 0:
                    print(f"Checked {commits_checked} commits...")
                
                # Check if this is a drift-fixing commit
                if not self.is_drift_fixing_commit(commit.commit.message):
                    continue
                
                print(f"Found drift-fixing commit: {commit.sha[:7]} - {commit.commit.message[:80]}")
                
                # Get the files changed in this commit
                try:
                    files = commit.files
                    
                    for file in files:
                        # Only process code files
                        if not any(file.filename.endswith(ext) for ext in self.CODE_EXTENSIONS):
                            continue
                        
                        # Extract before and after content
                        try:
                            # Get file content after the fix (consistent)
                            try:
                                after_content = repo.get_contents(file.filename, ref=commit.sha).decoded_content.decode('utf-8')
                            except (GithubException, UnicodeDecodeError, AttributeError):
                                after_content = None
                            
                            # Get file content before the fix (drifted)
                            before_content = None
                            if commit.parents:
                                try:
                                    before_content = repo.get_contents(file.filename, ref=commit.parents[0].sha).decoded_content.decode('utf-8')
                                except (GithubException, UnicodeDecodeError, AttributeError):
                                    pass
                            
                            if after_content and before_content:
                                # Extract code-documentation segments
                                after_segments = self.extract_code_segments(after_content, file.filename)
                                before_segments = self.extract_code_segments(before_content, file.filename)
                                
                                # Create drift event
                                drift_event = {
                                    'repository': repo_name,
                                    'commit_sha': commit.sha,
                                    'commit_message': commit.commit.message,
                                    'commit_date': commit.commit.author.date.isoformat(),
                                    'author': commit.commit.author.name,
                                    'file': file.filename,
                                    'patch': file.patch if hasattr(file, 'patch') else None,
                                    'before_segments': before_segments,  # Drifted
                                    'after_segments': after_segments,    # Consistent
                                }
                                
                                drift_events.append(drift_event)
                        
                        except Exception as e:
                            print(f"  Error processing file {file.filename}: {e}")
                            continue
                
                except Exception as e:
                    print(f"  Error processing commit files: {e}")
                    continue
        
        except GithubException as e:
            print(f"Error iterating commits: {e}")
        
        print(f"Found {len(drift_events)} drift events in {commits_checked} commits")
        return drift_events
    
    def save_results(self, output_file: str):
        """Save drift events to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.drift_events, f, indent=2)
        print(f"Results saved to {output_file}")
    
    def generate_summary(self) -> Dict:
        """Generate a summary of the mining results."""
        summary = {
            'total_drift_events': len(self.drift_events),
            'repositories': {},
            'common_keywords': {},
        }
        
        for event in self.drift_events:
            repo = event['repository']
            if repo not in summary['repositories']:
                summary['repositories'][repo] = 0
            summary['repositories'][repo] += 1
            
            # Count keywords in commit messages
            message = event['commit_message'].lower()
            for keyword in self.DRIFT_KEYWORDS:
                if keyword in message:
                    if keyword not in summary['common_keywords']:
                        summary['common_keywords'][keyword] = 0
                    summary['common_keywords'][keyword] += 1
        
        return summary


def main():
    """Main entry point for the drift miner."""
    parser = argparse.ArgumentParser(
        description='Mine documentation drift from GitHub repositories'
    )
    parser.add_argument(
        '--repos',
        nargs='+',
        default=['scipy/scipy', 'numpy/numpy'],
        help='GitHub repositories to mine (format: owner/repo)'
    )
    parser.add_argument(
        '--max-commits',
        type=int,
        default=100,
        help='Maximum number of commits to check per repository'
    )
    parser.add_argument(
        '--output',
        default='drift_events.json',
        help='Output file for drift events'
    )
    parser.add_argument(
        '--token',
        help='GitHub API token (or set GITHUB_TOKEN environment variable)'
    )
    
    args = parser.parse_args()
    
    # Initialize miner
    miner = DriftMiner(github_token=args.token)
    
    # Mine each repository
    for repo in args.repos:
        events = miner.mine_repository(repo, max_commits=args.max_commits)
        miner.drift_events.extend(events)
    
    # Save results
    miner.save_results(args.output)
    
    # Generate and display summary
    summary = miner.generate_summary()
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total drift events found: {summary['total_drift_events']}")
    print(f"\nBy repository:")
    for repo, count in summary['repositories'].items():
        print(f"  {repo}: {count} events")
    print(f"\nCommon keywords:")
    for keyword, count in sorted(summary['common_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  '{keyword}': {count} occurrences")


if __name__ == '__main__':
    main()
