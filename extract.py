#!/usr/bin/env python3
import os
import subprocess
import argparse
import fnmatch

def load_extractignore_patterns(repo_path):
    """Load ignore patterns from .extractignore file if it exists."""
    extractignore_path = os.path.join(repo_path, '.extractignore')
    patterns = []
    
    if os.path.exists(extractignore_path):
        try:
            with open(extractignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.append(line)
        except Exception as e:
            print(f"Warning: Error reading .extractignore file: {e}")
    
    return patterns

def should_ignore_file(file_path, ignore_patterns):
    """Check if a file should be ignored based on patterns."""
    for pattern in ignore_patterns:
        # Normalize path separators for cross-platform compatibility
        normalized_path = file_path.replace('\\', '/')
        normalized_pattern = pattern.replace('\\', '/')
        
        # Support both simple glob patterns and path-based patterns
        if fnmatch.fnmatch(normalized_path, normalized_pattern) or fnmatch.fnmatch(os.path.basename(normalized_path), normalized_pattern):
            return True
        
        # Check if the file path starts with the pattern (for directory patterns)
        if normalized_path.startswith(normalized_pattern + '/'):
            return True
            
        # Also check if any part of the path matches (for directory patterns)
        path_parts = normalized_path.split('/')
        for part in path_parts:
            if fnmatch.fnmatch(part, normalized_pattern):
                return True
    return False

def list_git_files(repo_path):
    """List all files tracked by git in a repository."""
    os.chdir(repo_path)
    result = subprocess.run(['git', 'ls-files'], 
                           stdout=subprocess.PIPE, 
                           text=True, 
                           check=True)
    return result.stdout.splitlines()

def extract_git_contents(repo_path, output_file):
    """Extract all git-tracked files' names and contents to a text file."""
    files = list_git_files(repo_path)
    ignore_patterns = load_extractignore_patterns(repo_path)
    
    if ignore_patterns:
        print(f"Found .extractignore with {len(ignore_patterns)} patterns")
    
    filtered_files = []
    ignored_files = []
    
    for file_path in files:
        if should_ignore_file(file_path, ignore_patterns):
            ignored_files.append(file_path)
            print(f"Ignoring: {file_path}")
            continue
        filtered_files.append(file_path)
    
    if ignored_files:
        print(f"Ignored {len(ignored_files)} files based on .extractignore patterns")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Repository: {repo_path}\n")
        f.write(f"Total files processed: {len(filtered_files)}\n")
        if ignored_files:
            f.write(f"Files ignored by .extractignore: {len(ignored_files)}\n")
        f.write("="*80 + "\n\n")
        
        for file_path in filtered_files:
            try:
                # Skip binary files, large files, or files that can't be read as text
                if os.path.getsize(os.path.join(repo_path, file_path)) > 1024 * 1024:  # Skip files > 1MB
                    f.write(f"\nFilename: {file_path}\n")
                    f.write("Content: [File too large to display]\n\n")
                    continue
                    
                # Try to read the file
                with open(os.path.join(repo_path, file_path), 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                f.write(f"\nFilename: {file_path}\n")
                f.write("Content:\n")
                f.write(content)
                f.write("\n\n" + "="*80 + "\n")
            except UnicodeDecodeError:
                f.write(f"\nFilename: {file_path}\n")
                f.write("Content: [Binary file - cannot display content]\n\n")
            except Exception as e:
                f.write(f"\nFilename: {file_path}\n")
                f.write(f"Error reading file: {str(e)}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract all files from a git repository')
    parser.add_argument('repo_path', help='Path to the git repository')
    parser.add_argument('--output', '-o', default='repo_contents.txt', 
                        help='Output file (default: repo_contents.txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' does not exist")
        exit(1)
        
    if not os.path.exists(os.path.join(args.repo_path, '.git')):
        print(f"Error: '{args.repo_path}' is not a git repository")
        exit(1)
    
    print(f"Extracting files from {args.repo_path}...")
    extract_git_contents(args.repo_path, args.output)
    print(f"Done! Results saved to {args.output}")
