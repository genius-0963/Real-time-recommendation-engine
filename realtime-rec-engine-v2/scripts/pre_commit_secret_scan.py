#!/usr/bin/env python3
"""
Pre-commit hook to detect secrets in staged files.
Blocks commits containing potential secrets, API keys, passwords, etc.
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Pattern

# Patterns that indicate potential secrets
SECRET_PATTERNS: List[Tuple[Pattern, str]] = [
    # Generic patterns
    (re.compile(r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?key|auth[_-]?token)\s*[:=]\s*["\']?[a-zA-Z0-9_\-]{20,}["\']?'), "API Key/Secret"),
    (re.compile(r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?[^\s"\']{8,}["\']?'), "Password"),
    (re.compile(r'(?i)(jwt[_-]?secret|jwt[_-]?key)\s*[:=]\s*["\']?[^\s"\']{16,}["\']?'), "JWT Secret"),
    (re.compile(r'(?i)(private[_-]?key|ssh[_-]?key)\s*[:=]\s*["\']?[^\s"\']{20,}["\']?'), "Private Key"),
    
    # Specific formats
    (re.compile(r'[a-zA-Z0-9_\-]{24}\.[a-zA-Z0-9_\-]{6}\.[a-zA-Z0-9_\-]{27}'), "JWT Token"),
    (re.compile(r'sk-[a-zA-Z0-9]{48}'), "OpenAI API Key"),
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'), "GitHub Personal Access Token"),
    (re.compile(r'ghs_[a-zA-Z0-9]{36}'), "GitHub Secret"),
    (re.compile(r'gho_[a-zA-Z0-9]{36}'), "GitHub OAuth Token"),
    (re.compile(r'glpat-[a-zA-Z0-9_\-]{20}'), "GitLab Personal Access Token"),
    (re.compile(r'xox[baprs]-[a-zA-Z0-9]{10,}-[a-zA-Z0-9]{10,}-[a-zA-Z0-9]{10,}-[a-zA-Z0-9]{10,}'), "Slack Token"),
    (re.compile(r'AKIA[0-9A-Z]{16}'), "AWS Access Key ID"),
    (re.compile(r'[a-zA-Z0-9/+=]{40}'), "AWS Secret Access Key"),
    (re.compile(r'eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*'), "JWT"),
    (re.compile(r'-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----'), "Private Key"),
    (re.compile(r'postgresql://[^:]+:[^@]+@[^/]+/\w+'), "PostgreSQL Connection String"),
    (re.compile(r'mongodb://[^:]+:[^@]+@[^/]+/\w+'), "MongoDB Connection String"),
    (re.compile(r'redis://[^:]+:[^@]+@[^/]+'), "Redis Connection String"),
    (re.compile(r'amqp://[^:]+:[^@]+@[^/]+'), "RabbitMQ Connection String"),
    
    # Cloud provider keys
    (re.compile(r'AIza[0-9A-Za-z_\-]{35}'), "Google API Key"),
    (re.compile(r'ya29\.[0-9A-Za-z_\-]+'), "Google OAuth Token"),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), "UUID (potential secret)"),
]

# Files/patterns to exclude from scanning
EXCLUDE_PATTERNS = [
    r'\.template$',
    r'\.example$',
    r'\.sample$',
    r'security\.yaml\.template',
    r'\.gitignore',
    r'README\.md',
    r'CHANGELOG\.md',
    r'LICENSE',
    r'test_.*\.py$',
    r'.*_test\.py$',
    r'conftest\.py',
    r'pytest\.ini',
    r'requirements.*\.txt',
    r'pyproject\.toml',
    r'package\.json',
    r'package-lock\.json',
    r'yarn\.lock',
    r'Cargo\.lock',
    r'go\.sum',
    r'composer\.lock',
]

# Allowed patterns (false positive reduction)
ALLOWED_PATTERNS = [
    r'your-secret-key',
    r'your-password',
    r'your-api-key',
    r'your-token',
    r'example',
    r'placeholder',
    r'dummy',
    r'test',
    r'changeme',
    r'changethis',
    r'REPLACE_ME',
    r'<.*>',
    r'\[.*\]',
    r'\{\{.*\}\}',
]

def should_exclude_file(filepath: str) -> bool:
    """Check if file should be excluded from scanning."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, filepath, re.IGNORECASE):
            return True
    return False

def is_allowed_match(match: str) -> bool:
    """Check if a match is a known false positive."""
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, match, re.IGNORECASE):
            return True
    return False

def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for potential secrets."""
    findings = []
    
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return findings
    
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
            continue
        
        for pattern, description in SECRET_PATTERNS:
            matches = pattern.findall(line)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ''
                
                if not is_allowed_match(match):
                    findings.append((line_num, description, match[:100]))
    
    return findings

def get_staged_files() -> List[Path]:
    """Get list of staged files from git."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
            capture_output=True,
            text=True,
            check=True
        )
        files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                filepath = Path(line)
                if filepath.exists() and filepath.is_file():
                    files.append(filepath)
        return files
    except subprocess.CalledProcessError:
        return []

def main():
    """Main entry point."""
    staged_files = get_staged_files()
    
    if not staged_files:
        print("No staged files to scan.")
        return 0
    
    all_findings = []
    
    for filepath in staged_files:
        if should_exclude_file(str(filepath)):
            continue
        
        findings = scan_file(filepath)
        if findings:
            all_findings.append((filepath, findings))
    
    if all_findings:
        print("\n❌ COMMIT BLOCKED: Potential secrets detected!\n")
        print("The following files contain potential secrets:\n")
        
        for filepath, findings in all_findings:
            print(f"  📄 {filepath}")
            for line_num, description, match in findings:
                print(f"     Line {line_num}: {description} - {match}")
            print()
        
        print("Remediation:")
        print("  1. Remove secrets from code")
        print("  2. Use environment variables or secret managers (Vault, AWS Secrets Manager)")
        print("  3. Add file to .gitignore if it contains local config")
        print("  4. Use security.yaml.template as reference for config structure")
        print()
        print("To bypass (NOT RECOMMENDED): git commit --no-verify")
        return 1
    
    print("✅ No secrets detected in staged files.")
    return 0

if __name__ == "__main__":
    sys.exit(main())