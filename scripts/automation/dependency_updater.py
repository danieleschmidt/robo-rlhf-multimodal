#!/usr/bin/env python3
"""
Automated dependency management and update script.

This script checks for outdated dependencies, security vulnerabilities,
and can automatically create pull requests for updates.
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


class DependencyUpdater:
    """Manages dependency updates and security vulnerability tracking."""

    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize the dependency updater."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Security thresholds from config
        self.security_config = self.config.get('metrics', {}).get('operations', {}).get('security', {})

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from project metrics file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            return {}

    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd
            )
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stderr.strip()
        except FileNotFoundError:
            return False, f"Command not found: {command[0]}"

    def check_outdated_packages(self) -> Dict[str, Any]:
        """Check for outdated packages using pip."""
        self.logger.info("Checking for outdated packages...")
        
        success, output = self._run_command(['pip', 'list', '--outdated', '--format=json'])
        
        if not success:
            return {
                'status': 'error',
                'message': 'Could not check for outdated packages',
                'details': output
            }
        
        try:
            outdated_packages = json.loads(output) if output else []
            
            # Categorize updates by severity
            patch_updates = []
            minor_updates = []
            major_updates = []
            
            for package in outdated_packages:
                current = package.get('version', '0.0.0')
                latest = package.get('latest_version', '0.0.0')
                
                current_parts = current.split('.')
                latest_parts = latest.split('.')
                
                if len(current_parts) >= 3 and len(latest_parts) >= 3:
                    if current_parts[0] != latest_parts[0]:
                        major_updates.append(package)
                    elif current_parts[1] != latest_parts[1]:
                        minor_updates.append(package)
                    else:
                        patch_updates.append(package)
                else:
                    minor_updates.append(package)  # Default to minor if version parsing fails
            
            result = {
                'status': 'success',
                'total_outdated': len(outdated_packages),
                'patch_updates': len(patch_updates),
                'minor_updates': len(minor_updates),
                'major_updates': len(major_updates),
                'packages': {
                    'patch': patch_updates,
                    'minor': minor_updates,
                    'major': major_updates
                },
                'message': f"Found {len(outdated_packages)} outdated packages"
            }
            
            return result
            
        except json.JSONDecodeError:
            return {
                'status': 'error',
                'message': 'Could not parse outdated packages output',
                'details': output
            }

    def check_security_vulnerabilities(self) -> Dict[str, Any]:
        """Check for security vulnerabilities using safety and pip-audit."""
        self.logger.info("Checking for security vulnerabilities...")
        
        vulnerabilities = []
        
        # Check with safety
        success, safety_output = self._run_command(['safety', 'check', '--json'])
        if success and safety_output:
            try:
                safety_data = json.loads(safety_output)
                for vuln in safety_data:
                    vulnerabilities.append({
                        'package': vuln.get('package_name'),
                        'version': vuln.get('analyzed_version'),
                        'vulnerability_id': vuln.get('vulnerability_id'),
                        'advisory': vuln.get('advisory'),
                        'cve': vuln.get('cve'),
                        'source': 'safety'
                    })
            except json.JSONDecodeError:
                self.logger.warning("Could not parse safety output")
        
        # Check with pip-audit
        success, audit_output = self._run_command(['pip-audit', '--format=json'])
        if success and audit_output:
            try:
                audit_data = json.loads(audit_output)
                for vuln in audit_data:
                    vulnerabilities.append({
                        'package': vuln.get('package'),
                        'version': vuln.get('version'),
                        'vulnerability_id': vuln.get('id'),
                        'advisory': vuln.get('description'),
                        'cve': vuln.get('aliases', []),
                        'source': 'pip-audit'
                    })
            except json.JSONDecodeError:
                self.logger.warning("Could not parse pip-audit output")
        
        # Categorize by severity
        critical_vulns = []
        high_vulns = []
        medium_vulns = []
        low_vulns = []
        
        for vuln in vulnerabilities:
            # Simple severity classification based on keywords
            advisory = vuln.get('advisory', '').lower()
            if any(word in advisory for word in ['critical', 'remote code execution', 'rce']):
                critical_vulns.append(vuln)
            elif any(word in advisory for word in ['high', 'sql injection', 'xss']):
                high_vulns.append(vuln)
            elif any(word in advisory for word in ['medium', 'denial of service', 'dos']):
                medium_vulns.append(vuln)
            else:
                low_vulns.append(vuln)
        
        result = {
            'status': 'success',
            'total_vulnerabilities': len(vulnerabilities),
            'critical': len(critical_vulns),
            'high': len(high_vulns),
            'medium': len(medium_vulns),
            'low': len(low_vulns),
            'vulnerabilities': {
                'critical': critical_vulns,
                'high': high_vulns,
                'medium': medium_vulns,
                'low': low_vulns
            },
            'message': f"Found {len(vulnerabilities)} security vulnerabilities"
        }
        
        # Add alert if critical or high vulnerabilities found
        if critical_vulns or high_vulns:
            severity_target = self.security_config.get('vulnerability_remediation_target_days', 7)
            result['alert'] = f"Found {len(critical_vulns + high_vulns)} critical/high vulnerabilities - fix within {severity_target} days"
        
        return result

    def generate_requirements_update(self, update_type: str = 'minor') -> Dict[str, Any]:
        """Generate updated requirements file."""
        self.logger.info(f"Generating {update_type} requirements update...")
        
        # Check if pyproject.toml exists
        pyproject_file = Path('pyproject.toml')
        if not pyproject_file.exists():
            return {
                'status': 'error',
                'message': 'pyproject.toml not found'
            }
        
        # Create temporary directory for update process
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy pyproject.toml to temp directory
            temp_pyproject = temp_path / 'pyproject.toml'
            subprocess.run(['cp', str(pyproject_file), str(temp_pyproject)], check=True)
            
            # Generate current and updated requirements
            current_req = temp_path / 'current-requirements.txt'
            updated_req = temp_path / 'updated-requirements.txt'
            
            # Generate current requirements
            success, _ = self._run_command([
                'pip-compile', str(temp_pyproject), '--output-file', str(current_req)
            ], cwd=temp_dir)
            
            if not success:
                return {
                    'status': 'error',
                    'message': 'Could not generate current requirements'
                }
            
            # Generate updated requirements based on update type
            upgrade_args = ['pip-compile', str(temp_pyproject), '--output-file', str(updated_req)]
            
            if update_type == 'patch':
                # Only patch updates (would need more sophisticated logic)
                upgrade_args.append('--upgrade')
            elif update_type == 'minor':
                upgrade_args.append('--upgrade')
            elif update_type == 'major':
                upgrade_args.extend(['--upgrade', '--resolver=backtracking'])
            elif update_type == 'security':
                # For security updates, we'd need to identify vulnerable packages
                upgrade_args.append('--upgrade')
            
            success, _ = self._run_command(upgrade_args, cwd=temp_dir)
            
            if not success:
                return {
                    'status': 'error',
                    'message': 'Could not generate updated requirements'
                }
            
            # Compare requirements files
            try:
                with open(current_req, 'r') as f:
                    current_content = f.read()
                with open(updated_req, 'r') as f:
                    updated_content = f.read()
                
                if current_content == updated_content:
                    return {
                        'status': 'success',
                        'has_updates': False,
                        'message': 'No updates available'
                    }
                
                # Parse differences
                current_lines = set(current_content.strip().split('\n'))
                updated_lines = set(updated_content.strip().split('\n'))
                
                added_packages = updated_lines - current_lines
                removed_packages = current_lines - updated_lines
                
                changes = []
                for package in added_packages:
                    if package and not package.startswith('#'):
                        changes.append({'type': 'added', 'package': package})
                for package in removed_packages:
                    if package and not package.startswith('#'):
                        changes.append({'type': 'removed', 'package': package})
                
                return {
                    'status': 'success',
                    'has_updates': True,
                    'changes': changes,
                    'current_requirements': current_content,
                    'updated_requirements': updated_content,
                    'message': f"Found {len(changes)} package changes"
                }
                
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error comparing requirements: {e}'
                }

    def create_update_branch(self, branch_name: str, requirements_content: str) -> Dict[str, Any]:
        """Create a new branch with updated requirements."""
        self.logger.info(f"Creating update branch: {branch_name}")
        
        try:
            # Create and checkout new branch
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            
            # Update requirements file
            requirements_file = Path('requirements.txt')
            with open(requirements_file, 'w') as f:
                f.write(requirements_content)
            
            # Add and commit changes
            subprocess.run(['git', 'add', 'requirements.txt'], check=True)
            subprocess.run(['git', 'commit', '-m', f'deps: automated dependency update ({branch_name})'], check=True)
            
            # Push branch
            subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)
            
            return {
                'status': 'success',
                'branch_name': branch_name,
                'message': f'Successfully created and pushed branch {branch_name}'
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'status': 'error',
                'message': f'Failed to create update branch: {e}'
            }

    def create_github_pr(self, branch_name: str, title: str, body: str, github_token: str) -> Dict[str, Any]:
        """Create a GitHub pull request for dependency updates."""
        if not github_token:
            return {
                'status': 'error',
                'message': 'GitHub token required to create PR'
            }
        
        repo = self.config.get('project', {}).get('repository')
        if not repo:
            return {
                'status': 'error',
                'message': 'Repository not configured'
            }
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        payload = {
            'title': title,
            'body': body,
            'head': branch_name,
            'base': 'main'
        }
        
        try:
            response = requests.post(
                f'https://api.github.com/repos/{repo}/pulls',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                pr_data = response.json()
                return {
                    'status': 'success',
                    'pr_number': pr_data['number'],
                    'pr_url': pr_data['html_url'],
                    'message': f'Successfully created PR #{pr_data["number"]}'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Failed to create PR: {response.status_code} - {response.text}'
                }
                
        except requests.RequestException as e:
            return {
                'status': 'error',
                'message': f'Error creating PR: {e}'
            }

    def run_dependency_check(self) -> Dict[str, Any]:
        """Run comprehensive dependency analysis."""
        self.logger.info("Running comprehensive dependency check...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'outdated_packages': self.check_outdated_packages(),
            'security_vulnerabilities': self.check_security_vulnerabilities()
        }
        
        # Determine overall status
        security_status = results['security_vulnerabilities'].get('status')
        outdated_status = results['outdated_packages'].get('status')
        
        critical_vulns = results['security_vulnerabilities'].get('critical', 0)
        high_vulns = results['security_vulnerabilities'].get('high', 0)
        total_outdated = results['outdated_packages'].get('total_outdated', 0)
        
        overall_status = 'good'
        recommendations = []
        
        if critical_vulns > 0:
            overall_status = 'critical'
            recommendations.append(f"Immediately update {critical_vulns} packages with critical vulnerabilities")
        elif high_vulns > 0:
            overall_status = 'high'
            recommendations.append(f"Update {high_vulns} packages with high severity vulnerabilities")
        elif total_outdated > 20:
            overall_status = 'medium'
            recommendations.append(f"Consider updating {total_outdated} outdated packages")
        elif total_outdated > 0:
            overall_status = 'low'
            recommendations.append(f"Monitor {total_outdated} outdated packages for updates")
        
        results['summary'] = {
            'overall_status': overall_status,
            'total_vulnerabilities': critical_vulns + high_vulns + results['security_vulnerabilities'].get('medium', 0) + results['security_vulnerabilities'].get('low', 0),
            'critical_issues': critical_vulns + high_vulns,
            'total_outdated': total_outdated,
            'recommendations': recommendations
        }
        
        return results

    def generate_report(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Generate a formatted report from dependency check results."""
        if format == 'json':
            return json.dumps(results, indent=2)
        
        elif format == 'markdown':
            report = "# Dependency Report\n\n"
            report += f"**Generated:** {results['timestamp']}\n"
            report += f"**Overall Status:** {results['summary']['overall_status'].upper()}\n\n"
            
            # Security vulnerabilities
            security = results['security_vulnerabilities']
            if security.get('total_vulnerabilities', 0) > 0:
                report += "## 🔒 Security Vulnerabilities\n\n"
                report += f"- **Critical:** {security.get('critical', 0)}\n"
                report += f"- **High:** {security.get('high', 0)}\n"
                report += f"- **Medium:** {security.get('medium', 0)}\n"
                report += f"- **Low:** {security.get('low', 0)}\n\n"
                
                # List critical and high vulnerabilities
                critical_vulns = security.get('vulnerabilities', {}).get('critical', [])
                high_vulns = security.get('vulnerabilities', {}).get('high', [])
                
                if critical_vulns or high_vulns:
                    report += "### High Priority Vulnerabilities\n\n"
                    for vuln in critical_vulns + high_vulns:
                        report += f"- **{vuln.get('package')}** ({vuln.get('version')}): {vuln.get('advisory', 'No description')}\n"
                    report += "\n"
            
            # Outdated packages
            outdated = results['outdated_packages']
            if outdated.get('total_outdated', 0) > 0:
                report += "## 📦 Outdated Packages\n\n"
                report += f"- **Total Outdated:** {outdated.get('total_outdated', 0)}\n"
                report += f"- **Major Updates:** {outdated.get('major_updates', 0)}\n"
                report += f"- **Minor Updates:** {outdated.get('minor_updates', 0)}\n"
                report += f"- **Patch Updates:** {outdated.get('patch_updates', 0)}\n\n"
            
            # Recommendations
            recommendations = results['summary'].get('recommendations', [])
            if recommendations:
                report += "## 💡 Recommendations\n\n"
                for rec in recommendations:
                    report += f"- {rec}\n"
                report += "\n"
            
            return report
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """Main entry point for the dependency updater."""
    parser = argparse.ArgumentParser(description="Automated dependency management")
    parser.add_argument(
        '--config',
        default='.github/project-metrics.json',
        help='Path to metrics configuration file'
    )
    parser.add_argument(
        '--action',
        choices=['check', 'update', 'create-pr'],
        default='check',
        help='Action to perform'
    )
    parser.add_argument(
        '--update-type',
        choices=['patch', 'minor', 'major', 'security'],
        default='minor',
        help='Type of updates to apply'
    )
    parser.add_argument(
        '--output',
        help='Output file for results (default: stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--github-token',
        help='GitHub token for creating PRs'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        updater = DependencyUpdater(args.config)
        
        if args.action == 'check':
            results = updater.run_dependency_check()
            report = updater.generate_report(results, args.format)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"Report saved to {args.output}")
            else:
                print(report)
        
        elif args.action == 'update':
            update_result = updater.generate_requirements_update(args.update_type)
            print(json.dumps(update_result, indent=2))
        
        elif args.action == 'create-pr':
            if not args.github_token:
                print("GitHub token required for creating PRs")
                sys.exit(1)
            
            # This would be a more complex workflow involving:
            # 1. Running dependency check
            # 2. Generating updates if needed
            # 3. Creating branch and PR
            print("PR creation workflow not fully implemented in this example")
            
    except Exception as e:
        logging.error(f"Error in dependency updater: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()