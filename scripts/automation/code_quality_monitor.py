#!/usr/bin/env python3
"""
Code quality monitoring and automated reporting script.

This script monitors code quality metrics and generates alerts when
thresholds are exceeded or quality trends are declining.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


class CodeQualityMonitor:
    """Monitors code quality metrics and generates reports."""

    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize the code quality monitor."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Quality thresholds from config
        self.thresholds = self.config.get('metrics', {}).get('development', {}).get('code_quality', {})

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
            self.logger.error(f"Command failed: {' '.join(command)}: {e}")
            return False, e.stderr.strip()
        except FileNotFoundError:
            self.logger.warning(f"Command not found: {command[0]}")
            return False, f"Command not found: {command[0]}"

    def check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage and compare against thresholds."""
        self.logger.info("Checking test coverage...")
        
        # Run coverage report
        success, output = self._run_command(['coverage', 'report', '--format=json'])
        
        if not success:
            return {
                'status': 'error',
                'message': 'Could not generate coverage report',
                'details': output
            }
        
        try:
            coverage_data = json.loads(output)
            coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
            missing_lines = coverage_data.get('totals', {}).get('missing_lines', 0)
            
            threshold = self.thresholds.get('coverage_threshold', 80)
            
            result = {
                'status': 'pass' if coverage_percent >= threshold else 'fail',
                'coverage_percent': coverage_percent,
                'missing_lines': missing_lines,
                'threshold': threshold,
                'message': f"Coverage is {coverage_percent:.1f}% (threshold: {threshold}%)"
            }
            
            if coverage_percent < threshold:
                result['alert'] = f"Code coverage ({coverage_percent:.1f}%) is below threshold ({threshold}%)"
            
            return result
            
        except json.JSONDecodeError:
            return {
                'status': 'error',
                'message': 'Could not parse coverage report',
                'details': output
            }

    def check_code_complexity(self) -> Dict[str, Any]:
        """Check code complexity using radon."""
        self.logger.info("Checking code complexity...")
        
        # Run radon complexity check
        success, output = self._run_command(['radon', 'cc', '--json', '.'])
        
        if not success:
            return {
                'status': 'warning',
                'message': 'Could not run complexity analysis',
                'details': output
            }
        
        try:
            complexity_data = json.loads(output)
            threshold = self.thresholds.get('complexity_threshold', 10)
            
            high_complexity_functions = []
            total_functions = 0
            
            for file_path, functions in complexity_data.items():
                for func in functions:
                    total_functions += 1
                    if func.get('complexity', 0) > threshold:
                        high_complexity_functions.append({
                            'file': file_path,
                            'function': func.get('name'),
                            'complexity': func.get('complexity'),
                            'line': func.get('lineno')
                        })
            
            result = {
                'status': 'pass' if not high_complexity_functions else 'fail',
                'total_functions': total_functions,
                'high_complexity_count': len(high_complexity_functions),
                'threshold': threshold,
                'high_complexity_functions': high_complexity_functions[:10],  # Limit to top 10
                'message': f"Found {len(high_complexity_functions)} functions above complexity threshold"
            }
            
            if high_complexity_functions:
                result['alert'] = f"Found {len(high_complexity_functions)} functions with complexity > {threshold}"
            
            return result
            
        except json.JSONDecodeError:
            return {
                'status': 'error',
                'message': 'Could not parse complexity report',
                'details': output
            }

    def check_code_duplication(self) -> Dict[str, Any]:
        """Check for code duplication."""
        self.logger.info("Checking code duplication...")
        
        # Run code duplication check using pylint or similar
        success, output = self._run_command(['pylint', '--disable=all', '--enable=duplicate-code', '.'])
        
        if not success and 'duplicate-code' not in output:
            return {
                'status': 'warning',
                'message': 'Could not run duplication analysis',
                'details': output
            }
        
        # Count duplicate code blocks
        duplicate_blocks = output.count('Similar lines in')
        threshold = self.thresholds.get('duplication_threshold', 5)
        
        result = {
            'status': 'pass' if duplicate_blocks <= threshold else 'fail',
            'duplicate_blocks': duplicate_blocks,
            'threshold': threshold,
            'message': f"Found {duplicate_blocks} duplicate code blocks (threshold: {threshold})"
        }
        
        if duplicate_blocks > threshold:
            result['alert'] = f"Code duplication ({duplicate_blocks} blocks) exceeds threshold ({threshold})"
        
        return result

    def check_maintainability_index(self) -> Dict[str, Any]:
        """Check maintainability index using radon."""
        self.logger.info("Checking maintainability index...")
        
        # Run radon maintainability index
        success, output = self._run_command(['radon', 'mi', '--json', '.'])
        
        if not success:
            return {
                'status': 'warning',
                'message': 'Could not run maintainability analysis',
                'details': output
            }
        
        try:
            mi_data = json.loads(output)
            threshold = self.thresholds.get('maintainability_index_threshold', 20)
            
            low_maintainability_files = []
            total_files = 0
            average_mi = 0
            
            for file_path, mi_score in mi_data.items():
                total_files += 1
                average_mi += mi_score
                
                if mi_score < threshold:
                    low_maintainability_files.append({
                        'file': file_path,
                        'maintainability_index': mi_score
                    })
            
            if total_files > 0:
                average_mi = average_mi / total_files
            
            result = {
                'status': 'pass' if not low_maintainability_files else 'fail',
                'average_maintainability_index': average_mi,
                'total_files': total_files,
                'low_maintainability_count': len(low_maintainability_files),
                'threshold': threshold,
                'low_maintainability_files': low_maintainability_files[:10],
                'message': f"Average maintainability index: {average_mi:.1f}"
            }
            
            if low_maintainability_files:
                result['alert'] = f"Found {len(low_maintainability_files)} files with low maintainability"
            
            return result
            
        except json.JSONDecodeError:
            return {
                'status': 'error',
                'message': 'Could not parse maintainability report',
                'details': output
            }

    def check_security_issues(self) -> Dict[str, Any]:
        """Check for security issues using bandit."""
        self.logger.info("Checking security issues...")
        
        # Run bandit security analysis
        success, output = self._run_command(['bandit', '-r', '.', '-f', 'json'])
        
        if not success:
            return {
                'status': 'warning',
                'message': 'Could not run security analysis',
                'details': output
            }
        
        try:
            security_data = json.loads(output)
            results = security_data.get('results', [])
            
            high_severity = [r for r in results if r.get('issue_severity') == 'HIGH']
            medium_severity = [r for r in results if r.get('issue_severity') == 'MEDIUM']
            low_severity = [r for r in results if r.get('issue_severity') == 'LOW']
            
            result = {
                'status': 'pass' if not high_severity else 'fail',
                'total_issues': len(results),
                'high_severity': len(high_severity),
                'medium_severity': len(medium_severity),
                'low_severity': len(low_severity),
                'issues': results[:10],  # Limit to top 10
                'message': f"Found {len(results)} security issues"
            }
            
            if high_severity:
                result['alert'] = f"Found {len(high_severity)} high severity security issues"
            
            return result
            
        except json.JSONDecodeError:
            return {
                'status': 'error',
                'message': 'Could not parse security report',
                'details': output
            }

    def check_linting_issues(self) -> Dict[str, Any]:
        """Check for linting issues using flake8."""
        self.logger.info("Checking linting issues...")
        
        # Run flake8 linting
        success, output = self._run_command(['flake8', '--format=json', '.'])
        
        if success and not output:
            return {
                'status': 'pass',
                'message': 'No linting issues found',
                'total_issues': 0
            }
        
        # Parse flake8 output (even if command "failed" due to issues)
        lines = output.split('\n')
        issues = []
        
        for line in lines:
            if ':' in line and len(line.split(':')) >= 4:
                parts = line.split(':')
                if len(parts) >= 4:
                    issues.append({
                        'file': parts[0],
                        'line': parts[1],
                        'column': parts[2],
                        'message': ':'.join(parts[3:]).strip()
                    })
        
        result = {
            'status': 'pass' if len(issues) == 0 else 'warning',
            'total_issues': len(issues),
            'issues': issues[:20],  # Limit to top 20
            'message': f"Found {len(issues)} linting issues"
        }
        
        return result

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all code quality checks."""
        self.logger.info("Running comprehensive code quality checks...")
        
        checks = {
            'coverage': self.check_test_coverage(),
            'complexity': self.check_code_complexity(),
            'duplication': self.check_code_duplication(),
            'maintainability': self.check_maintainability_index(),
            'security': self.check_security_issues(),
            'linting': self.check_linting_issues()
        }
        
        # Overall status
        failed_checks = [name for name, result in checks.items() if result.get('status') == 'fail']
        warning_checks = [name for name, result in checks.items() if result.get('status') == 'warning']
        
        overall_status = 'pass'
        if failed_checks:
            overall_status = 'fail'
        elif warning_checks:
            overall_status = 'warning'
        
        # Collect all alerts
        alerts = []
        for check_name, result in checks.items():
            if 'alert' in result:
                alerts.append(f"{check_name.title()}: {result['alert']}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'failed_checks': failed_checks,
            'warning_checks': warning_checks,
            'alerts': alerts,
            'checks': checks
        }
        
        return summary

    def generate_report(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Generate a formatted report from quality check results."""
        if format == 'json':
            return json.dumps(results, indent=2)
        
        elif format == 'markdown':
            report = "# Code Quality Report\n\n"
            report += f"**Generated:** {results['timestamp']}\n"
            report += f"**Overall Status:** {results['overall_status'].upper()}\n\n"
            
            if results['alerts']:
                report += "## 🚨 Alerts\n\n"
                for alert in results['alerts']:
                    report += f"- ⚠️ {alert}\n"
                report += "\n"
            
            report += "## Check Results\n\n"
            
            for check_name, result in results['checks'].items():
                status_emoji = {
                    'pass': '✅',
                    'fail': '❌', 
                    'warning': '⚠️',
                    'error': '🔥'
                }.get(result.get('status'), '❓')
                
                report += f"### {status_emoji} {check_name.title()}\n\n"
                report += f"**Status:** {result.get('status', 'unknown').upper()}\n"
                report += f"**Message:** {result.get('message', 'No message')}\n"
                
                # Add specific details for each check
                if check_name == 'coverage':
                    if 'coverage_percent' in result:
                        report += f"**Coverage:** {result['coverage_percent']:.1f}%\n"
                    if 'threshold' in result:
                        report += f"**Threshold:** {result['threshold']}%\n"
                
                elif check_name == 'complexity':
                    if 'high_complexity_count' in result:
                        report += f"**High Complexity Functions:** {result['high_complexity_count']}\n"
                    if 'threshold' in result:
                        report += f"**Threshold:** {result['threshold']}\n"
                
                elif check_name == 'security':
                    if 'high_severity' in result:
                        report += f"**High Severity Issues:** {result['high_severity']}\n"
                    if 'total_issues' in result:
                        report += f"**Total Issues:** {result['total_issues']}\n"
                
                elif check_name == 'linting':
                    if 'total_issues' in result:
                        report += f"**Total Issues:** {result['total_issues']}\n"
                
                report += "\n"
            
            return report
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    def send_notification(self, results: Dict[str, Any], webhook_url: str):
        """Send notification to Slack or other webhook."""
        if not webhook_url:
            self.logger.warning("No webhook URL provided for notifications")
            return
        
        status_color = {
            'pass': 'good',
            'warning': 'warning', 
            'fail': 'danger'
        }.get(results['overall_status'], 'warning')
        
        payload = {
            "text": f"Code Quality Report - {results['overall_status'].upper()}",
            "attachments": [{
                "color": status_color,
                "fields": [
                    {
                        "title": "Overall Status",
                        "value": results['overall_status'].upper(),
                        "short": True
                    },
                    {
                        "title": "Failed Checks",
                        "value": str(len(results['failed_checks'])),
                        "short": True
                    },
                    {
                        "title": "Alerts",
                        "value": str(len(results['alerts'])),
                        "short": True
                    }
                ]
            }]
        }
        
        if results['alerts']:
            payload["attachments"][0]["fields"].append({
                "title": "Alert Details",
                "value": "\n".join(results['alerts'][:5]),  # Limit to 5 alerts
                "short": False
            })
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=30)
            if response.status_code == 200:
                self.logger.info("Notification sent successfully")
            else:
                self.logger.error(f"Failed to send notification: {response.status_code}")
        except requests.RequestException as e:
            self.logger.error(f"Error sending notification: {e}")


def main():
    """Main entry point for the code quality monitor."""
    parser = argparse.ArgumentParser(description="Monitor code quality metrics")
    parser.add_argument(
        '--config',
        default='.github/project-metrics.json',
        help='Path to metrics configuration file'
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
        '--webhook-url',
        help='Webhook URL for notifications'
    )
    parser.add_argument(
        '--fail-on-issues',
        action='store_true',
        help='Exit with non-zero code if issues are found'
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
        monitor = CodeQualityMonitor(args.config)
        results = monitor.run_all_checks()
        report = monitor.generate_report(results, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
        
        # Send notification if webhook URL provided
        if args.webhook_url:
            monitor.send_notification(results, args.webhook_url)
        
        # Exit with appropriate code
        if args.fail_on_issues and results['overall_status'] in ['fail', 'warning']:
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error running code quality checks: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()