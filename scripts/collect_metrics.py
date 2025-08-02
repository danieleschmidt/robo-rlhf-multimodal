#!/usr/bin/env python3
"""
Automated metrics collection script for robo-rlhf-multimodal project.

This script collects various metrics from different sources and outputs them
in a standardized format for monitoring and reporting.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class MetricsCollector:
    """Collects and aggregates project metrics from various sources."""

    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize the metrics collector."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from project metrics file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            return {}

    def _run_command(self, command: List[str]) -> Optional[str]:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(command)}: {e}")
            return None
        except FileNotFoundError:
            self.logger.warning(f"Command not found: {command[0]}")
            return None

    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        metrics = {}
        
        # Commit activity
        commits_last_week = self._run_command([
            'git', 'rev-list', '--count', '--since="1 week ago"', 'HEAD'
        ])
        if commits_last_week:
            metrics['commits_last_week'] = int(commits_last_week)
        
        # Contributors
        contributors_last_month = self._run_command([
            'git', 'shortlog', '-sn', '--since="1 month ago"', 'HEAD'
        ])
        if contributors_last_month:
            metrics['active_contributors'] = len(contributors_last_month.split('\n'))
        
        # Branch information
        current_branch = self._run_command(['git', 'branch', '--show-current'])
        if current_branch:
            metrics['current_branch'] = current_branch
        
        # Latest commit info
        latest_commit = self._run_command([
            'git', 'log', '-1', '--format=%H,%s,%an,%ad', '--date=iso'
        ])
        if latest_commit:
            parts = latest_commit.split(',', 3)
            metrics['latest_commit'] = {
                'hash': parts[0],
                'message': parts[1],
                'author': parts[2],
                'date': parts[3]
            }
        
        # Repository size
        repo_size = self._run_command(['du', '-sh', '.git'])
        if repo_size:
            metrics['repository_size'] = repo_size.split('\t')[0]
        
        return metrics

    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Line count
        python_lines = self._run_command([
            'find', '.', '-name', '*.py', '-exec', 'wc', '-l', '{}', '+',
            '|', 'tail', '-1'
        ])
        if python_lines:
            metrics['python_lines_of_code'] = python_lines.split()[-2]
        
        # Test coverage (if coverage report exists)
        coverage_file = Path('.coverage')
        if coverage_file.exists():
            coverage_output = self._run_command(['coverage', 'report', '--format=json'])
            if coverage_output:
                try:
                    coverage_data = json.loads(coverage_output)
                    metrics['test_coverage'] = {
                        'line_coverage': coverage_data.get('totals', {}).get('percent_covered', 0),
                        'branch_coverage': coverage_data.get('totals', {}).get('percent_covered_display', '0%'),
                        'missing_lines': coverage_data.get('totals', {}).get('missing_lines', 0)
                    }
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse coverage JSON")
        
        # Complexity analysis (if available)
        complexity_output = self._run_command(['radon', 'cc', '--average', '.'])
        if complexity_output:
            metrics['cyclomatic_complexity'] = complexity_output
        
        # File counts
        python_files = self._run_command(['find', '.', '-name', '*.py', '-type', 'f', '|', 'wc', '-l'])
        test_files = self._run_command(['find', '.', '-name', 'test_*.py', '-type', 'f', '|', 'wc', '-l'])
        
        if python_files:
            metrics['python_file_count'] = int(python_files)
        if test_files:
            metrics['test_file_count'] = int(test_files)
            if python_files:
                metrics['test_ratio'] = int(test_files) / int(python_files)
        
        return metrics

    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {}
        
        # Count dependencies from pyproject.toml
        pyproject_file = Path('pyproject.toml')
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    pyproject_data = tomli.load(f)
                
                dependencies = pyproject_data.get('project', {}).get('dependencies', [])
                optional_deps = pyproject_data.get('project', {}).get('optional-dependencies', {})
                
                metrics['dependencies'] = {
                    'runtime_dependencies': len(dependencies),
                    'optional_dependencies': sum(len(deps) for deps in optional_deps.values()),
                    'total_dependencies': len(dependencies) + sum(len(deps) for deps in optional_deps.values())
                }
            except ImportError:
                self.logger.warning("tomli not available for parsing pyproject.toml")
            except Exception as e:
                self.logger.error(f"Error parsing pyproject.toml: {e}")
        
        # Security vulnerabilities (if safety is available)
        safety_output = self._run_command(['safety', 'check', '--json'])
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                metrics['security_vulnerabilities'] = len(safety_data)
            except json.JSONDecodeError:
                pass
        
        # Outdated packages (if pip-list is available)
        outdated_output = self._run_command(['pip', 'list', '--outdated', '--format=json'])
        if outdated_output:
            try:
                outdated_data = json.loads(outdated_output)
                metrics['outdated_packages'] = len(outdated_data)
            except json.JSONDecodeError:
                pass
        
        return metrics

    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect test execution metrics."""
        metrics = {}
        
        # Test execution time (if pytest-benchmark results exist)
        benchmark_file = Path('.benchmarks/Linux-*/latest.json')
        benchmark_files = list(Path('.benchmarks').glob('*/latest.json')) if Path('.benchmarks').exists() else []
        
        if benchmark_files:
            try:
                with open(benchmark_files[0], 'r') as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get('benchmarks', [])
                if benchmarks:
                    metrics['performance_benchmarks'] = {
                        'total_benchmarks': len(benchmarks),
                        'average_execution_time': sum(b.get('stats', {}).get('mean', 0) for b in benchmarks) / len(benchmarks),
                        'slowest_benchmark': max(b.get('stats', {}).get('mean', 0) for b in benchmarks)
                    }
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Test result files (JUnit XML)
        test_results = list(Path('.').glob('**/test-results.xml'))
        if test_results:
            metrics['test_files_found'] = len(test_results)
        
        return metrics

    def collect_docker_metrics(self) -> Dict[str, Any]:
        """Collect Docker image metrics."""
        metrics = {}
        
        # Docker image sizes (if images exist)
        docker_images = self._run_command(['docker', 'images', '--format', 'json'])
        if docker_images:
            try:
                images = [json.loads(line) for line in docker_images.split('\n') if line.strip()]
                robo_images = [img for img in images if 'robo-rlhf' in img.get('Repository', '')]
                
                if robo_images:
                    metrics['docker_images'] = {
                        'total_images': len(robo_images),
                        'total_size': sum(self._parse_size(img.get('Size', '0B')) for img in robo_images),
                        'latest_image_size': robo_images[0].get('Size', 'Unknown') if robo_images else None
                    }
            except (json.JSONDecodeError, ValueError):
                pass
        
        return metrics

    def _parse_size(self, size_str: str) -> int:
        """Parse Docker image size string to bytes."""
        try:
            if size_str.endswith('GB'):
                return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
            elif size_str.endswith('MB'):
                return int(float(size_str[:-2]) * 1024 * 1024)
            elif size_str.endswith('KB'):
                return int(float(size_str[:-2]) * 1024)
            else:
                return int(size_str.rstrip('B'))
        except ValueError:
            return 0

    def collect_github_metrics(self, github_token: Optional[str] = None) -> Dict[str, Any]:
        """Collect GitHub API metrics."""
        if not github_token:
            self.logger.warning("No GitHub token provided, skipping GitHub metrics")
            return {}
        
        metrics = {}
        headers = {'Authorization': f'token {github_token}'}
        repo = self.config.get('project', {}).get('repository')
        
        if not repo:
            self.logger.warning("No repository configured")
            return {}
        
        try:
            # Repository information
            repo_response = requests.get(
                f'https://api.github.com/repos/{repo}',
                headers=headers,
                timeout=30
            )
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                metrics['github_repo'] = {
                    'stars': repo_data.get('stargazers_count', 0),
                    'forks': repo_data.get('forks_count', 0),
                    'open_issues': repo_data.get('open_issues_count', 0),
                    'size': repo_data.get('size', 0),
                    'language': repo_data.get('language'),
                    'created_at': repo_data.get('created_at'),
                    'updated_at': repo_data.get('updated_at')
                }
            
            # Pull requests
            prs_response = requests.get(
                f'https://api.github.com/repos/{repo}/pulls?state=all&per_page=100',
                headers=headers,
                timeout=30
            )
            if prs_response.status_code == 200:
                prs_data = prs_response.json()
                open_prs = [pr for pr in prs_data if pr['state'] == 'open']
                merged_prs = [pr for pr in prs_data if pr['merged_at'] is not None]
                
                metrics['github_prs'] = {
                    'total_prs': len(prs_data),
                    'open_prs': len(open_prs),
                    'merged_prs': len(merged_prs),
                    'recent_prs': len([pr for pr in prs_data 
                                    if datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00')) > 
                                    datetime.now().replace(tzinfo=None) - timedelta(days=30)])
                }
            
            # Workflow runs
            workflows_response = requests.get(
                f'https://api.github.com/repos/{repo}/actions/runs?per_page=50',
                headers=headers,
                timeout=30
            )
            if workflows_response.status_code == 200:
                workflows_data = workflows_response.json()
                runs = workflows_data.get('workflow_runs', [])
                
                successful_runs = [run for run in runs if run['conclusion'] == 'success']
                failed_runs = [run for run in runs if run['conclusion'] == 'failure']
                
                metrics['github_workflows'] = {
                    'total_runs': len(runs),
                    'success_rate': len(successful_runs) / len(runs) if runs else 0,
                    'failed_runs': len(failed_runs),
                    'recent_runs': len([run for run in runs 
                                      if datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')) > 
                                      datetime.now().replace(tzinfo=None) - timedelta(days=7)])
                }
                
        except requests.RequestException as e:
            self.logger.error(f"Error fetching GitHub metrics: {e}")
        
        return metrics

    def collect_all_metrics(self, github_token: Optional[str] = None) -> Dict[str, Any]:
        """Collect all available metrics."""
        self.logger.info("Starting metrics collection...")
        
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'project': self.config.get('project', {}),
            'git': self.collect_git_metrics(),
            'code_quality': self.collect_code_quality_metrics(),
            'dependencies': self.collect_dependency_metrics(),
            'tests': self.collect_test_metrics(),
            'docker': self.collect_docker_metrics(),
            'github': self.collect_github_metrics(github_token)
        }
        
        self.logger.info("Metrics collection completed")
        return all_metrics

    def generate_report(self, metrics: Dict[str, Any], format: str = 'json') -> str:
        """Generate a formatted report from metrics."""
        if format == 'json':
            return json.dumps(metrics, indent=2)
        
        elif format == 'markdown':
            report = f"# Metrics Report\n\n"
            report += f"**Generated:** {metrics['timestamp']}\n\n"
            
            # Git metrics
            git_metrics = metrics.get('git', {})
            if git_metrics:
                report += "## Git Repository\n\n"
                report += f"- **Commits (last week):** {git_metrics.get('commits_last_week', 'N/A')}\n"
                report += f"- **Active contributors:** {git_metrics.get('active_contributors', 'N/A')}\n"
                report += f"- **Current branch:** {git_metrics.get('current_branch', 'N/A')}\n"
                report += f"- **Repository size:** {git_metrics.get('repository_size', 'N/A')}\n\n"
            
            # Code quality metrics
            quality_metrics = metrics.get('code_quality', {})
            if quality_metrics:
                report += "## Code Quality\n\n"
                report += f"- **Python files:** {quality_metrics.get('python_file_count', 'N/A')}\n"
                report += f"- **Test files:** {quality_metrics.get('test_file_count', 'N/A')}\n"
                report += f"- **Test ratio:** {quality_metrics.get('test_ratio', 'N/A'):.2f}\n"
                
                coverage = quality_metrics.get('test_coverage', {})
                if coverage:
                    report += f"- **Test coverage:** {coverage.get('line_coverage', 'N/A')}%\n"
                report += "\n"
            
            # GitHub metrics
            github_metrics = metrics.get('github', {})
            if github_metrics:
                repo_metrics = github_metrics.get('github_repo', {})
                pr_metrics = github_metrics.get('github_prs', {})
                workflow_metrics = github_metrics.get('github_workflows', {})
                
                if repo_metrics:
                    report += "## GitHub Repository\n\n"
                    report += f"- **Stars:** {repo_metrics.get('stars', 'N/A')}\n"
                    report += f"- **Forks:** {repo_metrics.get('forks', 'N/A')}\n"
                    report += f"- **Open issues:** {repo_metrics.get('open_issues', 'N/A')}\n"
                    report += f"- **Primary language:** {repo_metrics.get('language', 'N/A')}\n\n"
                
                if pr_metrics:
                    report += "## Pull Requests\n\n"
                    report += f"- **Total PRs:** {pr_metrics.get('total_prs', 'N/A')}\n"
                    report += f"- **Open PRs:** {pr_metrics.get('open_prs', 'N/A')}\n"
                    report += f"- **Merged PRs:** {pr_metrics.get('merged_prs', 'N/A')}\n"
                    report += f"- **Recent PRs (30 days):** {pr_metrics.get('recent_prs', 'N/A')}\n\n"
                
                if workflow_metrics:
                    report += "## CI/CD Workflows\n\n"
                    report += f"- **Total runs:** {workflow_metrics.get('total_runs', 'N/A')}\n"
                    report += f"- **Success rate:** {workflow_metrics.get('success_rate', 0):.1%}\n"
                    report += f"- **Failed runs:** {workflow_metrics.get('failed_runs', 'N/A')}\n"
                    report += f"- **Recent runs (7 days):** {workflow_metrics.get('recent_runs', 'N/A')}\n\n"
            
            return report
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_metrics(self, metrics: Dict[str, Any], output_file: str):
        """Save metrics to a file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {output_path}")


def main():
    """Main entry point for the metrics collection script."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument(
        '--config', 
        default='.github/project-metrics.json',
        help='Path to metrics configuration file'
    )
    parser.add_argument(
        '--output', 
        help='Output file for metrics (default: stdout)'
    )
    parser.add_argument(
        '--format', 
        choices=['json', 'markdown'], 
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--github-token',
        help='GitHub token for API access'
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
        collector = MetricsCollector(args.config)
        metrics = collector.collect_all_metrics(args.github_token)
        report = collector.generate_report(metrics, args.format)
        
        if args.output:
            if args.format == 'json':
                collector.save_metrics(metrics, args.output)
            else:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"Report saved to {args.output}")
        else:
            print(report)
            
    except Exception as e:
        logging.error(f"Error collecting metrics: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()