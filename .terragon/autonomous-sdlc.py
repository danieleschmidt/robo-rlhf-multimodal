#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Engine
Continuous value discovery and execution system for repository improvement.
"""

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import yaml


class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization engine."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        self.load_configuration()
    
    def load_configuration(self):
        """Load Terragon configuration."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
    
    def discover_value_opportunities(self) -> List[Dict[str, Any]]:
        """Discover and score potential value items."""
        opportunities = []
        
        # Git history analysis
        opportunities.extend(self._analyze_git_history())
        
        # Static analysis
        opportunities.extend(self._run_static_analysis())
        
        # Security scanning
        opportunities.extend(self._security_scan())
        
        # Dependency analysis
        opportunities.extend(self._dependency_analysis())
        
        # Score and prioritize
        for item in opportunities:
            item['composite_score'] = self._calculate_composite_score(item)
        
        return sorted(opportunities, key=lambda x: x['composite_score'], reverse=True)
    
    def _analyze_git_history(self) -> List[Dict[str, Any]]:
        """Extract opportunities from git history."""
        opportunities = []
        
        # Find TODO/FIXME/HACK comments
        result = subprocess.run([
            'grep', '-r', '-n', '-i', 
            '--include=*.py', '--include=*.md',
            r'\(TODO\|FIXME\|HACK\|XXX\)', '.'
        ], capture_output=True, text=True, cwd=self.repo_path)
        
        for line in result.stdout.split('\n'):
            if line.strip():
                opportunities.append({
                    'type': 'technical_debt',
                    'title': f'Address TODO/FIXME: {line[:50]}...',
                    'source': 'git_history',
                    'effort_hours': 2,
                    'impact_score': 30,
                    'confidence': 0.8,
                    'ease': 0.7,
                    'file': line.split(':')[0] if ':' in line else 'unknown'
                })
        
        return opportunities
    
    def _run_static_analysis(self) -> List[Dict[str, Any]]:
        """Run static analysis tools and extract opportunities."""
        opportunities = []
        
        # Run flake8
        result = subprocess.run([
            'flake8', '--format=json', 'robo_rlhf/'
        ], capture_output=True, text=True, cwd=self.repo_path)
        
        if result.stdout:
            try:
                flake8_issues = json.loads(result.stdout)
                for issue in flake8_issues:
                    opportunities.append({
                        'type': 'code_quality',
                        'title': f'Fix {issue["code"]}: {issue["text"][:50]}...',
                        'source': 'static_analysis',
                        'effort_hours': 0.5,
                        'impact_score': 20,
                        'confidence': 0.9,
                        'ease': 0.8,
                        'file': issue['filename']
                    })
            except json.JSONDecodeError:
                pass
        
        return opportunities
    
    def _security_scan(self) -> List[Dict[str, Any]]:
        """Run security scanning and identify vulnerabilities."""
        opportunities = []
        
        # Run bandit
        result = subprocess.run([
            'bandit', '-r', '-f', 'json', 'robo_rlhf/'
        ], capture_output=True, text=True, cwd=self.repo_path)
        
        if result.stdout:
            try:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get('results', []):
                    opportunities.append({
                        'type': 'security',
                        'title': f'Security: {issue["test_name"]}',
                        'source': 'security_scan',
                        'effort_hours': 3,
                        'impact_score': 80,
                        'confidence': 0.9,
                        'ease': 0.6,
                        'file': issue['filename'],
                        'priority_boost': 2.0
                    })
            except json.JSONDecodeError:
                pass
        
        return opportunities
    
    def _dependency_analysis(self) -> List[Dict[str, Any]]:
        """Analyze dependencies for updates and vulnerabilities."""
        opportunities = []
        
        # Run safety check
        result = subprocess.run([
            'safety', 'check', '--json'
        ], capture_output=True, text=True, cwd=self.repo_path)
        
        if result.stdout:
            try:
                safety_results = json.loads(result.stdout)
                for vuln in safety_results:
                    opportunities.append({
                        'type': 'dependency_security',
                        'title': f'Update {vuln["package"]} (security)',
                        'source': 'dependency_scan',
                        'effort_hours': 1,
                        'impact_score': 70,
                        'confidence': 1.0,
                        'ease': 0.9,
                        'priority_boost': 1.8
                    })
            except json.JSONDecodeError:
                pass
        
        return opportunities
    
    def _calculate_composite_score(self, item: Dict[str, Any]) -> float:
        """Calculate composite value score using WSJF + ICE + Technical Debt."""
        weights = self.config['scoring']['weights']['developing']
        
        # WSJF calculation
        impact = item.get('impact_score', 50) / 100
        urgency = 0.5  # Default medium urgency
        risk_reduction = 0.4 if item['type'] == 'security' else 0.2
        opportunity = 0.3
        
        cost_of_delay = impact + urgency + risk_reduction + opportunity
        job_size = item.get('effort_hours', 4) / 8  # Normalize to days
        wsjf = cost_of_delay / max(job_size, 0.1)
        
        # ICE calculation
        confidence = item.get('confidence', 0.7)
        ease = item.get('ease', 0.5)
        ice = impact * confidence * ease * 100
        
        # Technical debt score
        debt_score = 50 if item['type'] == 'technical_debt' else 20
        
        # Composite score
        composite = (
            weights['wsjf'] * wsjf * 10 +
            weights['ice'] * ice +
            weights['technicalDebt'] * debt_score +
            weights['security'] * (100 if item['type'] == 'security' else 0)
        )
        
        # Apply priority boosts
        if 'priority_boost' in item:
            composite *= item['priority_boost']
        
        return composite
    
    def select_next_best_value(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the next highest-value item to execute."""
        for item in opportunities:
            # Apply selection filters
            if self._meets_dependencies(item) and self._within_risk_threshold(item):
                return item
        
        # Fallback to housekeeping task
        return self._generate_housekeeping_task()
    
    def _meets_dependencies(self, item: Dict[str, Any]) -> bool:
        """Check if item dependencies are satisfied."""
        return True  # Simplified for MVP
    
    def _within_risk_threshold(self, item: Dict[str, Any]) -> bool:
        """Check if item risk is within acceptable threshold."""
        risk = item.get('risk_score', 0.3)
        return risk <= self.config['scoring']['thresholds']['maxRisk']
    
    def _generate_housekeeping_task(self) -> Dict[str, Any]:
        """Generate a low-risk housekeeping task."""
        return {
            'type': 'maintenance',
            'title': 'Update documentation formatting',
            'source': 'housekeeping',
            'effort_hours': 1,
            'impact_score': 20,
            'confidence': 1.0,
            'ease': 0.9,
            'composite_score': 25.0
        }
    
    def update_metrics(self, executed_item: Dict[str, Any], outcome: Dict[str, Any]):
        """Update value metrics with execution results."""
        with open(self.metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add execution history
        if 'executionHistory' not in metrics:
            metrics['executionHistory'] = []
        
        metrics['executionHistory'].append({
            'timestamp': datetime.now().isoformat(),
            'item': executed_item,
            'outcome': outcome
        })
        
        # Update learning metrics
        self._update_learning_metrics(executed_item, outcome, metrics)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _update_learning_metrics(self, item: Dict[str, Any], outcome: Dict[str, Any], metrics: Dict[str, Any]):
        """Update learning and calibration metrics."""
        if 'learning_metrics' not in metrics:
            metrics['learning_metrics'] = {
                'estimation_accuracy': [],
                'value_prediction_accuracy': [],
                'total_executions': 0
            }
        
        # Calculate accuracy ratios
        predicted_effort = item.get('effort_hours', 4)
        actual_effort = outcome.get('actual_effort', predicted_effort)
        effort_ratio = actual_effort / predicted_effort
        
        predicted_impact = item.get('impact_score', 50)
        actual_impact = outcome.get('actual_impact', predicted_impact)
        impact_ratio = actual_impact / predicted_impact
        
        metrics['learning_metrics']['estimation_accuracy'].append(effort_ratio)
        metrics['learning_metrics']['value_prediction_accuracy'].append(impact_ratio)
        metrics['learning_metrics']['total_executions'] += 1


def main():
    """Main autonomous execution loop."""
    repo_path = Path.cwd()
    engine = ValueDiscoveryEngine(repo_path)
    
    print("🚀 Terragon Autonomous SDLC Engine - Value Discovery Mode")
    print(f"Repository: {repo_path.name}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Discover opportunities
    print("\n🔍 Discovering value opportunities...")
    opportunities = engine.discover_value_opportunities()
    
    if not opportunities:
        print("✅ No high-value opportunities discovered. Repository in excellent state!")
        return
    
    # Select next best value item
    next_item = engine.select_next_best_value(opportunities)
    
    print(f"\n🎯 Next Best Value Item:")
    print(f"Title: {next_item['title']}")
    print(f"Type: {next_item['type']}")
    print(f"Score: {next_item['composite_score']:.1f}")
    print(f"Effort: {next_item['effort_hours']}h")
    
    # Update backlog
    print(f"\n📋 Total opportunities discovered: {len(opportunities)}")
    print(f"High-priority items (score > 70): {len([i for i in opportunities if i['composite_score'] > 70])}")
    
    print(f"\n📊 Value discovery complete. Next execution in 1 hour.")


if __name__ == "__main__":
    main()