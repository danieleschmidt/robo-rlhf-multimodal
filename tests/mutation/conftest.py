"""
Mutation testing configuration and utilities.
"""

import pytest
import ast
import random
from typing import List, Dict, Any, Callable
from pathlib import Path


class CodeMutator:
    """Simple code mutator for mutation testing."""
    
    def __init__(self):
        self.mutations = [
            self._mutate_comparison_operators,
            self._mutate_arithmetic_operators,
            self._mutate_boolean_operators,
            self._mutate_constants
        ]
    
    def _mutate_comparison_operators(self, code: str) -> List[str]:
        """Mutate comparison operators."""
        mutations = []
        replacements = {
            '==': '!=',
            '!=': '==',
            '<': '>=',
            '<=': '>',
            '>': '<=',
            '>=': '<'
        }
        
        for original, replacement in replacements.items():
            if original in code:
                mutated = code.replace(original, replacement, 1)
                mutations.append(mutated)
        
        return mutations
    
    def _mutate_arithmetic_operators(self, code: str) -> List[str]:
        """Mutate arithmetic operators."""
        mutations = []
        replacements = {
            '+': '-',
            '-': '+',
            '*': '/',
            '/': '*',
            '%': '*'
        }
        
        for original, replacement in replacements.items():
            if original in code and not code.count(original) == code.count(f'"{original}"'):
                mutated = code.replace(original, replacement, 1)
                mutations.append(mutated)
        
        return mutations
    
    def _mutate_boolean_operators(self, code: str) -> List[str]:
        """Mutate boolean operators."""
        mutations = []
        replacements = {
            ' and ': ' or ',
            ' or ': ' and ',
            'True': 'False',
            'False': 'True'
        }
        
        for original, replacement in replacements.items():
            if original in code:
                mutated = code.replace(original, replacement, 1)
                mutations.append(mutated)
        
        return mutations
    
    def _mutate_constants(self, code: str) -> List[str]:
        """Mutate numeric constants."""
        mutations = []
        
        # Simple regex-like approach for numeric constants
        import re
        numbers = re.findall(r'\b\d+\b', code)
        
        for number in numbers:
            try:
                original_value = int(number)
                # Mutate by adding/subtracting 1
                for mutation_value in [original_value + 1, original_value - 1, 0]:
                    if mutation_value != original_value:
                        mutated = code.replace(str(original_value), str(mutation_value), 1)
                        mutations.append(mutated)
            except ValueError:
                continue
        
        return mutations
    
    def generate_mutants(self, source_code: str) -> List[str]:
        """Generate all possible mutants for the given source code."""
        all_mutants = []
        
        for mutation_func in self.mutations:
            mutants = mutation_func(source_code)
            all_mutants.extend(mutants)
        
        return all_mutants


class MutationTester:
    """Run mutation tests."""
    
    def __init__(self, mutator: CodeMutator):
        self.mutator = mutator
        self.results: Dict[str, Any] = {}
    
    def run_mutation_test(self, source_file: Path, test_function: Callable) -> Dict[str, Any]:
        """Run mutation testing on a source file."""
        with open(source_file, 'r') as f:
            original_code = f.read()
        
        mutants = self.mutator.generate_mutants(original_code)
        
        killed_mutants = 0
        survived_mutants = 0
        
        for i, mutant_code in enumerate(mutants):
            try:
                # This is a simplified approach - in practice, you'd need to
                # actually execute the mutant code and run tests against it
                test_passed = self._run_test_on_mutant(mutant_code, test_function)
                
                if test_passed:
                    survived_mutants += 1
                else:
                    killed_mutants += 1
                    
            except Exception:
                # Mutant caused compilation error, consider it killed
                killed_mutants += 1
        
        total_mutants = len(mutants)
        mutation_score = killed_mutants / total_mutants if total_mutants > 0 else 0
        
        result = {
            "total_mutants": total_mutants,
            "killed_mutants": killed_mutants,
            "survived_mutants": survived_mutants,
            "mutation_score": mutation_score,
            "source_file": str(source_file)
        }
        
        self.results[str(source_file)] = result
        return result
    
    def _run_test_on_mutant(self, mutant_code: str, test_function: Callable) -> bool:
        """Run test function on mutant code."""
        # This is a placeholder - actual implementation would need to:
        # 1. Write mutant code to temporary file
        # 2. Import and execute it
        # 3. Run the test function
        # 4. Return whether test passed or failed
        
        # For now, randomly determine if mutant survives (for demo purposes)
        return random.random() < 0.1  # 10% survival rate
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all mutation test results."""
        if not self.results:
            return {}
        
        total_mutants = sum(r["total_mutants"] for r in self.results.values())
        total_killed = sum(r["killed_mutants"] for r in self.results.values())
        
        return {
            "total_files_tested": len(self.results),
            "total_mutants": total_mutants,
            "total_killed": total_killed,
            "overall_mutation_score": total_killed / total_mutants if total_mutants > 0 else 0,
            "file_results": self.results
        }


@pytest.fixture
def code_mutator():
    """Code mutator fixture."""
    return CodeMutator()


@pytest.fixture
def mutation_tester(code_mutator):
    """Mutation tester fixture."""
    return MutationTester(code_mutator)


@pytest.fixture
def mutation_config():
    """Configuration for mutation testing."""
    return {
        "target_directories": ["robo_rlhf/core", "robo_rlhf/models"],
        "excluded_files": ["__init__.py", "test_*.py"],
        "mutation_score_threshold": 0.8,  # 80% mutation score required
        "max_mutants_per_file": 100,
        "timeout_per_mutant": 5.0  # seconds
    }


class MutationReporter:
    """Generate reports for mutation testing results."""
    
    def __init__(self):
        self.reports: List[Dict[str, Any]] = []
    
    def add_result(self, result: Dict[str, Any]):
        """Add a mutation test result."""
        self.reports.append(result)
    
    def generate_html_report(self, output_file: Path):
        """Generate HTML report."""
        html_content = self._create_html_report()
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def generate_json_report(self, output_file: Path):
        """Generate JSON report."""
        import json
        with open(output_file, 'w') as f:
            json.dump(self.reports, f, indent=2)
    
    def _create_html_report(self) -> str:
        """Create HTML report content."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mutation Testing Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f0f0f0; padding: 15px; margin-bottom: 20px; }
                .file-result { margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; }
                .score-good { color: green; font-weight: bold; }
                .score-poor { color: red; font-weight: bold; }
                .score-medium { color: orange; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Mutation Testing Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total files tested: {}</p>
                <p>Overall mutation score: {:.2%}</p>
            </div>
            <h2>File Results</h2>
            {}
        </body>
        </html>
        """.format(
            len(self.reports),
            sum(r.get('mutation_score', 0) for r in self.reports) / len(self.reports) if self.reports else 0,
            self._create_file_results_html()
        )
    
    def _create_file_results_html(self) -> str:
        """Create HTML for file results."""
        html_parts = []
        for result in self.reports:
            score = result.get('mutation_score', 0)
            score_class = 'score-good' if score >= 0.8 else 'score-poor' if score < 0.5 else 'score-medium'
            
            html_parts.append(f"""
            <div class="file-result">
                <h3>{result.get('source_file', 'Unknown')}</h3>
                <p>Mutation Score: <span class="{score_class}">{score:.2%}</span></p>
                <p>Mutants: {result.get('killed_mutants', 0)}/{result.get('total_mutants', 0)} killed</p>
            </div>
            """)
        
        return ''.join(html_parts)


@pytest.fixture
def mutation_reporter():
    """Mutation reporter fixture."""
    return MutationReporter()


def pytest_configure(config):
    """Configure pytest markers for mutation testing."""
    config.addinivalue_line(
        "markers", "mutation: mark test as a mutation test"
    )