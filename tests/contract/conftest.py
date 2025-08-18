"""
Contract testing configuration and fixtures for API testing.
"""

import pytest
import json
import yaml
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class APIContract:
    """API contract definition."""
    endpoint: str
    method: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    status_codes: List[int]
    headers: Dict[str, str] = None


class ContractValidator:
    """Validate API responses against contracts."""
    
    def __init__(self):
        self.contracts: Dict[str, APIContract] = {}
    
    def load_contracts(self, contract_dir: Path):
        """Load API contracts from directory."""
        for contract_file in contract_dir.glob("*.json"):
            with open(contract_file) as f:
                contract_data = json.load(f)
                contract = APIContract(**contract_data)
                self.contracts[contract.endpoint] = contract
    
    def validate_response(self, endpoint: str, response: Dict[str, Any]) -> bool:
        """Validate response against contract."""
        if endpoint not in self.contracts:
            return False
        
        contract = self.contracts[endpoint]
        
        # Validate status code
        if response.get("status_code") not in contract.status_codes:
            return False
        
        # Validate response schema (simplified)
        response_data = response.get("data", {})
        return self._validate_schema(response_data, contract.response_schema)
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Simple schema validation."""
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False
        
        # Check field types
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return False
        
        return True
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True


@pytest.fixture
def contract_validator():
    """Contract validator fixture."""
    validator = ContractValidator()
    
    # Load contracts if they exist
    contract_dir = Path(__file__).parent / "contracts"
    if contract_dir.exists():
        validator.load_contracts(contract_dir)
    
    return validator


@pytest.fixture
def api_client():
    """Mock API client for testing."""
    class MockAPIClient:
        def __init__(self):
            self.base_url = "http://localhost:8080"
        
        async def get(self, endpoint: str, **kwargs):
            """Mock GET request."""
            return {
                "status_code": 200,
                "data": {"message": "success", "endpoint": endpoint},
                "headers": {"content-type": "application/json"}
            }
        
        async def post(self, endpoint: str, data: Dict[str, Any] = None, **kwargs):
            """Mock POST request."""
            return {
                "status_code": 201,
                "data": {"message": "created", "endpoint": endpoint, "input": data},
                "headers": {"content-type": "application/json"}
            }
        
        async def put(self, endpoint: str, data: Dict[str, Any] = None, **kwargs):
            """Mock PUT request."""
            return {
                "status_code": 200,
                "data": {"message": "updated", "endpoint": endpoint, "input": data},
                "headers": {"content-type": "application/json"}
            }
        
        async def delete(self, endpoint: str, **kwargs):
            """Mock DELETE request."""
            return {
                "status_code": 204,
                "data": None,
                "headers": {"content-type": "application/json"}
            }
    
    return MockAPIClient()


@pytest.fixture
def sample_contracts():
    """Sample API contracts for testing."""
    return {
        "/api/v1/health": APIContract(
            endpoint="/api/v1/health",
            method="GET",
            request_schema={},
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "version": {"type": "string"}
                },
                "required": ["status"]
            },
            status_codes=[200]
        ),
        "/api/v1/preferences": APIContract(
            endpoint="/api/v1/preferences",
            method="POST",
            request_schema={
                "type": "object",
                "properties": {
                    "pair_id": {"type": "string"},
                    "preference": {"type": "integer"},
                    "annotator_id": {"type": "string"}
                },
                "required": ["pair_id", "preference", "annotator_id"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string"}
                },
                "required": ["id", "status"]
            },
            status_codes=[201, 400, 422]
        ),
        "/api/v1/training/status": APIContract(
            endpoint="/api/v1/training/status",
            method="GET",
            request_schema={},
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "progress": {"type": "number"},
                    "epoch": {"type": "integer"},
                    "metrics": {"type": "object"}
                },
                "required": ["status", "progress"]
            },
            status_codes=[200, 404]
        )
    }


def create_contract_file(endpoint: str, contract: APIContract, output_dir: Path):
    """Create a contract file for the given endpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    contract_data = {
        "endpoint": contract.endpoint,
        "method": contract.method,
        "request_schema": contract.request_schema,
        "response_schema": contract.response_schema,
        "status_codes": contract.status_codes,
        "headers": contract.headers or {}
    }
    
    filename = endpoint.replace("/", "_").replace(":", "") + ".json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(contract_data, f, indent=2)


@pytest.fixture
def contract_generator(tmp_path):
    """Contract file generator fixture."""
    contract_dir = tmp_path / "contracts"
    
    def generate_contracts(contracts: Dict[str, APIContract]):
        for endpoint, contract in contracts.items():
            create_contract_file(endpoint, contract, contract_dir)
        return contract_dir
    
    return generate_contracts


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "base_url": "http://localhost:8080",
        "timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "expected_response_time": 1.0,  # seconds
        "rate_limit": 100  # requests per minute
    }


class IntegrationTestHelper:
    """Helper class for integration testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.response_times: List[float] = []
    
    def record_response_time(self, time: float):
        """Record API response time."""
        self.response_times.append(time)
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def assert_response_time_acceptable(self):
        """Assert that response times are within acceptable limits."""
        avg_time = self.get_average_response_time()
        assert avg_time <= self.config["expected_response_time"], \
            f"Average response time {avg_time:.2f}s exceeds threshold {self.config['expected_response_time']}s"


@pytest.fixture
def integration_helper(integration_test_config):
    """Integration test helper fixture."""
    return IntegrationTestHelper(integration_test_config)