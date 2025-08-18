#!/usr/bin/env python3
"""
Generate Software Bill of Materials (SBOM) for the project.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pkg_resources
import hashlib
import os


class SBOMGenerator:
    """Generate SBOM in SPDX format."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.sbom_data = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": "robo-rlhf-multimodal-sbom",
            "documentNamespace": f"https://github.com/danieleschmidt/robo-rlhf-multimodal/sbom-{datetime.now().isoformat()}",
            "creationInfo": {
                "created": datetime.now().isoformat(),
                "creators": ["Tool: SBOM Generator"],
                "licenseListVersion": "3.19"
            },
            "packages": [],
            "relationships": []
        }
    
    def get_installed_packages(self) -> List[Dict[str, Any]]:
        """Get list of installed packages with versions."""
        packages = []
        
        try:
            # Get packages from pip freeze
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                                  capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if line and '==' in line:
                    name, version = line.split('==', 1)
                    packages.append({
                        "name": name,
                        "version": version,
                        "source": "pip"
                    })
        except subprocess.CalledProcessError:
            print("Warning: Could not get pip packages", file=sys.stderr)
        
        # Also try pkg_resources for additional info
        try:
            for dist in pkg_resources.working_set:
                package_info = {
                    "name": dist.project_name,
                    "version": dist.version,
                    "location": dist.location,
                    "source": "pkg_resources"
                }
                
                # Add license info if available
                try:
                    if hasattr(dist, '_get_metadata'):
                        metadata = dist._get_metadata('METADATA') or dist._get_metadata('PKG-INFO')
                        if metadata:
                            for line in metadata.split('\n'):
                                if line.startswith('License:'):
                                    package_info["license"] = line.split(':', 1)[1].strip()
                                    break
                except:
                    pass
                
                # Check if this package is already in our list
                existing = next((p for p in packages if p["name"] == dist.project_name), None)
                if not existing:
                    packages.append(package_info)
                else:
                    # Update with additional info
                    existing.update({k: v for k, v in package_info.items() if k not in existing})
        
        except Exception as e:
            print(f"Warning: Could not get pkg_resources info: {e}", file=sys.stderr)
        
        return packages
    
    def get_project_files(self) -> List[Dict[str, Any]]:
        """Get list of project source files."""
        files = []
        source_dirs = ["robo_rlhf", "tests", "scripts", "examples"]
        
        for source_dir in source_dirs:
            source_path = self.project_root / source_dir
            if source_path.exists():
                for file_path in source_path.rglob("*.py"):
                    if file_path.is_file():
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                sha256_hash = hashlib.sha256(content).hexdigest()
                            
                            files.append({
                                "path": str(file_path.relative_to(self.project_root)),
                                "size": len(content),
                                "sha256": sha256_hash,
                                "type": "source"
                            })
                        except Exception as e:
                            print(f"Warning: Could not process {file_path}: {e}", file=sys.stderr)
        
        return files
    
    def generate_package_spdx_id(self, name: str) -> str:
        """Generate SPDX ID for a package."""
        return f"SPDXRef-Package-{name.replace('-', '').replace('_', '').replace('.', '')}"
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate complete SBOM."""
        # Add main project package
        main_package = {
            "SPDXID": "SPDXRef-Package-robo-rlhf-multimodal",
            "name": "robo-rlhf-multimodal",
            "downloadLocation": "https://github.com/danieleschmidt/robo-rlhf-multimodal",
            "filesAnalyzed": True,
            "packageVerificationCode": {
                "packageVerificationCodeValue": self.calculate_package_hash()
            },
            "copyrightText": "Copyright (c) 2025 Daniel Schmidt",
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "supplier": "Person: Daniel Schmidt"
        }
        
        self.sbom_data["packages"].append(main_package)
        
        # Add dependency packages
        packages = self.get_installed_packages()
        for pkg in packages:
            package_data = {
                "SPDXID": self.generate_package_spdx_id(pkg["name"]),
                "name": pkg["name"],
                "versionInfo": pkg["version"],
                "downloadLocation": f"https://pypi.org/project/{pkg['name']}/",
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION",
                "licenseConcluded": pkg.get("license", "NOASSERTION"),
                "licenseDeclared": pkg.get("license", "NOASSERTION"),
                "supplier": "NOASSERTION"
            }
            
            self.sbom_data["packages"].append(package_data)
            
            # Add relationship
            self.sbom_data["relationships"].append({
                "spdxElementId": "SPDXRef-Package-robo-rlhf-multimodal",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": self.generate_package_spdx_id(pkg["name"])
            })
        
        return self.sbom_data
    
    def calculate_package_hash(self) -> str:
        """Calculate verification hash for the main package."""
        files = self.get_project_files()
        
        # Sort files by path for consistent hashing
        files.sort(key=lambda x: x["path"])
        
        # Concatenate all file hashes
        combined_hash = hashlib.sha256()
        for file_info in files:
            combined_hash.update(file_info["sha256"].encode())
        
        return combined_hash.hexdigest()
    
    def save_sbom(self, output_path: Path):
        """Save SBOM to file."""
        sbom = self.generate_sbom()
        
        with open(output_path, 'w') as f:
            json.dump(sbom, f, indent=2, sort_keys=True)
        
        print(f"SBOM generated: {output_path}")
        print(f"Total packages: {len(sbom['packages'])}")
        print(f"Total relationships: {len(sbom['relationships'])}")
    
    def validate_sbom(self, sbom_path: Path) -> bool:
        """Basic validation of generated SBOM."""
        try:
            with open(sbom_path) as f:
                sbom = json.load(f)
            
            required_fields = ["spdxVersion", "dataLicense", "SPDXID", "name", "packages"]
            for field in required_fields:
                if field not in sbom:
                    print(f"Error: Missing required field: {field}")
                    return False
            
            if not sbom["packages"]:
                print("Error: No packages found in SBOM")
                return False
            
            print("SBOM validation passed")
            return True
            
        except Exception as e:
            print(f"Error validating SBOM: {e}")
            return False


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    output_path = project_root / "sbom.json"
    
    # Generate SBOM
    generator = SBOMGenerator(project_root)
    generator.save_sbom(output_path)
    
    # Validate SBOM
    if generator.validate_sbom(output_path):
        print("✅ SBOM generated and validated successfully")
        return 0
    else:
        print("❌ SBOM validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())