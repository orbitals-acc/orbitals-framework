# python scripts/create_project.py project_template.yaml my_project
# python scripts/create_project.py project_template.yaml my_project -b ~/projects/

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any


def create_required_directories_from_yaml(yaml_file: str, project_name: str, base_path: str = ".") -> None:
    """
    Reads a YAML template and creates only required directories
    
    Args:
        yaml_file: Path to YAML file with template
        project_name: Project name
        base_path: Base path for project creation
    """
    
    # Load YAML template
    with open(yaml_file, 'r', encoding='utf-8') as f:
        try:
            template = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            sys.exit(1)
    
    # Get project template
    project_template = template.get("project_template")
    if not project_template:
        print("Error: 'project_template' missing in YAML file")
        sys.exit(1)
    
    # Create project root directory
    project_path = Path(base_path) / project_name
    project_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating project: {project_name}")
    print(f"Path: {project_path.absolute()}")
    print(f"Template: {yaml_file}")
    print("-" * 50)
    
    # Counter for created directories
    dirs_created = 0
    dirs_skipped = 0
    
    def process_directory(path: Path, dir_config: Dict[str, Any], parent_required: bool = True) -> None:
        """
        Recursively processes directories and creates required ones only
        
        Args:
            path: Current path
            dir_config: Directory configuration
            parent_required: Whether parent directory is required
        """
        nonlocal dirs_created, dirs_skipped
        
        # Check if current directory is required
        is_required = dir_config.get("required", True) and parent_required
        
        # Create directory if required
        if is_required:
            dir_name = dir_config["name"]
            dir_path = path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            dirs_created += 1
            
            # Print info about created directory
            description = dir_config.get("description", "")
            print(f"✓ Created directory: {dir_path.relative_to(project_path)}")
            if description:
                print(f"  Description: {description}")
            
            # Recursively process nested structure
            if "structure" in dir_config:
                for subdir_config in dir_config["structure"]:
                    process_directory(dir_path, subdir_config, is_required)
        else:
            # Skip optional directory
            dir_name = dir_config["name"]
            dir_path = path / dir_name
            dirs_skipped += 1
            print(f"✗ Skipped directory: {dir_path.relative_to(project_path)} (required=False)")
    
    # Process all directories from template
    directories = project_template.get("directories", [])
    for dir_config in directories:
        process_directory(project_path, dir_config)
    
    print("-" * 50)
    print(f"✅ Done!")
    print(f"📁 Directories created: {dirs_created}")
    print(f"⏭️  Directories skipped: {dirs_skipped}")
    print(f"\nCreated project structure:")
    print_structure(project_path)

def print_structure(path: Path, prefix: str = "") -> None:
    """
    Recursively prints directory structure
    """
    items = list(path.iterdir())
    
    for i, item in enumerate(sorted(items)):
        is_last = i == len(items) - 1
        
        if item.is_dir():
            # Print directory
            print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
            
            # Recursively print contents
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_structure(item, new_prefix)

def main():
    parser = argparse.ArgumentParser(
        description="Creates directory structure from YAML template (only required=True)"
    )
    parser.add_argument(
        "yaml_file",
        help="Path to YAML file with structure template"
    )
    parser.add_argument(
        "project_name",
        help="Name of project to create"
    )
    parser.add_argument(
        "-b", "--base-path",
        default=".",
        help="Base path for project creation (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Check if YAML file exists
    if not os.path.exists(args.yaml_file):
        print(f"Error: YAML file not found: {args.yaml_file}")
        sys.exit(1)
    
    # Create structure
    create_required_directories_from_yaml(
        yaml_file=args.yaml_file,
        project_name=args.project_name,
        base_path=args.base_path
    )

if __name__ == "__main__":
    main()
