#!/usr/bin/env python3
"""
Command Line Interface for Research Link Technology Landscaping - Modeling Workflow

This module provides a Typer-based CLI for running the complete modeling workflow:
1. Keyword extraction from research grants
2. Keyword clustering and harmonization
3. Topic classification and assignment
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
import yaml

# Add the project root to the Python path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.keywords_extraction import extract_grant_keywords
from modeling.keywords_clustering import cluster_research_keywords
from modeling.topic_classification import classify_grants_by_topics

console = Console()
app = typer.Typer(
    name="modeling-cli",
    help="Research Link Technology Landscaping - Modeling Workflow CLI",
    add_completion=False
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]Error parsing YAML configuration: {e}[/red]")
        raise typer.Exit(1)


def setup_logging(verbose: bool = False, logs_dir: str = "logs"):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(logs_dir) / "modeling.log"),
            logging.StreamHandler()
        ]
    )


def validate_config(config: dict) -> bool:
    """Validate configuration structure."""
    required_sections = ['data', 'modeling', 'logs']
    for section in required_sections:
        if section not in config:
            console.print(f"[red]Error: Missing required configuration section: {section}[/red]")
            return False
    
    # Validate data section
    data_config = config['data']
    required_data_keys = ['base_dir', 'grants_file']
    for key in required_data_keys:
        if key not in data_config:
            console.print(f"[red]Error: Missing required data configuration: {key}[/red]")
            return False
    
    return True


@app.command("extract")
def extract_keywords(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Configuration YAML file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model from config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """Extract keywords from research grants using LLM analysis."""
    
    console.print(Panel.fit("🔍 Research Keywords Extraction", style="bold blue"))
    
    # Load configuration
    config = load_config(config_file)
    if not validate_config(config):
        raise typer.Exit(1)
    
    # Setup logging
    setup_logging(verbose, config['logs']['base_dir'])
    
    # Build file paths
    grants_file = Path(config['data']['base_dir']) / config['data']['grants_file']
    
    if not grants_file.exists():
        console.print(f"[red]Error: Grants file not found: {grants_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"📊 Grants file: {grants_file}")
    console.print(f"🤖 Model: {model or config['modeling']['default_model']}")
    
    if dry_run:
        console.print("[yellow]Dry run mode - no extraction will be performed[/yellow]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Extracting keywords from grants...", total=None)
            
            # Import and run inspect AI task
            from inspect_ai import eval
            
            keyword_task = extract_grant_keywords(grants_file=str(grants_file))
            
            # Override model if specified
            if model:
                keyword_task.config.model = model
            elif 'default_model' in config['modeling']:
                keyword_task.config.model = config['modeling']['default_model']
            
            # Run the extraction
            eval(keyword_task, log_dir=config['logs']['base_dir'])
            
            progress.update(task, completed=True, description="✅ Keywords extraction completed")
        
        console.print("[green]✅ Keywords extraction completed successfully![/green]")
        console.print(f"📁 Logs saved to: {config['logs']['base_dir']}")
        
    except Exception as e:
        console.print(f"[red]❌ Error during keyword extraction: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("cluster")
def cluster_keywords(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Configuration YAML file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model from config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """Cluster and harmonize extracted keywords into research topics."""
    
    console.print(Panel.fit("🔗 Keywords Clustering & Harmonization", style="bold green"))
    
    # Load configuration
    config = load_config(config_file)
    if not validate_config(config):
        raise typer.Exit(1)
    
    # Setup logging
    setup_logging(verbose, config['logs']['base_dir'])
    
    data_dir = config['data']['base_dir']
    output_file = config['data'].get('keyword_clusters_file', 'keyword_clusters.json')
    
    console.print(f"📁 Data directory: {data_dir}")
    console.print(f"💾 Output file: {output_file}")
    console.print(f"🤖 Model: {model or config['modeling']['default_model']}")
    
    if dry_run:
        console.print("[yellow]Dry run mode - no clustering will be performed[/yellow]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Clustering keywords into topics...", total=None)
            
            # Import and run inspect AI task
            from inspect_ai import eval
            
            clustering_task = cluster_research_keywords(
                data_dir=data_dir,
                output_file=output_file
            )
            
            # Override model if specified
            if model:
                clustering_task.config.model = model
            elif 'default_model' in config['modeling']:
                clustering_task.config.model = config['modeling']['default_model']
            
            # Run the clustering
            eval(clustering_task, log_dir=config['logs']['base_dir'])
            
            progress.update(task, completed=True, description="✅ Keywords clustering completed")
        
        console.print("[green]✅ Keywords clustering completed successfully![/green]")
        console.print(f"📁 Logs saved to: {config['logs']['base_dir']}")
        
    except Exception as e:
        console.print(f"[red]❌ Error during keyword clustering: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("classify")
def classify_topics(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Configuration YAML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """Classify grants into topics based on extracted keywords and clusters."""
    
    console.print(Panel.fit("📋 Grant Topic Classification", style="bold yellow"))
    
    # Load configuration
    config = load_config(config_file)
    if not validate_config(config):
        raise typer.Exit(1)
    
    # Setup logging
    setup_logging(verbose, config['logs']['base_dir'])
    
    # Build file paths
    grants_file = Path(config['data']['base_dir']) / config['data']['grants_file']
    output_file = Path(config['data']['base_dir']) / config['data'].get('grant_topic_assignments_file', 'grant_topic_assignments.json')
    
    console.print(f"📊 Grants file: {grants_file}")
    console.print(f"💾 Output file: {output_file}")
    
    # Get classification parameters
    classification_config = config['modeling'].get('classification', {})
    min_keyword_matches = classification_config.get('min_keyword_matches', 2)
    
    console.print(f"🎯 Min keyword matches: {min_keyword_matches}")
    
    if dry_run:
        console.print("[yellow]Dry run mode - no classification will be performed[/yellow]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Classifying grants into topics...", total=None)
            
            # Run the classification
            result = classify_grants_by_topics(
                grants_file=str(grants_file),
                logs_dir=config['logs']['base_dir'],
                output_file=str(output_file),
                min_keyword_matches=min_keyword_matches
            )
            
            progress.update(task, completed=True, description="✅ Grant classification completed")
        
        # Display results summary
        summary_table = Table(title="Classification Results")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary = result.to_dict()['summary']
        summary_table.add_row("Total Grants", str(summary['total_grants']))
        summary_table.add_row("Assigned Grants", str(summary['assigned_grants']))
        summary_table.add_row("Unassigned Grants", str(summary['unassigned_grants']))
        summary_table.add_row("Total Topics", str(summary['total_topics']))
        
        console.print(summary_table)
        console.print("[green]✅ Grant classification completed successfully![/green]")
        console.print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        console.print(f"[red]❌ Error during topic classification: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("run-all")
def run_full_workflow(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Configuration YAML file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model from config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
    skip_extraction: bool = typer.Option(False, "--skip-extraction", help="Skip keyword extraction step"),
    skip_clustering: bool = typer.Option(False, "--skip-clustering", help="Skip keyword clustering step"),
    skip_classification: bool = typer.Option(False, "--skip-classification", help="Skip topic classification step")
):
    """Run the complete modeling workflow: extract keywords, cluster them, and classify grants."""
    
    console.print(Panel.fit("🚀 Complete Research Modeling Workflow", style="bold magenta"))
    
    # Load and validate configuration
    config = load_config(config_file)
    if not validate_config(config):
        raise typer.Exit(1)
    
    console.print(f"📖 Configuration: {config_file}")
    console.print(f"🤖 Model: {model or config['modeling']['default_model']}")
    
    if dry_run:
        console.print("[yellow]Dry run mode - no operations will be performed[/yellow]")
    
    workflow_steps = []
    if not skip_extraction:
        workflow_steps.append("1. Extract Keywords")
    if not skip_clustering:
        workflow_steps.append("2. Cluster Keywords")
    if not skip_classification:
        workflow_steps.append("3. Classify Topics")
    
    console.print(f"📋 Workflow steps: {', '.join(workflow_steps)}")
    
    try:
        # Step 1: Extract Keywords
        if not skip_extraction:
            console.print("\n" + "="*50)
            extract_keywords(config_file, model, verbose, dry_run)
        
        # Step 2: Cluster Keywords
        if not skip_clustering:
            console.print("\n" + "="*50)
            cluster_keywords(config_file, model, verbose, dry_run)
        
        # Step 3: Classify Topics
        if not skip_classification:
            console.print("\n" + "="*50)
            classify_topics(config_file, verbose, dry_run)
        
        console.print("\n" + "="*50)
        console.print(Panel.fit("🎉 Complete Workflow Finished Successfully!", style="bold green"))
        
    except typer.Exit:
        # Re-raise typer exits to maintain proper error codes
        raise
    except Exception as e:
        console.print(f"[red]❌ Workflow failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("status")
def check_status(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Configuration YAML file")
):
    """Check the status of data files and workflow outputs."""
    
    console.print(Panel.fit("📊 Workflow Status Check", style="bold cyan"))
    
    config = load_config(config_file)
    
    status_table = Table(title="File Status")
    status_table.add_column("File", style="cyan")
    status_table.add_column("Path", style="white")
    status_table.add_column("Status", style="green")
    status_table.add_column("Size", style="yellow")
    
    # Check input files
    grants_file = Path(config['data']['base_dir']) / config['data']['grants_file']
    if grants_file.exists():
        size = f"{grants_file.stat().st_size / 1024:.1f} KB"
        status_table.add_row("Grants Data", str(grants_file), "✅ Exists", size)
    else:
        status_table.add_row("Grants Data", str(grants_file), "❌ Missing", "-")
    
    # Check output files
    clusters_file = Path(config['data']['base_dir']) / config['data'].get('keyword_clusters_file', 'keyword_clusters.json')
    if clusters_file.exists():
        size = f"{clusters_file.stat().st_size / 1024:.1f} KB"
        status_table.add_row("Keyword Clusters", str(clusters_file), "✅ Exists", size)
    else:
        status_table.add_row("Keyword Clusters", str(clusters_file), "❌ Missing", "-")
    
    assignments_file = Path(config['data']['base_dir']) / config['data'].get('grant_topic_assignments_file', 'grant_topic_assignments.json')
    if assignments_file.exists():
        size = f"{assignments_file.stat().st_size / 1024:.1f} KB"
        status_table.add_row("Topic Assignments", str(assignments_file), "✅ Exists", size)
    else:
        status_table.add_row("Topic Assignments", str(assignments_file), "❌ Missing", "-")
    
    # Check logs directory
    logs_dir = Path(config['logs']['base_dir'])
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.eval"))
        status_table.add_row("Logs Directory", str(logs_dir), f"✅ {len(log_files)} files", "-")
    else:
        status_table.add_row("Logs Directory", str(logs_dir), "❌ Missing", "-")
    
    console.print(status_table)


if __name__ == "__main__":
    app()
