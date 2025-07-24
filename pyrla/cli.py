"""
Command Line Interface for PyRLA - Python interface for Research Link Australia API

This module provides a Typer-based CLI for interacting with the RLA API.
"""

import traceback
import json
import asyncio
from typing import Optional, List
from dataclasses import asdict
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from .client import RLAClient
from .models import Researcher, Grant, Organisation, Publication
from .exceptions import RLAError, RLAAuthenticationError, RLANotFoundError
from .utils import get_funding_statistics, filter_grants_by_amount, filter_researchers_by_orcid

# Main app
app = typer.Typer(
    name="pyrla",
    help="PyRLA - Python interface for Research Link Australia API",
    epilog="Visit https://researchlink.ardc.edu.au/v3/api-docs/public for API documentation."
)

# Subcommand apps
researchers_app = typer.Typer(help="Commands for working with researchers")
grants_app = typer.Typer(help="Commands for working with grants")
organisations_app = typer.Typer(help="Commands for working with organisations")
publications_app = typer.Typer(help="Commands for working with publications")

# Register subcommands
app.add_typer(researchers_app, name="researchers")
app.add_typer(grants_app, name="grants")
app.add_typer(organisations_app, name="organisations")
app.add_typer(publications_app, name="publications")

console = Console()

# Global client instance and debug flag
client: Optional[RLAClient] = None
debug_mode: bool = False


def save_json_to_file(data: List[dict], filename: str) -> None:
    """Save JSON data to a file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[green]✓ JSON data saved to: {filename}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving JSON file: {e}[/red]")
        raise typer.Exit(1)


def handle_exception(e: Exception, operation: str = "operation"):
    """Handle exceptions with optional debug information"""
    if debug_mode:
        console.print(f"[red]Error during {operation}:[/red]")
        console.print(f"[red]{type(e).__name__}: {str(e)}[/red]")
        console.print("\n[yellow]Full traceback:[/yellow]")
        console.print(traceback.format_exc())
    else:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Use --debug flag to see full traceback[/dim]")


def get_client() -> RLAClient:
    """Get or initialize the RLA client"""
    global client, debug_mode
    # Reset client if debug mode has changed
    if client is not None and client.debug != debug_mode:
        client = None
    
    if client is None:
        try:
            client = RLAClient(debug=debug_mode)
        except RLAAuthenticationError as e:
            console.print(f"[red]Authentication Error:[/red] {e}")
            console.print("[yellow]Please set your API token using:[/yellow]")
            console.print("export ARL_API_TOKEN='your_token_here'")
            raise typer.Exit(1)
    return client


def run_async(coro):
    """Helper function to run async code in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def format_researcher_table(researchers: List[Researcher], title: str = "Researchers") -> Table:
    """Format researchers data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", min_width=20)
    table.add_column("ORCID", style="green", min_width=15)
    table.add_column("Organisation", style="yellow", min_width=30)
    table.add_column("Publications", justify="right", style="blue")
    table.add_column("Grants", justify="right", style="red")
    
    for researcher in researchers:
        name = researcher.full_name or "N/A"
        orcid = researcher.orcid or "N/A"
        org = ", ".join(researcher.current_organisation_name) if researcher.current_organisation_name else "N/A"
        pubs = str(researcher.summary.publication_count) if researcher.summary and researcher.summary.publication_count is not None else "N/A"
        grants = str(researcher.summary.grant_count) if researcher.summary and researcher.summary.grant_count is not None else "N/A"
        
        table.add_row(name, orcid, org, pubs, grants)
    
    return table


def format_grant_table(grants: List[Grant], title: str = "Grants") -> Table:
    """Format grants data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", min_width=40)
    table.add_column("Funder", style="green", min_width=15)
    table.add_column("Status", style="yellow", min_width=10)
    table.add_column("Amount (AUD)", justify="right", style="blue")
    table.add_column("Start Date", style="red")
    
    for grant in grants:
        grant_title = grant.title or grant.grant_title or "N/A"
        title_display = grant_title[:60] + "..." if len(grant_title) > 60 else grant_title
        funder = grant.funder or "N/A"
        status = grant.status or "N/A"
        amount = f"${grant.funding_amount:,.2f}" if grant.funding_amount is not None else "N/A"
        start_date = grant.start_date or "N/A"
        
        table.add_row(title_display, funder, status, amount, start_date)
    
    return table


def format_organisation_table(orgs: List[Organisation], title: str = "Organisations") -> Table:
    """Format organisations data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", min_width=30)
    table.add_column("ABN", style="green", min_width=15)
    table.add_column("Type", style="yellow", min_width=15)
    table.add_column("Location", style="blue", min_width=20)
    table.add_column("Researchers", justify="right", style="red")
    
    for org in orgs:
        name = org.name or org.organisation_name or "N/A"
        abn = org.abn or "N/A"
        org_type = org.abr_type or "N/A"
        
        # Handle location safely
        location_parts = []
        if org.countries:
            location_parts.append(", ".join(org.countries))
        if org.states:
            location_parts.append(", ".join(org.states))
        location = ", ".join(filter(None, location_parts)) or "N/A"
        
        researcher_count = str(len(org.researcher_ids)) if org.researcher_ids else "0"
        
        table.add_row(name, abn, org_type, location, researcher_count)
    
    return table


def format_publication_table(publications: List[Publication], title: str = "Publications") -> Table:
    """Format publications data as a rich table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", min_width=40)
    table.add_column("Authors", style="green", min_width=20)
    table.add_column("Type", style="yellow", min_width=15)
    table.add_column("Year", justify="right", style="blue")
    table.add_column("Journal", style="red", min_width=20)
    table.add_column("DOI", style="magenta", min_width=15)
    
    for pub in publications:
        title_display = pub.title[:50] + "..." if len(pub.title) > 50 else pub.title or "N/A"
        authors = pub.authors_list[:30] + "..." if len(pub.authors_list) > 30 else pub.authors_list or "N/A"
        pub_type = pub.publication_type or "N/A"
        year = str(pub.publication_year) if pub.publication_year else "N/A"
        journal = pub.journal[:25] + "..." if len(pub.journal) > 25 else pub.journal or "N/A"
        doi = pub.doi[:20] + "..." if len(pub.doi) > 20 else pub.doi or "N/A"
        
        table.add_row(title_display, authors, pub_type, year, journal, doi)
    
    return table


@app.command()
def test_connection():
    """Test connection to the RLA API"""
    try:
        client = get_client()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Testing connection...", total=None)
            success = client.test_connection()
            progress.update(task, completed=100)
        
        if success:
            console.print("[green]✓ Successfully connected to RLA API![/green]")
        else:
            console.print("[red]✗ Failed to connect to RLA API[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error testing connection: {e}[/red]")
        raise typer.Exit(1)


@researchers_app.command("search")
def search_researchers(
    query: str = typer.Argument(..., help="Search query for researchers"),
    page_size: int = typer.Option(10, "--size", "-s", help="Number of results per page"),
    page_number: int = typer.Option(1, "--page", "-p", help="Page number"),
    filter_query: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter query (e.g., 'currentOrganisationName:University')"),
    advanced_query: Optional[str] = typer.Option(None, "--advanced", "-a", help="Advanced search query"),
    orcid_only: bool = typer.Option(False, "--orcid-only", help="Show only researchers with ORCID"),
    all_results: bool = typer.Option(False, "--all", help="Fetch all results from all pages"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Search for researchers in the RLA database"""
    global debug_mode
    debug_mode = debug
    try:
        client = get_client()
        
        if all_results:
            # Fetch all results from all pages using async for better performance
            console.print("[yellow]Fetching all results using concurrent requests...[/yellow]")
            
            async def fetch_all_async():
                async with client:  # Use async context manager
                    return await client.search_all_researchers(
                        value=query,
                        filter_query=filter_query,
                        advanced_search_query=advanced_query,
                        max_concurrent=5  # Limit concurrent requests to be API-friendly
                    )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching all researchers concurrently...", total=None)
                response = run_async(fetch_all_async())
                progress.update(task, completed=100, description=f"Fetched {len(response.results)} researchers")
            
            researchers = response.results
        else:
            # Single page fetch
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching researchers...", total=None)
                response = client.search_researchers_sync(
                    value=query,
                    filter_query=filter_query,
                    page_number=page_number,
                    page_size=page_size,
                    advanced_search_query=advanced_query
                )
                progress.update(task, completed=100)
            
            researchers = response.results
        if orcid_only:
            researchers = filter_researchers_by_orcid(researchers)
        
        if json_file:
            # Save to JSON file
            data = [asdict(r) for r in researchers]
            save_json_to_file(data, json_file)
        else:
            # Display table format to stdout
            if all_results:
                total_pages = 1  # All results in one "page"
                info_panel = Panel(
                    f"Found {response.total_results} researchers total\n"
                    f"Showing all results\n"
                    f"Results displayed: {len(researchers)}",
                    title="Search Results (All Pages)",
                    border_style="blue"
                )
            else:
                total_pages = (response.total_results + response.size - 1) // response.size  # Calculate total pages
                info_panel = Panel(
                    f"Found {response.total_results} researchers total\n"
                    f"Showing page {response.current_page} of {total_pages}\n"
                    f"Results per page: {page_size}",
                    title="Search Results",
                    border_style="blue"
                )
            console.print(info_panel)
            
            if researchers:
                table = format_researcher_table(researchers, f"Researchers: {query}")
                console.print(table)
            else:
                console.print("[yellow]No researchers found.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if debug_mode:
            console.print("\n[yellow]Full traceback:[/yellow]")
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "researcher search")
        raise typer.Exit(1)


@researchers_app.command("get")
def get_researcher(
    researcher_id: str = typer.Argument(..., help="Researcher ID"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save result to JSON file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Get a specific researcher by ID"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = get_client()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching researcher...", total=None)
            researcher = client.get_researcher_sync(researcher_id)
            progress.update(task, completed=100)
        
        if json_file:
            # Save to JSON file
            save_json_to_file([asdict(researcher)], json_file)
        else:
            # Display table format to stdout
            table = format_researcher_table([researcher], f"Researcher Details: {researcher.full_name}")
            console.print(table)
            
    except RLANotFoundError:
        console.print(f"[red]Researcher with ID '{researcher_id}' not found.[/red]")
        raise typer.Exit(1)
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "retrieving researcher")
        raise typer.Exit(1)


@grants_app.command("search")
def search_grants(
    query: str = typer.Argument("", help="Search query for grants (empty returns all grants)"),
    page_size: int = typer.Option(10, "--size", "-s", help="Number of results per page"),
    page_number: int = typer.Option(1, "--page", "-p", help="Page number"),
    filter_query: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter query (e.g., 'funder:NHMRC,status:Active')"),
    advanced_query: Optional[str] = typer.Option(None, "--advanced", "-a", help="Advanced search query"),
    min_amount: Optional[float] = typer.Option(None, "--min-amount", help="Minimum funding amount"),
    active: bool = typer.Option(False, "--active", help="Show only active grants"),
    all_results: bool = typer.Option(False, "--all", help="Fetch all results from all pages"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    show_stats: bool = typer.Option(False, "--stats", help="Show funding statistics"),
    debug: bool = typer.Option(False, "--debug", help="Show full traceback on errors")
):
    """Search for grants in the RLA database"""
    global debug_mode
    debug_mode = debug
    try:
        client = get_client()
        
        # Add active filter if requested
        if active and filter_query:
            filter_query += ",status:Active"
        elif active:
            filter_query = "status:Active"
        
        if all_results:
            # Fetch all results from all pages using async for better performance
            console.print("[yellow]Fetching all grant results using concurrent requests...[/yellow]")
            
            async def fetch_all_async():
                async with client:  # Use async context manager
                    return await client.search_all_grants(
                        value=query,
                        filter_query=filter_query,
                        advanced_search_query=advanced_query,
                        max_concurrent=5  # Limit concurrent requests to be API-friendly
                    )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching all grants concurrently...", total=None)
                response = run_async(fetch_all_async())
                progress.update(task, completed=100, description=f"Fetched {len(response.results)} grants")
            
            grants = response.results
        else:
            # Single page fetch
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching grants...", total=None)
                response = client.search_grants_sync(
                    value=query,
                    filter_query=filter_query,
                    page_number=page_number,
                    page_size=page_size,
                    advanced_search_query=advanced_query
                )
                progress.update(task, completed=100)
            
            grants = response.results
        if min_amount:
            grants = filter_grants_by_amount(grants, min_amount=min_amount)
        
        if show_stats and grants:
            stats = get_funding_statistics(grants)
            stats_panel = Panel(
                f"Total Funding: ${stats['total_funding']:,.2f}\n"
                f"Average Funding: ${stats['average_funding']:,.2f}\n"
                f"Median Funding: ${stats['median_funding']:,.2f}\n"
                f"Number of Grants: {stats['grant_count']}",
                title="Funding Statistics",
                border_style="green"
            )
            console.print(stats_panel)
        
        if json_file:
            # Save to JSON file
            data = [asdict(g) for g in grants]
            save_json_to_file(data, json_file)
        else:
            # Display table format to stdout
            if all_results:
                total_pages = 1  # All results in one "page"
                info_panel = Panel(
                    f"Found {response.total_results} grants total\n"
                    f"Showing all results\n"
                    f"Results displayed: {len(grants)}",
                    title="Search Results (All Pages)",
                    border_style="blue"
                )
            else:
                total_pages = (response.total_results + response.size - 1) // response.size  # Calculate total pages
                info_panel = Panel(
                    f"Found {response.total_results} grants total\n"
                    f"Showing page {response.current_page} of {total_pages}\n"
                    f"Results per page: {page_size}",
                    title="Search Results",
                    border_style="blue"
                )
            console.print(info_panel)
            
            if grants:
                table = format_grant_table(grants, f"Grants: {query}")
                console.print(table)
            else:
                console.print("[yellow]No grants found.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@grants_app.command("get")
def get_grant(
    grant_id: str = typer.Argument(..., help="Grant ID"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save result to JSON file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Get a specific grant by ID"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = RLAClient(debug=debug)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching grant...", total=None)
            grant = client.get_grant_sync(grant_id)
            progress.update(task, completed=100)
        
        if json_file:
            # Save to JSON file
            save_json_to_file([asdict(grant)], json_file)
        else:
            # Display table format to stdout
            table = format_grant_table([grant], f"Grant Details: {grant.title or grant.grant_title}")
            console.print(table)
            
    except Exception as e:
        handle_exception(e, debug_mode)


@organisations_app.command("search")
def search_organisations(
    query: str = typer.Argument(..., help="Search query for organisations"),
    page_size: int = typer.Option(10, "--size", "-s", help="Number of results per page"),
    page_number: int = typer.Option(1, "--page", "-p", help="Page number"),
    filter_query: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter query"),
    advanced_query: Optional[str] = typer.Option(None, "--advanced", "-a", help="Advanced search query"),
    all_results: bool = typer.Option(False, "--all", help="Fetch all results from all pages"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file")
):
    """Search for organisations in the RLA database"""
    try:
        client = get_client()
        
        if all_results:
            # Fetch all results from all pages using async for better performance
            console.print("[yellow]Fetching all organisation results using concurrent requests...[/yellow]")
            
            async def fetch_all_async():
                async with client:  # Use async context manager
                    return await client.search_all_organisations(
                        value=query,
                        filter_query=filter_query,
                        advanced_search_query=advanced_query,
                        max_concurrent=5  # Limit concurrent requests to be API-friendly
                    )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching all organisations concurrently...", total=None)
                response = run_async(fetch_all_async())
                progress.update(task, completed=100, description=f"Fetched {len(response.results)} organisations")
            
            orgs = response.results
        else:
            # Single page fetch
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching organisations...", total=None)
                response = client.search_organisations_sync(
                    value=query,
                    filter_query=filter_query,
                    page_number=page_number,
                    page_size=page_size,
                    advanced_search_query=advanced_query
                )
                progress.update(task, completed=100)
            
            orgs = response.results
        
        if json_file:
            save_json_to_file([asdict(org) for org in orgs], json_file)
        else:
            # Table format (default)
            if all_results:
                total_pages = 1  # All results in one "page"
                info_panel = Panel(
                    f"Found {response.total_results} organisations total\n"
                    f"Showing all results\n"
                    f"Results displayed: {len(orgs)}",
                    title="Search Results (All Pages)",
                    border_style="blue"
                )
            else:
                total_pages = (response.total_results + response.size - 1) // response.size  # Calculate total pages
                info_panel = Panel(
                    f"Found {response.total_results} organisations total\n"
                    f"Showing page {response.current_page} of {total_pages}\n"
                    f"Results per page: {page_size}",
                    title="Search Results",
                    border_style="blue"
                )
            console.print(info_panel)
            
            if orgs:
                table = format_organisation_table(orgs, f"Organisations: {query}")
                console.print(table)
            else:
                console.print("[yellow]No organisations found.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@organisations_app.command("get")
def get_organisation(
    organisation_id: str = typer.Argument(..., help="Organisation ID"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save result to JSON file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Get a specific organisation by ID"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = RLAClient(debug=debug)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching organisation...", total=None)
            org = client.get_organisation_sync(organisation_id)
            progress.update(task, completed=100)
        
        if json_file:
            save_json_to_file(asdict(org), json_file)
        else:
            table = format_organisation_table([org], f"Organisation Details: {org.name or org.organisation_name}")
            console.print(table)
            
    except Exception as e:
        handle_exception(e, debug_mode)


@grants_app.command("by-funder")
def grants_by_funder(
    funder: str = typer.Argument(..., help="Funder name (e.g., 'Australian Research Council')"),
    status: Optional[str] = typer.Option(None, "--status", help="Grant status filter"),
    page_size: int = typer.Option(10, "--size", "-s", help="Number of results per page"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    show_stats: bool = typer.Option(False, "--stats", help="Show funding statistics")
):
    """Search for grants by funding agency"""
    try:
        client = get_client()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Searching grants by {funder}...", total=None)
            response = client.run_async(client.search_grants_by_funder(funder, status=status or "", page_size=page_size))
            progress.update(task, completed=100)
        
        grants = response.results
        
        if show_stats and grants:
            stats = get_funding_statistics(grants)
            stats_panel = Panel(
                f"Total Funding: ${stats['total_funding']:,.2f}\n"
                f"Average Funding: ${stats['average_funding']:,.2f}\n"
                f"Median Funding: ${stats['median_funding']:,.2f}\n"
                f"Number of Grants: {stats['grant_count']}",
                title=f"Funding Statistics - {funder}",
                border_style="green"
            )
            console.print(stats_panel)
        
        if json_file:
            save_json_to_file([asdict(g) for g in grants], json_file)
        else:
            total_pages = (response.total_results + response.size - 1) // response.size  # Calculate total pages
            info_panel = Panel(
                f"Found {response.total_results} grants from {funder}\n"
                f"Showing page {response.current_page} of {total_pages}",
                title="Search Results",
                border_style="blue"
            )
            console.print(info_panel)
            
            if grants:
                table = format_grant_table(grants, f"Grants from {funder}")
                console.print(table)
            else:
                console.print(f"[yellow]No grants found from {funder}.[/yellow]")
                
    except RLAError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Publication Commands

@publications_app.command("search")
def search_publications(
    query: str = typer.Argument(..., help="Search query for publications"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    size: int = typer.Option(10, "--size", "-s", help="Number of results to return"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    filter: str = typer.Option("", "--filter", help="Filter query (e.g., 'publicationType:journal article,publicationYear:2023')"),
    advanced: str = typer.Option("", "--advanced", help="Advanced search query"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Filter by publication year"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by publication type"),
    doi_only: bool = typer.Option(False, "--doi-only", help="Show only publications with DOI"),
    stats: bool = typer.Option(False, "--stats", help="Show publication statistics"),
    async_search: bool = typer.Option(False, "--async", help="Use async search for better performance (recommended for larger searches)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Search for publications"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = get_client()
        
        # Build filter query
        filter_parts = []
        if filter:
            filter_parts.append(filter)
        if year:
            filter_parts.append(f"publicationYear:{year}")
        if type:
            filter_parts.append(f"publicationType:{type}")
        
        combined_filter = ",".join(filter_parts)
        
        if async_search:
            # Use async search for better performance
            console.print("[yellow]Using async search for better performance...[/yellow]")
            
            async def search_async():
                async with client:
                    return await client.search_publications(
                        value=query,
                        filter_query=combined_filter,
                        page_number=page,
                        page_size=size,
                        advanced_search_query=advanced,
                        max_concurrent=10  # Higher concurrency for publications
                    )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching publications concurrently...", total=None)
                response = run_async(search_async())
                progress.update(task, completed=100, description=f"Found {len(response.results)} publications")
        else:
            # Use traditional sync search
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching publications...", total=None)
                response = client.search_publications_sync(
                    value=query,
                    filter_query=combined_filter,
                    page_number=page,
                    page_size=size,
                    advanced_search_query=advanced
                )
                progress.update(task, completed=100)
        
        publications = response.results
        
        # Apply additional filtering
        if doi_only:
            from .utils import filter_publications_by_doi
            publications = filter_publications_by_doi(publications)
        
        if json_file:
            # Save to JSON file
            data = [asdict(pub) for pub in publications]
            save_json_to_file(data, json_file)
        else:
            # Display as table
            table = format_publication_table(publications, f"Publications: {query}")
            console.print(table)
            
            console.print(f"\n[dim]Showing {len(publications)} of {response.total_results} total results[/dim]")
            
            if stats and publications:
                from .utils import get_publication_statistics
                pub_stats = get_publication_statistics(publications)
                stats_table = Table(title="Publication Statistics", show_header=True)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                
                stats_table.add_row("Total Publications", str(pub_stats['total_publications']))
                stats_table.add_row("Publications with DOI", f"{pub_stats['publications_with_doi']} ({pub_stats['doi_percentage']:.1f}%)")
                stats_table.add_row("Unique Publishers", str(pub_stats['unique_publishers']))
                stats_table.add_row("Unique Journals", str(pub_stats['unique_journals']))
                if pub_stats['year_range']:
                    stats_table.add_row("Year Range", f"{pub_stats['year_range'][0]} - {pub_stats['year_range'][1]}")
                
                console.print("\n")
                console.print(stats_table)
                
                # Show publication types breakdown
                if pub_stats['publication_types']:
                    type_table = Table(title="Publication Types", show_header=True)
                    type_table.add_column("Type", style="cyan")
                    type_table.add_column("Count", style="green")
                    
                    for pub_type, count in sorted(pub_stats['publication_types'].items(), key=lambda x: x[1], reverse=True):
                        type_table.add_row(pub_type, str(count))
                    
                    console.print("\n")
                    console.print(type_table)
                    
    except Exception as e:
        handle_exception(e, "searching publications")
        raise typer.Exit(1)


@publications_app.command("get")
def get_publication(
    publication_id: str = typer.Argument(..., help="Publication ID to retrieve"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save result to JSON file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Get a specific publication by ID"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = get_client()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving publication...", total=None)
            publication = client.get_publication_sync(publication_id)
            progress.update(task, completed=100)
        
        if json_file:
            # Save to JSON file
            data = asdict(publication)
            save_json_to_file([data], json_file)
        else:
            # Display as table
            table = format_publication_table([publication], f"Publication Details: {publication.title}")
            console.print(table)
            
            # Show additional details
            if publication.abstract:
                abstract_panel = Panel(
                    publication.abstract,
                    title="Abstract",
                    border_style="blue"
                )
                console.print("\n")
                console.print(abstract_panel)
                
    except RLANotFoundError:
        console.print(f"[red]Publication with ID '{publication_id}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        handle_exception(e, "retrieving publication")
        raise typer.Exit(1)


@publications_app.command("by-doi")
def get_publication_by_doi(
    doi: str = typer.Argument(..., help="DOI of the publication"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Search for publications by DOI"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = get_client()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching by DOI...", total=None)
            # Search for publications using DOI in filter query
            response = client.search_publications_sync(
                filter_query=f"doi:{doi}",
                page_size=100
            )
            progress.update(task, completed=100)
        
        if not response.results:
            console.print(f"[yellow]No publications found with DOI: {doi}[/yellow]")
            return
            
        publications = response.results
        
        if json_file:
            # Save to JSON file
            data = [asdict(pub) for pub in publications]
            save_json_to_file(data, json_file)
        else:
            # Display as table
            table = format_publication_table(publications, f"Publications by DOI: {doi}")
            console.print(table)
            
    except Exception as e:
        handle_exception(e, "searching publications by DOI")
        raise typer.Exit(1)


@publications_app.command("by-year")
def search_publications_by_year(
    year: int = typer.Argument(..., help="Publication year"),
    end_year: Optional[int] = typer.Option(None, "--end-year", help="End year for range search"),
    size: int = typer.Option(10, "--size", "-s", help="Number of results to return"),
    json_file: Optional[str] = typer.Option(None, "--json", help="Save results to JSON file"),
    stats: bool = typer.Option(False, "--stats", help="Show publication statistics"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Search for publications by year or year range"""
    global debug_mode
    debug_mode = debug
    
    try:
        client = get_client()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching publications by year...", total=None)
            # Search for publications using year in filter query
            if end_year and end_year != year:
                year_filter = f"publicationYear:{year}-{end_year}"
            else:
                year_filter = f"publicationYear:{year}"
            
            response = client.search_publications_sync(
                filter_query=year_filter,
                page_size=size
            )
            progress.update(task, completed=100)
        
        publications = response.results
        year_range = f"{year}-{end_year}" if end_year and end_year != year else str(year)
        
        if json_file:
            # Save to JSON file
            data = [asdict(pub) for pub in publications]
            save_json_to_file(data, json_file)
        else:
            # Display as table
            table = format_publication_table(publications, f"Publications: {year_range}")
            console.print(table)
            console.print(f"\n[dim]Showing {len(publications)} of {response.total_results} total results[/dim]")
            
            if stats and publications:
                from .utils import get_publication_statistics
                pub_stats = get_publication_statistics(publications)
                
                stats_table = Table(title="Publication Statistics", show_header=True)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                
                stats_table.add_row("Total Publications", str(pub_stats['total_publications']))
                stats_table.add_row("Publications with DOI", f"{pub_stats['publications_with_doi']} ({pub_stats['doi_percentage']:.1f}%)")
                stats_table.add_row("Unique Publishers", str(pub_stats['unique_publishers']))
                
                console.print("\n")
                console.print(stats_table)
                
    except Exception as e:
        handle_exception(e, "searching publications by year")


@app.command()
def version():
    """Show PyRLA version information"""
    try:
        from . import __version__
        version_info = __version__
    except ImportError:
        version_info = "unknown"
    
    console.print(f"[bold blue]PyRLA[/bold blue] version [bold green]{version_info}[/bold green]")
    console.print("Python interface for Research Link Australia API")
    console.print("API Documentation: https://researchlink.ardc.edu.au/v3/api-docs/public")


def main():
    """Main entry point for the CLI"""
    # Load environment variables from .env file in current working directory
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
