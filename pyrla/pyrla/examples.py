"""
Example usage of PyRLA - Python interface for Research Link Australia API

This module contains example functions demonstrating how to use the RLA API client
to search for researchers, grants, and organisations.
"""

import os
import asyncio
from typing import List, Dict, Any
from pyrla import RLAClient
from pyrla.models import Researcher, Grant, Organisation
from pyrla.utils import (
    filter_researchers_by_orcid,
    filter_grants_by_amount,
    get_funding_statistics,
    paginate_all_results,
    create_researcher_summary,
    create_grant_summary
)


def basic_researcher_search_example():
    """
    Example: Basic researcher search using synchronous wrapper
    """
    # Initialize client (will use RLA_API_TOKEN environment variable)
    client = RLAClient()
    
    # Test connection
    if not client.test_connection():
        print("Failed to connect to API")
        return
    
    print("=== Basic Researcher Search Example ===")
    
    # Search for researchers with name "Smith" using sync wrapper
    response = client.search_researchers_sync(value="Smith", page_size=5)
    
    print(f"Found {response.total_results} researchers total")
    print(f"Showing {len(response.results)} results from page {response.current_page}")
    
    for researcher in response.results:
        print(f"- {researcher.full_name} (ID: {researcher.id})")
        if researcher.current_organisation_name:
            print(f"  Organisation: {', '.join(researcher.current_organisation_name)}")
        if researcher.orcid:
            print(f"  ORCID: {researcher.orcid}")
        print()


async def async_researcher_search_example():
    """
    Example: Async researcher search for better performance
    """
    # Initialize client (will use RLA_API_TOKEN environment variable)
    async with RLAClient() as client:
        print("=== Async Researcher Search Example ===")
        
        # Search for researchers with name "Smith"
        response = await client.search_researchers(value="Smith", page_size=5)
        
        print(f"Found {response.total_results} researchers total")
        print(f"Showing {len(response.results)} results from page {response.current_page}")
        
        for researcher in response.results:
            print(f"- {researcher.full_name} (ID: {researcher.id})")
            if researcher.current_organisation_name:
                print(f"  Organisation: {', '.join(researcher.current_organisation_name)}")
            if researcher.orcid:
                print(f"  ORCID: {researcher.orcid}")
            print()


def advanced_researcher_search_example():
    """
    Example: Advanced researcher search with filters
    """
    client = RLAClient()
    
    print("=== Advanced Researcher Search Example ===")
    
    # Search for researchers at Australian National University in specific subject areas
    response = client.search_researchers_sync(
        filter_query="currentOrganisationName:Australian National University,summary.forSubjectCodes:31",
        advanced_search_query="research-topic:Biological Sciences",
        page_size=10
    )
    
    print(f"Found {response.total_results} researchers at ANU in Biological Sciences")
    
    # Filter to only those with ORCID
    researchers_with_orcid = filter_researchers_by_orcid(response.results)
    print(f"Of these, {len(researchers_with_orcid)} have ORCID identifiers")
    
    for researcher in researchers_with_orcid[:3]:  # Show first 3
        summary = create_researcher_summary(researcher)
        print(f"- {summary['name']}")
        print(f"  ORCID: {summary['orcid']}")
        print(f"  Publications: {summary['publication_count']}")
        print(f"  Grants: {summary['grant_count']}")
        print()


def grants_search_example():
    """
    Example: Search for grants by funder and analyze funding
    """
    client = RLAClient()
    
    print("=== Grants Search Example ===")
    
    # Search for active grants from Australian Research Council
    response = client.search_grants_by_funder(
        funder="Australian Research Council",
        status="Active",
        page_size=20
    )
    
    print(f"Found {response.total_results} active ARC grants")
    
    # Filter grants by funding amount (above $100,000)
    high_value_grants = filter_grants_by_amount(response.results, min_amount=100000)
    print(f"Of these, {len(high_value_grants)} are above $100,000")
    
    # Get funding statistics
    stats = get_funding_statistics(high_value_grants)
    print(f"\nFunding Statistics for high-value grants:")
    print(f"- Total funding: ${stats['total_funding']:,.2f}")
    print(f"- Average funding: ${stats['average_funding']:,.2f}")
    print(f"- Min funding: ${stats['min_funding']:,.2f}")
    print(f"- Max funding: ${stats['max_funding']:,.2f}")
    
    # Show top 3 grants by funding amount
    sorted_grants = sorted(high_value_grants, key=lambda g: g.funding_amount or 0, reverse=True)
    print(f"\nTop 3 highest funded grants:")
    for i, grant in enumerate(sorted_grants[:3], 1):
        print(f"{i}. {grant.grant_title or grant.title}")
        print(f"   Funding: ${grant.funding_amount:,.2f}")
        print(f"   Organisation: {grant.admin_organisation_name}")
        print()


def organisation_search_example():
    """
    Example: Search for organisations and explore their research areas
    """
    client = RLAClient()
    
    print("=== Organisation Search Example ===")
    
    # Search for universities in NSW
    response = client.search_organisations(
        filter_query="stateCodes:nsw",
        advanced_search_query="type:abr",  # Australian Business Register entities
        page_size=10
    )
    
    print(f"Found {response.total_results} organisations in NSW")
    
    for org in response.results[:5]:  # Show first 5
        print(f"- {org.organisation_name or org.name}")
        print(f"  Type: {org.abr_type}")
        print(f"  ABN: {org.abn}")
        if org.summary:
            print(f"  Researchers: {org.summary.researcher_count}")
            print(f"  Grants: {org.summary.grant_count}")
        print()


def get_specific_researcher_example():
    """
    Example: Get specific researcher by ID and explore their details
    """
    client = RLAClient()
    
    print("=== Get Specific Researcher Example ===")
    
    # First, search for a researcher to get an ID
    response = client.search_researchers(value="machine learning", page_size=1)
    
    if not response.results:
        print("No researchers found for this search")
        return
    
    researcher_id = response.results[0].id
    print(f"Getting details for researcher ID: {researcher_id}")
    
    # Get full researcher details
    researcher = client.get_researcher(researcher_id)
    
    print(f"\nResearcher: {researcher.full_name}")
    print(f"ORCID: {researcher.orcid}")
    print(f"Current organisations: {', '.join(researcher.current_organisation_name)}")
    
    if researcher.summary:
        print(f"\nSummary:")
        print(f"- Publications: {researcher.summary.publication_count}")
        print(f"- Grants: {researcher.summary.grant_count}")
        print(f"- Total funding: ${researcher.summary.total_funding or 0:,.2f}")
    
    if researcher.publications:
        print(f"\nRecent publications ({len(researcher.publications)}):")
        for pub in researcher.publications[:3]:  # Show first 3
            print(f"- {pub.title}")
            if pub.publication_date:
                print(f"  Date: {pub.publication_date.year}")
            print()


def pagination_example():
    """
    Example: How to paginate through all results
    """
    client = RLAClient()
    
    print("=== Pagination Example ===")
    
    # Use the utility function to get all grants from a specific funder
    print("Fetching all grants from NHMRC (limited to first 100)...")
    
    all_grants = []
    count = 0
    
    for grant in paginate_all_results(
        client, 
        client.search_grants,
        page_size=50,
        max_pages=2,  # Limit to first 2 pages for example
        filter_query="funder:National Health and Medical Research Council"
    ):
        all_grants.append(grant)
        count += 1
        if count >= 100:  # Limit for example
            break
    
    print(f"Retrieved {len(all_grants)} NHMRC grants")
    
    # Analyze by status
    status_counts = {}
    for grant in all_grants:
        status = grant.status or "Unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nGrants by status:")
    for status, count in sorted(status_counts.items()):
        print(f"- {status}: {count}")


def search_by_research_topic_example():
    """
    Example: Search for researchers and grants in a specific research topic
    """
    client = RLAClient()
    
    print("=== Search by Research Topic Example ===")
    
    topic = "artificial intelligence"
    
    # Search researchers
    researcher_response = client.search_researchers(
        value=topic,
        page_size=5
    )
    
    print(f"Found {researcher_response.total_results} researchers related to '{topic}':")
    for researcher in researcher_response.results:
        print(f"- {researcher.full_name}")
        print(f"  Organisations: {', '.join(researcher.current_organisation_name)}")
        print()
    
    # Search grants
    grant_response = client.search_grants(
        value=topic,
        page_size=5
    )
    
    print(f"Found {grant_response.total_results} grants related to '{topic}':")
    for grant in grant_response.results:
        summary = create_grant_summary(grant)
        print(f"- {summary['title']}")
        print(f"  Funder: {summary['funder']}")
        if summary['funding_amount']:
            print(f"  Amount: ${summary['funding_amount']:,.2f}")
        print()


def main():
    """
    Run all examples
    """
    try:
        basic_researcher_search_example()
        print("\n" + "="*60 + "\n")
        
        advanced_researcher_search_example()
        print("\n" + "="*60 + "\n")
        
        grants_search_example()
        print("\n" + "="*60 + "\n")
        
        organisation_search_example()
        print("\n" + "="*60 + "\n")
        
        get_specific_researcher_example()
        print("\n" + "="*60 + "\n")
        
        pagination_example()
        print("\n" + "="*60 + "\n")
        
        search_by_research_topic_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set the RLA_API_TOKEN environment variable")


if __name__ == "__main__":
    main()
