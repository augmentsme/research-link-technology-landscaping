"""
Utility functions for PyRLA
"""

from typing import Dict, List, Any, Optional
from .models import Researcher, Grant, Organisation, Publication, SearchResponse


def filter_researchers_by_orcid(researchers: List[Researcher]) -> List[Researcher]:
    """
    Filter researchers that have ORCID identifiers
    
    Args:
        researchers: List of Researcher objects
        
    Returns:
        List of researchers with ORCID IDs
    """
    return [r for r in researchers if r.orcid]


def filter_grants_by_amount(grants: List[Grant], min_amount: Optional[float] = None, max_amount: Optional[float] = None) -> List[Grant]:
    """
    Filter grants by funding amount range
    
    Args:
        grants: List of Grant objects
        min_amount: Minimum funding amount (inclusive)
        max_amount: Maximum funding amount (inclusive)
        
    Returns:
        List of grants within the specified amount range
    """
    filtered = []
    for grant in grants:
        if grant.funding_amount is None:
            continue
            
        if min_amount is not None and grant.funding_amount < min_amount:
            continue
            
        if max_amount is not None and grant.funding_amount > max_amount:
            continue
            
        filtered.append(grant)
    
    return filtered


def extract_unique_funders(grants: List[Grant]) -> List[str]:
    """
    Extract unique funding agencies from a list of grants
    
    Args:
        grants: List of Grant objects
        
    Returns:
        Sorted list of unique funder names
    """
    funders = {grant.funder for grant in grants if grant.funder}
    return sorted(list(funders))


def extract_unique_organisations(researchers: List[Researcher]) -> List[str]:
    """
    Extract unique current organisation names from researchers
    
    Args:
        researchers: List of Researcher objects
        
    Returns:
        Sorted list of unique organisation names
    """
    orgs = set()
    for researcher in researchers:
        orgs.update(researcher.current_organisation_name)
    return sorted(list(orgs))


def group_grants_by_status(grants: List[Grant]) -> Dict[str, List[Grant]]:
    """
    Group grants by their status
    
    Args:
        grants: List of Grant objects
        
    Returns:
        Dictionary with status as key and list of grants as value
    """
    grouped = {}
    for grant in grants:
        status = grant.status or "Unknown"
        if status not in grouped:
            grouped[status] = []
        grouped[status].append(grant)
    return grouped


def calculate_total_funding(grants: List[Grant]) -> float:
    """
    Calculate total funding amount from a list of grants
    
    Args:
        grants: List of Grant objects
        
    Returns:
        Total funding amount (excludes grants without funding amount)
    """
    total = 0.0
    for grant in grants:
        if grant.funding_amount is not None:
            total += grant.funding_amount
    return total


def get_funding_statistics(grants: List[Grant]) -> Dict[str, Any]:
    """
    Get funding statistics from a list of grants
    
    Args:
        grants: List of Grant objects
        
    Returns:
        Dictionary with funding statistics
    """
    amounts = [g.funding_amount for g in grants if g.funding_amount is not None]
    
    if not amounts:
        return {
            "grant_count": len(grants),
            "total_grants": len(grants),
            "grants_with_funding": 0,
            "total_funding": 0.0,
            "average_funding": 0.0,
            "median_funding": 0.0,
            "min_funding": 0.0,
            "max_funding": 0.0
        }
    
    # Calculate median
    sorted_amounts = sorted(amounts)
    n = len(sorted_amounts)
    if n % 2 == 0:
        median = (sorted_amounts[n//2 - 1] + sorted_amounts[n//2]) / 2
    else:
        median = sorted_amounts[n//2]
    
    return {
        "grant_count": len(grants),
        "total_grants": len(grants),
        "grants_with_funding": len(amounts),
        "total_funding": sum(amounts),
        "average_funding": sum(amounts) / len(amounts),
        "median_funding": median,
        "min_funding": min(amounts),
        "max_funding": max(amounts)
    }


def search_response_to_list(response: SearchResponse) -> List[Any]:
    """
    Extract the results list from a SearchResponse
    
    Args:
        response: SearchResponse object
        
    Returns:
        List of results (Researcher, Grant, or Organisation objects)
    """
    return response.results


def paginate_all_results(client, search_method, page_size: int = 250, max_pages: int = None, **search_kwargs):
    """
    Helper function to paginate through all search results
    
    Args:
        client: RLAClient instance
        search_method: The search method to call (e.g., client.search_grants)
        page_size: Number of results per page (max 250)
        max_pages: Maximum number of pages to fetch (None for all)
        **search_kwargs: Additional keyword arguments for the search method
        
    Yields:
        Individual results from all pages
    """
    page = 1
    
    while True:
        if max_pages and page > max_pages:
            break
            
        try:
            response = search_method(
                page_number=page,
                page_size=page_size,
                **search_kwargs
            )
            
            # If no results, we're done
            if not response.results:
                break
                
            # Yield each result
            for result in response.results:
                yield result
            
            # If we got fewer results than page_size, we're on the last page
            if len(response.results) < page_size:
                break
                
            page += 1
            
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break


def create_researcher_summary(researcher: Researcher) -> Dict[str, Any]:
    """
    Create a summary dictionary for a researcher
    
    Args:
        researcher: Researcher object
        
    Returns:
        Dictionary with researcher summary information
    """
    return {
        "id": researcher.id,
        "name": researcher.full_name,
        "first_name": researcher.first_name,
        "last_name": researcher.last_name,
        "orcid": researcher.orcid,
        "scopus_id": researcher.scopus_author_id,
        "current_organisations": researcher.current_organisation_name,
        "publication_count": researcher.summary.publication_count if researcher.summary else 0,
        "grant_count": researcher.summary.grant_count if researcher.summary else 0,
        "countries": researcher.countries,
        "states": researcher.states
    }


def create_grant_summary(grant: Grant) -> Dict[str, Any]:
    """
    Create a summary dictionary for a grant
    
    Args:
        grant: Grant object
        
    Returns:
        Dictionary with grant summary information
    """
    return {
        "id": grant.id,
        "title": grant.grant_title or grant.title,
        "funder": grant.funder,
        "funding_scheme": grant.funding_scheme,
        "funding_amount": grant.funding_amount,
        "status": grant.status,
        "start_date": grant.start_date.isoformat() if grant.start_date else None,
        "end_date": grant.end_date.isoformat() if grant.end_date else None,
        "admin_organisation": grant.admin_organisation_name,
        "primary_subject": grant.primary_subject,
        "countries": grant.countries,
        "states": grant.states
    }


# Publication utility functions

def filter_publications_by_year(publications: List[Publication], start_year: int, end_year: Optional[int] = None) -> List[Publication]:
    """
    Filter publications by publication year range
    
    Args:
        publications: List of Publication objects
        start_year: Starting year (inclusive)
        end_year: Ending year (inclusive). If None, uses start_year
        
    Returns:
        List of publications within the specified year range
    """
    if end_year is None:
        end_year = start_year
        
    filtered = []
    for pub in publications:
        if pub.publication_year and start_year <= pub.publication_year <= end_year:
            filtered.append(pub)
    
    return filtered


def filter_publications_by_type(publications: List[Publication], publication_type: str) -> List[Publication]:
    """
    Filter publications by publication type
    
    Args:
        publications: List of Publication objects
        publication_type: Type to filter by (e.g., "journal article", "conference paper")
        
    Returns:
        List of publications of the specified type
    """
    return [p for p in publications if p.publication_type.lower() == publication_type.lower()]


def filter_publications_by_doi(publications: List[Publication]) -> List[Publication]:
    """
    Filter publications that have DOI identifiers
    
    Args:
        publications: List of Publication objects
        
    Returns:
        List of publications with DOI identifiers
    """
    return [p for p in publications if p.doi]


def extract_unique_publishers(publications: List[Publication]) -> List[str]:
    """
    Extract unique publishers from a list of publications
    
    Args:
        publications: List of Publication objects
        
    Returns:
        Sorted list of unique publisher names
    """
    publishers = {pub.publisher for pub in publications if pub.publisher}
    return sorted(list(publishers))


def extract_unique_journals(publications: List[Publication]) -> List[str]:
    """
    Extract unique journals from a list of publications
    
    Args:
        publications: List of Publication objects
        
    Returns:
        Sorted list of unique journal names
    """
    journals = {pub.journal for pub in publications if pub.journal}
    return sorted(list(journals))


def group_publications_by_year(publications: List[Publication]) -> Dict[int, List[Publication]]:
    """
    Group publications by publication year
    
    Args:
        publications: List of Publication objects
        
    Returns:
        Dictionary with year as key and list of publications as value
    """
    grouped = {}
    for pub in publications:
        if pub.publication_year:
            year = pub.publication_year
            if year not in grouped:
                grouped[year] = []
            grouped[year].append(pub)
    return grouped


def group_publications_by_type(publications: List[Publication]) -> Dict[str, List[Publication]]:
    """
    Group publications by publication type
    
    Args:
        publications: List of Publication objects
        
    Returns:
        Dictionary with publication type as key and list of publications as value
    """
    grouped = {}
    for pub in publications:
        pub_type = pub.publication_type or "Unknown"
        if pub_type not in grouped:
            grouped[pub_type] = []
        grouped[pub_type].append(pub)
    return grouped


def get_publication_statistics(publications: List[Publication]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of publications
    
    Args:
        publications: List of Publication objects
        
    Returns:
        Dictionary with publication statistics
    """
    if not publications:
        return {
            "total_publications": 0,
            "publications_with_doi": 0,
            "unique_publishers": 0,
            "unique_journals": 0,
            "year_range": None,
            "publication_types": {}
        }
    
    # Count publications with DOI
    pubs_with_doi = len([p for p in publications if p.doi])
    
    # Get unique publishers and journals
    publishers = extract_unique_publishers(publications)
    journals = extract_unique_journals(publications)
    
    # Calculate year range
    years = [p.publication_year for p in publications if p.publication_year]
    year_range = (min(years), max(years)) if years else None
    
    # Count by publication type
    type_counts = {}
    for pub in publications:
        pub_type = pub.publication_type or "Unknown"
        type_counts[pub_type] = type_counts.get(pub_type, 0) + 1
    
    return {
        "total_publications": len(publications),
        "publications_with_doi": pubs_with_doi,
        "doi_percentage": (pubs_with_doi / len(publications)) * 100 if publications else 0,
        "unique_publishers": len(publishers),
        "unique_journals": len(journals),
        "year_range": year_range,
        "publication_types": type_counts
    }


def create_publication_summary(publication: Publication) -> Dict[str, Any]:
    """
    Create a summary dictionary from a Publication object
    
    Args:
        publication: Publication object
        
    Returns:
        Dictionary with publication summary information
    """
    return {
        "id": publication.id,
        "title": publication.title,
        "authors": publication.authors_list,
        "publication_type": publication.publication_type,
        "journal": publication.journal,
        "publisher": publication.publisher,
        "publication_year": publication.publication_year,
        "doi": publication.doi,
        "abstract": publication.abstract[:200] + "..." if len(publication.abstract) > 200 else publication.abstract,
        "keywords": publication.keywords,
        "countries": publication.countries,
        "url": publication.url
    }
