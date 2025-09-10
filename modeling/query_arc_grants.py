import json
from neo4j import GraphDatabase
from pathlib import Path
import config
import utils
from dotenv import dotenv_values


def query_grants_from_organisations(uri, user, password):
    """
    Connects to Neo4j and executes a query to find all grants
    from Australian organisations linked to researchers.
    
    Returns:
        list: List of dictionaries containing organisation, researcher, and grant data
    """
    query = """
    MATCH (o:organisation {country: 'AU'})--(r:orcid:researcher)--(g:orcid:grant)
    RETURN o, r, g
    """
    
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            
            records, summary, keys = driver.execute_query(query, database_="neo4j")
            
            results = []
            for record in records:
                org_node = record["o"]
                researcher_node = record["r"]
                grant_node = record["g"]
                
                result = {
                    "organisation": dict(org_node),
                    "researcher": dict(researcher_node),
                    "grant": dict(grant_node)
                }
                results.append(result)
            
            return results
            
    except Exception as e:
        print(f"Error executing Neo4j query: {e}")
        return None



def save_separated_data(results, output_dir):
    """
    Saves the results to 3 separate JSONL files: organisations, researchers, and grants.
    """
    if not results:
        print("No data to save")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract unique organisations, researchers, and grants
    organisations = {}
    researchers = {}
    grants = {}
    
    for result in results:
        org = result['organisation']
        researcher = result['researcher']
        grant = result['grant']
        
        # Use appropriate ID fields as keys to avoid duplicates
        if 'id' in org:
            organisations[org['id']] = org
        elif 'name' in org:
            organisations[org['name']] = org
            
        if 'orcid' in researcher:
            researchers[researcher['orcid']] = researcher
        elif 'id' in researcher:
            researchers[researcher['id']] = researcher
            
        if 'id' in grant:
            grants[grant['id']] = grant
        elif 'title' in grant:
            grants[grant['title']] = grant
    
    # Save to separate JSONL files using utils function
    utils.save_jsonl_file(list(organisations.values()), output_dir / "organisations.jsonl")
    utils.save_jsonl_file(list(researchers.values()), output_dir / "researchers.jsonl")
    utils.save_jsonl_file(list(grants.values()), output_dir / "grants.jsonl")
    
    print(f"Data saved to JSONL files in {output_dir}")
    print(f"Separated data: {len(organisations)} unique organisations, {len(researchers)} unique researchers, {len(grants)} unique grants")


if __name__ == "__main__":
    print("=== Querying Grants from Australian Organisations ===\n")
    
    # Output directory for the results
    output_dir = config.DATA_DIR / "neo4j_exports"
    
    # Execute the query
    results = query_grants_from_organisations(
        config.Grants.neo4j_uri,
        config.Grants.neo4j_username,
        config.Grants.neo4j_password
    )
    
    if results is not None:
        print(f"Found {len(results)} organisation-researcher-grant relationships")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Save separated data to individual JSONL files
        save_separated_data(results, output_dir)
        
        # Print first few results as preview
        if results:
            print("\nFirst 3 results:")
            for i, result in enumerate(results[:3], 1):
                org_name = result['organisation'].get('name', 'Unknown')
                researcher_name = result['researcher'].get('given_names', '') + ' ' + result['researcher'].get('family_name', '')
                grant_title = result['grant'].get('title', 'No title')
                print(f"{i}. Org: {org_name}")
                print(f"   Researcher: {researcher_name.strip()}")
                print(f"   Grant: {grant_title}")
                print()
            
            if len(results) > 3:
                print(f"... and {len(results) - 3} more relationships")
    else:
        print("Failed to fetch data from Neo4j database")
