import json
import asyncio
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import pandas as pd
from config import DATA_DIR, FOR_CODES_CLEANED_PATH

# Add the parent directory to the path to import pyrla
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "pyrla"))


from pyrla.cli import enrich_grants_with_summaries
from pyrla.client import RLAClient
from for_codes_cleaner import main as clean_for_codes


load_dotenv()
# --- Connection Details (replace with your credentials) ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"

# --- Cypher Query ---
# This query finds all nodes with the label "grant" and returns them.
cypher_query = "MATCH (g:grant) RETURN g"

# --- Output Files ---
output_file = DATA_DIR / "grants_raw.json"
enriched_output_file = DATA_DIR / "grants_enriched.json"
cleaned_output_file = DATA_DIR / "grants_cleaned.json"

def get_grant_nodes_as_json(uri, user, password, query):
    """
    Connects to Neo4j, fetches nodes with the 'grant' label,
    and returns them as a list of dictionaries.
    """
    try:
        # Establish a connection to the Neo4j database. [4]
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            # Verify that the connection is established successfully. [7]
            driver.verify_connectivity()
            print("Connection to Neo4j successful.")

            # Execute the Cypher query to get all grant nodes.
            records, summary, keys = driver.execute_query(query, database_="neo4j")

            # Process the results.
            grant_nodes = []
            for record in records:
                # Each record contains a node 'g'.
                node = record["g"]
                # Convert the node properties to a dictionary.
                grant_nodes.append(dict(node))

            return grant_nodes

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_to_json(data, filename):
    """
    Saves a list of dictionaries to a JSON file. [1, 2, 3]
    """
    if data is not None:
        with open(filename, 'w') as json_file:
            # Use json.dump() to write the data to the file. [5, 6]
            # The 'indent' parameter formats the JSON for readability.
            json.dump(data, json_file, indent=4)
        print(f"Successfully saved {len(data)} grant nodes to {filename}")


async def enrich_grants_data(grants_data):
    """
    Enrich grants data with summaries from the RLA API.
    Assumes each grant has a 'key' field.
    """

    
    if not grants_data:
        print("No grants data to enrich")
        return grants_data
    
    try:
        # Initialize RLA client
        client = RLAClient(debug=False)
        
        print(f"Starting enrichment of {len(grants_data)} grants...")
        
        # Use the enrichment function from pyrla CLI
        enriched_data = await enrich_grants_with_summaries(
            client=client, 
            input_records=grants_data, 
            key="key"  # Assume all records have a 'key' field
        )
        
        # Count successful enrichments
        successful_count = sum(1 for record in enriched_data if record.get('grant_summary', ''))
        print(f"Successfully enriched {successful_count}/{len(enriched_data)} grants with summaries")
        
        return enriched_data
        
    except Exception as e:
        print(f"Error during enrichment: {e}")
        print("Returning original data without enrichment")
        return grants_data


def run_async_enrichment(grants_data):
    """
    Helper function to run async enrichment in sync context
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, enrich_grants_data(grants_data))
                return future.result()
        else:
            return loop.run_until_complete(enrich_grants_data(grants_data))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(enrich_grants_data(grants_data))


def clean_enriched_data(enriched_file_path):
    """
    Clean the enriched grants data by filtering ARC and NHMRC grants
    and standardizing the funding amount field.
    
    Args:
        enriched_file_path: Path to the enriched grants JSON file
        
    Returns:
        pandas.DataFrame: Cleaned grants data
    """
    try:
        print(f"Cleaning enriched data from {enriched_file_path}...")
        
        # Load the enriched data
        df = pd.read_json(enriched_file_path)
        df = df.rename(columns={"key": "id"})
        df = df.set_index("id")

        print(f"Loaded {len(df)} grants for cleaning")
        
        # Filter ARC grants and standardize funding_amount
        arc = df[df.index.str.startswith("arc")]
        arc['funding_amount'] = arc['arc_funding_at_announcement']
        print(f"Found {len(arc)} ARC grants")
        
        # Filter NHMRC grants and standardize funding_amount  
        nhmrc = df[df.index.str.startswith("nhmrc")]
        nhmrc['funding_amount'] = nhmrc['nhmrc_funding_amount']
        print(f"Found {len(nhmrc)} NHMRC grants")
        
        # Combine both datasets
        df_cleaned = pd.concat([arc, nhmrc], axis=0)
        
        # Select only the required columns
        df_cleaned = df_cleaned[['title', 'grant_summary', 'funding_amount', 'start_year', 'end_year', 'funder']]
        
        
        print(f"Cleaned dataset contains {len(df_cleaned)} grants")
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

if __name__ == "__main__":
    print("=== Research Link Technology Landscaping - Data Preprocessing ===\n")
    
    # ===============================================================================
    # PROCESS FOR CODES DATA
    # ===============================================================================
    print("1. Processing Field of Research (FOR) codes...")
    for_hierarchy = clean_for_codes(FOR_CODES_CLEANED_PATH)
    
    if for_hierarchy:
        divisions = len(for_hierarchy)
        groups = sum(len(div['groups']) for div in for_hierarchy.values())
        print(f"✅ FOR codes processing completed successfully")
        print(f"   - {divisions} divisions")
        print(f"   - {groups} groups")
    else:
        print("⚠️  FOR codes processing failed or skipped")
    
    print("\n" + "="*70 + "\n")
    
    # ===============================================================================
    # PROCESS GRANTS DATA
    # ===============================================================================
    print("2. Processing grants data...")
    
    grants_data = None
    
    # Check if grants data already exists
    if output_file.exists():
        print(f"Found existing grants data at {output_file}")
        try:
            with open(output_file, 'r') as json_file:
                grants_data = json.load(json_file)
            print(f"Loaded {len(grants_data)} existing grants from file")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading existing file: {e}")
            print("Will fetch fresh data from Neo4j...")
            grants_data = None
    
    # If no existing data, fetch from Neo4j
    if grants_data is None:
        print("Fetching grant nodes from Neo4j database...")
        grants_data = get_grant_nodes_as_json(NEO4J_URI, NEO4J_USER, os.getenv("NEO4J_PASSWORD"), cypher_query)
        
        if grants_data is not None:
            save_to_json(grants_data, output_file)
        else:
            print("Failed to fetch data from Neo4j database")
            exit(1)
    
    # Enrich the data with grant summaries from RLA API (skip if enriched file exists)
    if enriched_output_file.exists():
        print(f"Found existing enriched data at {enriched_output_file}, loading...")
        try:
            with open(enriched_output_file, 'r') as f:
                enriched_data = json.load(f)
            print(f"Loaded {len(enriched_data)} enriched records from file")
        except Exception as e:
            print(f"Error reading enriched file: {e}")
            print("\nEnriching grants with summaries from RLA API...")
            enriched_data = run_async_enrichment(grants_data)
            save_to_json(enriched_data, enriched_output_file)
    else:
        print("\nEnriching grants with summaries from RLA API...")
        enriched_data = run_async_enrichment(grants_data)
        # Save the enriched data to a separate JSON file
        save_to_json(enriched_data, enriched_output_file)
    
    # Clean the enriched data
    print("\nCleaning enriched grants data...")
    df_cleaned = clean_enriched_data(enriched_output_file)
    
    if df_cleaned is not None:
        # Save cleaned data as JSON
        cleaned_data = df_cleaned.reset_index().to_dict('records')
        save_to_json(cleaned_data, cleaned_output_file)
        
        print(f"\n✅ Grants processing complete:")
        print(f"  - Original data: {output_file} ({len(grants_data)} grants)")
        print(f"  - Enriched data: {enriched_output_file} ({len(enriched_data)} grants)")
        print(f"  - Cleaned data: {cleaned_output_file} ({len(df_cleaned)} grants)")
    else:
        print("❌ Grants data cleaning failed - only original and enriched data available")
        print(f"\nProcessing complete:")
        print(f"  - Original data: {output_file}")
        print(f"  - Enriched data: {enriched_output_file}")
    
    print("\n" + "="*70)
    print("🎉 Data preprocessing pipeline completed!")
    print("="*70)
    

