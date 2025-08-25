import json
import asyncio
from neo4j import GraphDatabase
# from dotenv import load_dotenv
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


# load_dotenv()
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
        print(f"Error fetching grants from Neo4j: {e}")
        return None

def save_to_json(data, filename):
    """
    Saves a list of dictionaries to a JSON file. [1, 2, 3]
    """
    if data is not None:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)


async def enrich_grants_data(grants_data):
    """
    Enrich grants data with summaries from the RLA API.
    Assumes each grant has a 'key' field.
    """

    
    if not grants_data:
        return grants_data
    
    try:
        client = RLAClient(debug=False)
        enriched_data = await enrich_grants_with_summaries(
            client=client,
            input_records=grants_data,
            key="key"
        )
        return enriched_data
        
    except Exception as e:
        print(f"Error during enrichment: {e}")
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
        df = pd.read_json(enriched_file_path)
        df = df.rename(columns={"key": "id"})
        df = df.set_index("id")

        arc = df[df.index.str.startswith("arc")]
        arc.loc[:, 'funding_amount'] = arc['arc_funding_at_announcement']

        nhmrc = df[df.index.str.startswith("nhmrc")]
        nhmrc.loc[:, 'funding_amount'] = nhmrc['nhmrc_funding_amount']

        df_cleaned = pd.concat([arc, nhmrc], axis=0)
        df_cleaned = df_cleaned[['title', 'grant_summary', 'funding_amount', 'start_year', 'end_year', 'funder']]
        return df_cleaned

    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

if __name__ == "__main__":
    print("=== Research Link Technology Landscaping - Data Preprocessing ===\n")
    
    # ===============================================================================
    # PROCESS FOR CODES DATA
    # ===============================================================================
    # Process FOR codes (single short status print)
    for_hierarchy = clean_for_codes(FOR_CODES_CLEANED_PATH)
    if for_hierarchy:
        divisions = len(for_hierarchy)
        groups = sum(len(div['groups']) for div in for_hierarchy.values())
        print(f"FOR codes processed: {divisions} divisions, {groups} groups")
    else:
        print("FOR codes processing skipped or failed")
    
    print("\n" + "="*70 + "\n")
    
    # ===============================================================================
    # PROCESS GRANTS DATA
    # ===============================================================================
    # Process grants data (single short status print)
    
    grants_data = None
    
    # Check if grants data already exists
    if output_file.exists():
        try:
            with open(output_file, 'r') as json_file:
                grants_data = json.load(json_file)
        except (json.JSONDecodeError, Exception):
            grants_data = None
    
    # If no existing data, fetch from Neo4j
    from dotenv import dotenv_values
    CONFIG = dotenv_values()

    if grants_data is None:
        grants_data = get_grant_nodes_as_json(CONFIG["NEO4J_URL"], CONFIG["NEO4J_USER"], CONFIG["NEO4J_PASSWORD"], cypher_query)

        if grants_data is not None:
            save_to_json(grants_data, output_file)
        else:
            print("Failed to fetch data from Neo4j database")
            exit(1)
    
    # Enrich the data with grant summaries from RLA API (skip if enriched file exists)
    if enriched_output_file.exists():
        try:
            with open(enriched_output_file, 'r') as f:
                enriched_data = json.load(f)
        except Exception:
            enriched_data = run_async_enrichment(grants_data)
            save_to_json(enriched_data, enriched_output_file)
    else:
        enriched_data = run_async_enrichment(grants_data)
        save_to_json(enriched_data, enriched_output_file)
    
    # Clean the enriched data
    df_cleaned = clean_enriched_data(enriched_output_file)
    if df_cleaned is not None:
        cleaned_data = df_cleaned.reset_index().to_dict('records')
        save_to_json(cleaned_data, cleaned_output_file)
        print(f"Grants processed: original={len(grants_data) if grants_data else 0}, enriched={len(enriched_data) if enriched_data else 0}, cleaned={len(df_cleaned)}")
    else:
        print("Grants processing completed with errors (cleaning failed)")
    
    print("\n" + "="*70)
    print("🎉 Data preprocessing pipeline completed!")
    print("="*70)
    

