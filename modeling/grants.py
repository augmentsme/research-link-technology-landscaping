import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
import typer
from inspect_ai.dataset import Sample
from neo4j import GraphDatabase

from config import CONFIG
from database import get_db_manager

# Add the parent directory to the path to import pyrla
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "pyrla"))

from pyrla.cli import enrich_grants_with_summaries
from pyrla.client import RLAClient

# --- Connection Details (replace with your credentials) ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"

# --- Cypher Query ---
# This query finds all nodes with the label "grant" and returns them.
cypher_query = "MATCH (g:grant) RETURN g"

# --- Output Files ---
DATA_DIR = CONFIG.root_dir / "data"
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
    Saves a list of dictionaries to a JSON file.
    """
    if data is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
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


def process_grants_pipeline():
    """
    Complete grants processing pipeline: fetch, enrich, and clean grants data.
    
    Returns:
        tuple: (original_count, enriched_count, cleaned_count)
    """
    print("Processing grants data...")
    
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
        grants_data = get_grant_nodes_as_json(
            CONFIG["NEO4J_URL"], 
            CONFIG["NEO4J_USER"], 
            CONFIG["NEO4J_PASSWORD"], 
            cypher_query
        )

        if grants_data is not None:
            save_to_json(grants_data, output_file)
        else:
            print("Failed to fetch data from Neo4j database")
            return (0, 0, 0)
    
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
        
        original_count = len(grants_data) if grants_data else 0
        enriched_count = len(enriched_data) if enriched_data else 0
        cleaned_count = len(df_cleaned)
        
        return (original_count, enriched_count, cleaned_count)
    else:
        print("Grants processing completed with errors (cleaning failed)")
        return (len(grants_data) if grants_data else 0, len(enriched_data) if enriched_data else 0, 0)


def record_to_sample(record: dict) -> Sample:
    """Convert a record to a Sample, loading from ChromaDB if available."""
    
    # Try to get record from ChromaDB first
    try:
        db_manager = get_db_manager()
        grant_data = db_manager.get_grant(record["id"])
        if grant_data:
            record = grant_data  # Use ChromaDB data
    except Exception as e:
        print(f"Warning: Could not load grant {record['id']} from ChromaDB: {e}")
        # Fall back to provided record
    
    return Sample(
        id=record["id"],
        input=f"""**Grant ID**: {record["id"]}
**Title**: {record["title"]}
**Summary**: 
{record["grant_summary"]}
                """,
        metadata={
            "title": record["title"],
            "summary": record["grant_summary"],
            "funding_amount": record["funding_amount"],
            "funder": record["funder"],
            "start_year": record["start_year"],
            "end_year": record["end_year"],
        }
    )


def load_grants_to_chromadb(vllm_url: str = None):
    """Load grants from JSON file into ChromaDB (management function)."""
    try:
        # Use grants embedding configuration from YAML config
        grants_config = CONFIG.get_grants_config()
        db_manager = get_db_manager(vllm_url)
        
        # Check if grants are already stored
        existing_count = db_manager.count_documents("grants")
        print(f"📊 Found {existing_count} existing grants in ChromaDB")
        
        # Load grants from JSON file
        with open(CONFIG.grants_file, 'r', encoding='utf-8') as f:
            grants_data = json.load(f)
        
        print(f"📄 Loading {len(grants_data)} grants from {CONFIG.grants_file}")
        
        # Get existing grant IDs to avoid duplicates
        existing_grant_ids = set(db_manager.get_all_grant_ids())
        print(f"📋 Found {len(existing_grant_ids)} existing grant IDs")
        
        # Filter out grants that already exist
        new_grants = []
        skipped_count = 0
        for grant in grants_data:
            if grant["id"] not in existing_grant_ids:
                new_grants.append(grant)
            else:
                skipped_count += 1
        
        print(f"🔍 Found {len(new_grants)} new grants to store, {skipped_count} already exist")
        
        # Store new grants in batches
        if new_grants:
            # Prepare batch data
            documents = []
            metadatas = []
            ids = []
            
            for grant in new_grants:
                # Create document text combining title and summary
                document_text = f"Title: {grant.get('title', '')}\n\nSummary: {grant.get('grant_summary', '')}"
                
                # Prepare the metadata
                metadata = {
                    "grant_id": grant["id"],
                    "title": grant.get("title", ""),
                    "funder": grant.get("funder", ""),
                    "funding_amount": grant.get("funding_amount", 0),
                    "start_year": grant.get("start_year", 0),
                    "end_year": grant.get("end_year", 0),
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(grant["id"])
            
            # Store all new grants in batches using optimal batch size
            db_manager.store_documents(
                collection_name="grants",
                documents=documents,
                metadatas=metadatas,
                ids=ids
                # Use optimal batch size automatically
            )
            
            stored_count = len(new_grants)
        else:
            stored_count = 0
        
        total_count = db_manager.count_documents("grants")
        embeddings_status = "with embeddings" if grants_config['use_embeddings'] else "without embeddings"
        print(f"✅ Stored {stored_count} new grants. Total grants in ChromaDB: {total_count}")
        print(f"🔧 Grants stored {embeddings_status}")
        
    except Exception as e:
        print(f"❌ Error loading grants: {e}")
        raise


def grants_info_command(vllm_url: str = None):
    """Show information about grants in ChromaDB (management function)."""
    try:
        grants_config = CONFIG.get_grants_config()
        db_manager = get_db_manager(vllm_url)
        total_count = db_manager.count_documents("grants")
        embeddings_status = "with embeddings" if grants_config['use_embeddings'] else "without embeddings"
        print(f"📊 Total grants in ChromaDB: {total_count} ({embeddings_status})")
        
        if total_count > 0:
            # Get a sample of grant IDs
            grant_ids = db_manager.get_all_grant_ids()
            print(f"📝 Sample grant IDs: {grant_ids[:5]}")
            
    except Exception as e:
        print(f"❌ Error getting grants info: {e}")
        raise


def clear_grants_command(force: bool = False, vllm_url: str = None):
    """Clear all grants from ChromaDB (management function)."""
    try:
        grants_config = CONFIG.get_grants_config()
        db_manager = get_db_manager(vllm_url)
        
        if not force:
            total_count = db_manager.count_documents("grants")
            if total_count == 0:
                print("📊 No grants found in ChromaDB")
                return
            
            confirm = input(f"⚠️  Are you sure you want to delete {total_count} grants? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Operation cancelled")
                return
        
        # Clear the collection
        db_manager.clear_collection("grants")
        print("✅ All grants cleared from ChromaDB")
                
    except Exception as e:
        print(f"❌ Error clearing grants: {e}")
        raise


def load_and_store_grants_in_chromadb(vllm_url: str = None):
    """Load grants from JSON file and store them in ChromaDB."""
    # Use YAML config defaults if not provided
    vllm_config = CONFIG.get_vllm_config()
    grants_config = CONFIG.get_grants_config()
    actual_vllm_url = vllm_url or vllm_config['base_url']
    
    # Use the grants embedding configuration from YAML config
    db_manager = get_db_manager(actual_vllm_url)
    
    # Check if grants are already stored
    existing_count = db_manager.count_documents("grants")
    print(f"📊 Found {existing_count} existing grants in ChromaDB")
    
    # Load grants from JSON file
    with open(CONFIG.grants_file, 'r', encoding='utf-8') as f:
        grants_data = json.load(f)
    
    print(f"📄 Loading {len(grants_data)} grants from {CONFIG.grants_file}")
    
    # Get existing grant IDs to avoid duplicates
    existing_grant_ids = set(db_manager.get_all_grant_ids())
    print(f"📋 Found {len(existing_grant_ids)} existing grant IDs")
    
    # Filter out grants that already exist
    new_grants = []
    skipped_count = 0
    for grant in grants_data:
        if grant["id"] not in existing_grant_ids:
            new_grants.append(grant)
        else:
            skipped_count += 1
    
    print(f"🔍 Found {len(new_grants)} new grants to store, {skipped_count} already exist")
    
    # Store new grants in batches
    if new_grants:
        # Prepare batch data
        documents = []
        metadatas = []
        ids = []
        
        for grant in new_grants:
            # Create document text combining title and summary
            document_text = f"Title: {grant.get('title', '')}\n\nSummary: {grant.get('grant_summary', '')}"
            
            # Prepare the metadata
            metadata = {
                "grant_id": grant["id"],
                "title": grant.get("title", ""),
                "funder": grant.get("funder", ""),
                "funding_amount": grant.get("funding_amount", 0),
                "start_year": grant.get("start_year", 0),
                "end_year": grant.get("end_year", 0),
            }
            
            documents.append(document_text)
            metadatas.append(metadata)
            ids.append(grant["id"])
        
        # Store all new grants in batches using optimal batch size
        db_manager.store_documents(
            collection_name="grants",
            documents=documents,
            metadatas=metadatas,
            ids=ids
            # Use optimal batch size automatically
        )
        
        stored_count = len(new_grants)
    else:
        stored_count = 0
    
    total_count = db_manager.count_documents("grants")
    print(f"✅ Stored {stored_count} new grants. Total grants in ChromaDB: {total_count}")
    
    embeddings_status = "with embeddings" if grants_config['use_embeddings'] else "without embeddings"
    print(f"🔧 Grants stored {embeddings_status}")
    
    return db_manager


# Grants management functionality has been integrated into the unified CLI (cli.py)
# Use: python cli.py data --help
