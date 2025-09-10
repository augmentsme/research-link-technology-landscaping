import json
import asyncio
from neo4j import GraphDatabase
import os
import sys
from pathlib import Path
import pandas as pd
from config import DATA_DIR, FOR_CODES_CLEANED_PATH
import config
import utils
from typing import List, Dict, Any, Optional, Union
from for_codes_cleaner import main as clean_for_codes
import httpx
import logging
from urllib.parse import urljoin
from datetime import datetime
from dataclasses import dataclass, field
import typer


def get_cipher_query(limit: Optional[int] = None) -> str:
    """Generate Neo4j cipher query with optional limit"""
    base_query = "MATCH (g:grant) RETURN g"
    if limit is not None:
        return f"{base_query} LIMIT {limit}"
    return base_query


# RLA Client Configuration
class APIConfig:
    """API-related configuration constants"""
    DEFAULT_TIMEOUT = 60
    DEFAULT_PAGE_SIZE = 10
    MAX_PAGE_SIZE = 250
    DEFAULT_MAX_CONCURRENT = 100
    BASE_URL = "https://researchlink.ardc.edu.au"


# RLA Client Exceptions
class RLAError(Exception):
    """Base exception class for RLA operations"""
    pass


class RLAAuthenticationError(RLAError):
    """Raised when authentication fails"""
    pass


class RLANotFoundError(RLAError):
    """Raised when a resource is not found"""
    pass


class RLAValidationError(RLAError):
    """Raised when request validation fails"""
    pass


class RLAServerError(RLAError):
    """Raised when server error occurs"""
    pass


# RLA Data Models
@dataclass
class Grant:
    """Represents a grant entity from the RLA API"""
    id: str = ""
    key: str = ""
    title: str = ""
    grant_title: str = ""
    grant_summary: str = ""
    funder: str = ""
    grant_id: str = ""
    funding_amount: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grant':
        """Create Grant from dictionary data"""
        return cls(
            id=data.get('id', ''),
            key=data.get('key', ''),
            title=data.get('title', ''),
            grant_title=data.get('grantTitle', ''),
            grant_summary=data.get('grantSummary', ''),
            funder=data.get('funder', ''),
            grant_id=data.get('grantId', ''),
            funding_amount=data.get('fundingAmount')
        )


@dataclass
class SearchResponse:
    """Response object for search operations"""
    total_results: int = 0
    current_page: int = 1
    from_index: int = 0
    size: int = 0
    results: List[Grant] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], result_type) -> 'SearchResponse':
        """Create SearchResponse from dictionary data"""
        results = []
        if 'results' in data and isinstance(data['results'], list):
            results = [result_type.from_dict(item) for item in data['results']]
        
        return cls(
            total_results=data.get('totalResults', 0),
            current_page=data.get('currentPage', 1),
            from_index=data.get('fromIndex', 0),
            size=data.get('size', 0),
            results=results
        )


class RLAClient:
    """Real RLA client implementation for grant enrichment"""
    
    def __init__(self, api_token: Optional[str] = None, base_url: str = APIConfig.BASE_URL, debug: bool = False):
        self.base_url = base_url.rstrip('/')
        self.debug = debug or os.getenv('PYRLA_DEBUG', '').lower() in ('1', 'true', 'yes')
        
        # Get token from parameter or environment
        self.api_token = api_token or os.getenv('ARL_API_TOKEN')
        if not self.api_token:
            raise RLAAuthenticationError(
                "API token is required. Provide it via api_token parameter or ARL_API_TOKEN environment variable"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        self.timeout = APIConfig.DEFAULT_TIMEOUT
        self._session: Optional[httpx.AsyncClient] = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create async session"""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout
            )
        return self._session
    
    async def _close_session(self):
        """Close async session"""
        if self._session and not self._session.is_closed:
            await self._session.aclose()
            self._session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _make_async_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async HTTP request to API endpoint"""
        url = urljoin(self.base_url, endpoint)
        session = await self._get_session()
        
        try:
            if self.debug:
                if params:
                    param_str = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                    full_url = f"{url}?{param_str}" if param_str else url
                    print(f"[DEBUG] API Request: {full_url}")
                else:
                    print(f"[DEBUG] API Request: {url}")
                
            response = await session.get(url, params=params or {})
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise RLAAuthenticationError("Authentication failed - check API token")
            elif response.status_code == 404:
                raise RLANotFoundError("Resource not found")
            elif response.status_code == 400:
                raise RLAValidationError(f"Invalid request parameters: {response.text}")
            elif 500 <= response.status_code < 600:
                raise RLAServerError(f"Server error ({response.status_code}): {response.text}")
            else:
                raise RLAError(f"API request failed ({response.status_code}): {response.text}")
                    
        except httpx.RequestError as e:
            raise RLAError(f"Request failed: {e}")
    
    async def search_grants(
        self,
        value: str = "",
        filter_query: str = "",
        page_number: int = 1,
        page_size: int = APIConfig.DEFAULT_PAGE_SIZE,
        advanced_search_query: str = ""
    ) -> SearchResponse:
        """Search for grants"""
        params = {
            "value": value,
            "filterQuery": filter_query,
            "pageNumber": page_number,
            "pageSize": min(page_size, APIConfig.MAX_PAGE_SIZE),
            "advancedSearchQuery": advanced_search_query
        }
        
        # Remove empty parameters
        params = {k: v for k, v in params.items() if v}
        
        data = await self._make_async_request("/search/grants", params)
        return SearchResponse.from_dict(data, Grant)
    
    async def get_grant(self, grant_id: str) -> Grant:
        """Get a specific grant by ID"""
        data = await self._make_async_request(f"/search/grant/{grant_id}")
        return Grant.from_dict(data)


class GrantsEnricher:
    """Handles grant data enrichment with summaries"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    async def enrich_grants_with_summaries(self, client: RLAClient, input_records: List[dict], key: str = "key") -> List[dict]:
        """Enrich grant records with grant summaries from the RLA API"""
        enriched_records = []
        semaphore = asyncio.Semaphore(APIConfig.DEFAULT_MAX_CONCURRENT)
        
        async def enrich_single_record(record: dict, index: int):
            async with semaphore:
                try:
                    # Only use key-based search strategy
                    grant = None
                    
                    # Search by key if it exists
                    if key in record and record[key]:
                        key_value = record[key]
                        
                        try:
                            # Search by key filter
                            response = await client.search_grants(
                                value="",
                                filter_query=f"key:{key_value}",
                                page_size=1
                            )
                            
                            if response.results:
                                grant = response.results[0]
                            else:
                                # Try extracting ID from key (e.g., "nhmrc/1047630" -> "1047630") as fallback
                                if '/' in key_value:
                                    extracted_id = key_value.split('/')[-1]
                                    try:
                                        grant = await client.get_grant(extracted_id)
                                    except RLAError:
                                        if self.debug:
                                            print(f"Failed to get grant by extracted ID '{extracted_id}' from key '{key_value}'")
                                
                        except RLAError as e:
                            if self.debug:
                                print(f"Search by key '{key_value}' failed: {e}")
                    else:
                        if self.debug:
                            print(f"Record {index + 1}: No '{key}' field found in record")
                    
                    # Create enriched record
                    enriched_record = record.copy()
                    if grant and grant.grant_summary:
                        enriched_record['grant_summary'] = grant.grant_summary
                        if self.debug:
                            print(f"Enriched record {index + 1}: Found grant summary")
                    else:
                        enriched_record['grant_summary'] = ""
                        if self.debug:
                            print(f"Record {index + 1}: No grant summary found for {key} '{record.get(key, f'No {key} found')}'")
                    
                    return enriched_record
                    
                except Exception as e:
                    if self.debug:
                        print(f"Error processing record {index + 1}: {e}")
                    enriched_record = record.copy()
                    enriched_record['grant_summary'] = ""
                    return enriched_record
        
        # Process all records concurrently
        tasks = [enrich_single_record(record, i) for i, record in enumerate(input_records)]
        enriched_records = await asyncio.gather(*tasks, return_exceptions=False)
        
        return enriched_records
    
    async def enrich_grants_data(self, grants_data: List[dict]) -> List[dict]:
        """Enrich grants data with summaries from the RLA API"""
        if not grants_data:
            return grants_data
        
        try:
            client = RLAClient(debug=self.debug)
            async with client:
                enriched_data = await self.enrich_grants_with_summaries(
                    client=client,
                    input_records=grants_data,
                    key="key"
                )
                return enriched_data
                
        except Exception as e:
            print(f"Error during enrichment: {e}")
            # Return data with empty summaries as fallback
            fallback_data = []
            for record in grants_data:
                enriched_record = record.copy()
                enriched_record['grant_summary'] = ""
                fallback_data.append(enriched_record)
            return fallback_data
    
    def run_async_enrichment(self, grants_data: List[dict]) -> List[dict]:
        """Helper function to run async enrichment in sync context"""
        try:
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # If we're already in an event loop, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.enrich_grants_data(grants_data))
                    return future.result()
            except RuntimeError:
                # No event loop is running, safe to use asyncio.run
                return asyncio.run(self.enrich_grants_data(grants_data))
        except Exception:
            # Fallback: create new event loop
            return asyncio.run(self.enrich_grants_data(grants_data))


class Neo4jConnector:
    """Handles Neo4j database connections and queries"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
    
    def get_grant_nodes_as_json(self, query: str) -> Optional[List[dict]]:
        """
        Connects to Neo4j, fetches nodes with the 'grant' label,
        and returns them as a list of dictionaries.
        """
        try:
            with GraphDatabase.driver(self.uri, auth=(self.username, self.password)) as driver:
                driver.verify_connectivity()
                records, summary, keys = driver.execute_query(query, database_="neo4j")
                
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


class GrantsCleaner:
    """Handles cleaning and processing of enriched grant data"""
    
    def clean_enriched_data(self, enriched_file_path: Path) -> Optional[pd.DataFrame]:
        """
        Clean the enriched grants data by filtering and standardizing data from all sources:
        - nhmrc.org: NHMRC grants with funding amounts
        - arc.gov.au: ARC grants with funding amounts and FOR codes
        - orcid.org: Additional grant records (often without summaries)
        - crossref.org: International grants with funding amounts
        
        Args:
            enriched_file_path: Path to the enriched grants JSONL file
            
        Returns:
            pandas.DataFrame: Cleaned grants data with standardized columns
        """
        try:
            df = utils.load_jsonl_file(enriched_file_path, as_dataframe=True)
            df = df.rename(columns={"key": "id"})
            df = df.set_index("id")
            
            print(f"Loaded {len(df):,} total records from {len(df.source.unique())} sources")
            for source in df.source.unique():
                count = len(df[df.source == source])
                print(f"  - {source}: {count:,} records")
            
            return self._clean_all_sources(df)
        
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return None
    
    def _clean_all_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean grant data from all sources with unified approach"""
        cleaned_parts = []
        
        # Process NHMRC grants (nhmrc.org)
        nhmrc_data = self._clean_nhmrc_grants(df[df.source == 'nhmrc.org'])
        if not nhmrc_data.empty:
            cleaned_parts.append(nhmrc_data)
            print(f"Processed NHMRC grants: {len(nhmrc_data):,} records")
        
        # Process ARC grants (arc.gov.au)
        arc_data = self._clean_arc_grants(df[df.source == 'arc.gov.au'])
        if not arc_data.empty:
            cleaned_parts.append(arc_data)
            print(f"Processed ARC grants: {len(arc_data):,} records")
        
        # Process ORCID grants (orcid.org) - filter for Australian funders
        orcid_data = self._clean_orcid_grants(df[df.source == 'orcid.org'])
        if not orcid_data.empty:
            cleaned_parts.append(orcid_data)
            print(f"Processed ORCID grants: {len(orcid_data):,} records")
        
        # Process Crossref grants (crossref.org) - keep all with funding amounts
        crossref_data = self._clean_crossref_grants(df[df.source == 'crossref.org'])
        if not crossref_data.empty:
            cleaned_parts.append(crossref_data)
            print(f"Processed Crossref grants: {len(crossref_data):,} records")
        
        # Combine all cleaned data
        if cleaned_parts:
            df_cleaned = pd.concat(cleaned_parts, axis=0, ignore_index=False)
            df_cleaned = self._standardize_final_columns(df_cleaned)
            print(f"Total cleaned records: {len(df_cleaned):,}")
            return df_cleaned
        else:
            print("No grants found after cleaning")
            return pd.DataFrame()
    
    def _clean_nhmrc_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NHMRC grants from nhmrc.org source"""
        if df.empty:
            return df
        
        # Filter for NHMRC grants with funding amounts
        nhmrc_grants = df[
            (df['funder'] == 'National Health and Medical Research Council') &
            (df['nhmrc_funding_amount'].notna())
        ].copy()
        
        if nhmrc_grants.empty:
            return pd.DataFrame()
        
        # Standardize columns
        nhmrc_grants.loc[:, 'funding_amount'] = nhmrc_grants['nhmrc_funding_amount']
        nhmrc_grants.loc[:, 'for_primary'] = pd.NA  # NHMRC doesn't have FOR codes
        nhmrc_grants.loc[:, 'for'] = pd.NA
        nhmrc_grants.loc[:, 'funder'] = 'NHMRC'
        
        return nhmrc_grants
    
    def _clean_arc_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ARC grants from arc.gov.au source"""
        if df.empty:
            return df
        
        # Take all ARC grants (they should all have funding amounts)
        arc_grants = df[df['arc_funding_at_announcement'].notna()].copy()
        
        if arc_grants.empty:
            return pd.DataFrame()
        
        # Standardize columns
        arc_grants.loc[:, 'funding_amount'] = arc_grants['arc_funding_at_announcement']
        arc_grants.loc[:, 'for_primary'] = arc_grants['arc_for_primary']
        arc_grants.loc[:, 'for'] = arc_grants['arc_for']
        arc_grants.loc[:, 'funder'] = 'ARC'
        
        return arc_grants
    
    def _clean_orcid_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ORCID grants - filter for Australian funders"""
        if df.empty:
            return df
        
        # Filter for Australian funders from ORCID data
        australian_funders = [
            'Australian Research Council',
            'National Health and Medical Research Council',
            'Cancer Australia'
        ]
        
        orcid_grants = df[df['funder'].isin(australian_funders)].copy()
        
        if orcid_grants.empty:
            return pd.DataFrame()
        
        # Standardize columns - ORCID grants typically don't have funding amounts or FOR codes
        orcid_grants.loc[:, 'funding_amount'] = pd.NA  # Use pandas NA instead of None
        orcid_grants.loc[:, 'for_primary'] = pd.NA
        orcid_grants.loc[:, 'for'] = pd.NA
        
        # Standardize funder names
        funder_mapping = {
            'Australian Research Council': 'ARC',
            'National Health and Medical Research Council': 'NHMRC',
            'Cancer Australia': 'Cancer Australia'
        }
        orcid_grants.loc[:, 'funder'] = orcid_grants['funder'].map(funder_mapping)
        
        return orcid_grants
    
    def _clean_crossref_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Crossref grants - keep those with funding amounts"""
        if df.empty:
            return df
        
        # Filter for grants with funding amounts
        crossref_grants = df[df['funding_amount'].notna()].copy()
        
        if crossref_grants.empty:
            return pd.DataFrame()
        
        # Standardize columns
        crossref_grants.loc[:, 'for_primary'] = pd.NA
        crossref_grants.loc[:, 'for'] = pd.NA
        # Keep original funder name for international grants
        
        return crossref_grants
    
    def _standardize_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns are present and in the correct order"""
        required_columns = [
            'title', 'grant_summary', 'funding_amount', 'start_year', 'end_year', 
            'funder', 'for_primary', 'for', 'source'
        ]
        
        # Add missing columns with pd.NA values
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Select final columns in the desired order
        df_final = df[required_columns].copy()
        
        # Fill missing end_year with pd.NA if not present
        if 'end_year' not in df.columns:
            df_final['end_year'] = pd.NA
        
        return df_final


class DataPreprocessor:
    """Main orchestrator for the data preprocessing pipeline"""
    
    def __init__(self, debug: bool = False, force_rerun: bool = False, limit: Optional[int] = None):
        self.debug = debug
        self.force_rerun = force_rerun
        self.limit = limit
        self.output_file = DATA_DIR / "grants_raw.jsonl"
        self.enricher = GrantsEnricher(debug=debug)
        self.cleaner = GrantsCleaner()
        
        # Initialize Neo4j connector
        self.neo4j_connector = Neo4jConnector(
            uri=config.Grants.neo4j_uri,
            username=config.Grants.neo4j_username,
            password=config.Grants.neo4j_password
        )
    
    def save_to_jsonl(self, data: List[dict], filename: Path) -> None:
        """Saves a list of dictionaries to a JSONL file using utils function."""
        if data is not None:
            utils.save_jsonl_file(data, filename)
    
    def load_or_fetch_grants_data(self) -> Optional[List[dict]]:
        """Load existing grants data or fetch from Neo4j"""
        grants_data = None
        
        # Check if grants data already exists and force_rerun is not enabled
        if self.output_file.exists() and not self.force_rerun:
            try:
                grants_data = utils.load_jsonl_file(self.output_file)
                print(f"Loaded existing grants data: {len(grants_data)} records")
            except Exception:
                grants_data = None
        
        # If no existing data, force_rerun is enabled, or loading failed - fetch from Neo4j
        if grants_data is None or self.force_rerun:
            if self.force_rerun:
                print("Force rerun enabled - fetching fresh data from Neo4j...")
            else:
                print("Fetching grants data from Neo4j...")
            
            cipher_query = get_cipher_query(self.limit)
            grants_data = self.neo4j_connector.get_grant_nodes_as_json(cipher_query)
            
            if grants_data is not None:
                self.save_to_jsonl(grants_data, self.output_file)
                print(f"Fetched and saved {len(grants_data)} grants from Neo4j")
            else:
                print("Failed to fetch data from Neo4j database")
                return None
        
        return grants_data
    
    def enrich_grants_data(self, grants_data: List[dict]) -> List[dict]:
        """Enrich grants data with summaries"""
        # Enrich the data with grant summaries from RLA API (skip if enriched file exists and force_rerun is not enabled)
        if config.Grants.enriched_path.exists() and not self.force_rerun:
            print("Loading existing enriched data...")
            enriched_data = utils.load_jsonl_file(config.Grants.enriched_path)
        else:
            if self.force_rerun:
                print("Force rerun enabled - enriching grants data from scratch...")
            else:
                print("Enriching grants data...")
            
            enriched_data = self.enricher.run_async_enrichment(grants_data)
            self.save_to_jsonl(enriched_data, config.Grants.enriched_path)
            print(f"Enriched and saved {len(enriched_data)} grants")
        
        return enriched_data
    
    def clean_and_save_grants(self, enriched_data: List[dict]) -> int:
        """Clean enriched data and save final grants file"""
        # Always clean and save when called, but show appropriate message for force_rerun
        if self.force_rerun:
            print("Force rerun enabled - cleaning grants data from scratch...")
        else:
            print("Cleaning grants data...")
        
        df_cleaned = self.cleaner.clean_enriched_data(config.Grants.enriched_path)
        
        if df_cleaned is not None:
            cleaned_data = df_cleaned.reset_index().to_dict('records')
            utils.save_jsonl_file(cleaned_data, config.Grants.grants_path)
            print(f"Cleaned and saved {len(df_cleaned)} grants")
            return len(df_cleaned)
        else:
            print("Grants processing completed with errors (cleaning failed)")
            return 0
    
    def run_pipeline(self) -> None:
        """Run the complete data preprocessing pipeline"""
        print("=== Research Link Technology Landscaping - Data Preprocessing ===")
        if self.force_rerun:
            print("🔄 FORCE RERUN MODE - All data will be regenerated from scratch")
        print()
        

        
        # Process grants data
        grants_data = self.load_or_fetch_grants_data()
        if grants_data is None:
            print("Failed to load or fetch grants data. Exiting.")
            return
        
        enriched_data = self.enrich_grants_data(grants_data)
        cleaned_count = self.clean_and_save_grants(enriched_data)
        
        print(f"Grants processed: original={len(grants_data)}, enriched={len(enriched_data)}, cleaned={cleaned_count}")
        
        print("\n" + "="*70)
        print("🎉 Data preprocessing pipeline completed!")
        print("="*70)


app = typer.Typer(help="Research Link Technology Landscaping - Data Preprocessing")


@app.command()
def main(
    force_rerun: bool = typer.Option(
        False, 
        "--force-rerun", 
        help="Force rerun all steps including Neo4j query, ignoring existing data files"
    ),
    debug: bool = typer.Option(
        False, 
        "--debug", 
        help="Enable debug output"
    ),
    limit: Optional[int] = typer.Option(
        None, 
        "--limit", 
        help="Limit the number of grants retrieved from Neo4j (default: no limit, retrieves all grants)"
    )
):
    """Run the data preprocessing pipeline"""
    # Create and run the data preprocessing pipeline
    preprocessor = DataPreprocessor(debug=debug, force_rerun=force_rerun, limit=limit)
    preprocessor.run_pipeline()


if __name__ == "__main__":
    app()
    

