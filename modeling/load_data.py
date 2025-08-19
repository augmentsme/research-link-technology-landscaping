import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()
# --- Connection Details (replace with your credentials) ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"

# --- Cypher Query ---
# This query finds all nodes with the label "grant" and returns them.
cypher_query = "MATCH (g:grant) RETURN g"

# --- Output File ---
output_file = "grants.json"

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
        print(f"Successfully downloaded {len(data)} grant nodes to {filename}")

if __name__ == "__main__":
    # Fetch the grant nodes from the database.
    grants_data = get_grant_nodes_as_json(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, cypher_query)

    # Save the fetched data to a JSON file.
    save_to_json(grants_data, output_file)