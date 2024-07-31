from utils.neo4jutils.neo4j_schema import *
from utils import schema_template

# NOTE: for BH schema generation from a local BH instance
gutils = Neo4jSchema(url = "neo4j://localhost:7687", username = "neo4j", password="bloodhoundcommunityedition",  database = "neo4j")
jschema = gutils.get_structured_schema
schema = schema_template.generate_full_schema_template(jschema)
with open("schema.txt", 'w') as f:
    f.write(schema)
