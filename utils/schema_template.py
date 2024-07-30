import json
from typing import Any

"""LLM format schema generator"""

def read_json(file_path: str) -> Any:
    """Reads a json file to the Python object it contains."""
    with open(file_path, 'rb') as fp:
        data=json.load(fp)
        return data
    
def relationship_string_generator(schema):        
    relations_str = "The relationships:\n"
    for jsn in schema["relationships"]:
        cypher_format = f"(:{jsn['start']})-[:{jsn['type']}]->(:{jsn['end']})\n"
        relations_str += cypher_format
    
    return relations_str


def properties_string_generator(schema, key):
    if key == "node_props":
        final_str = "Node properties:\n"
    elif key == "rel_props":
        final_str = "Relationship properties:\n"

    for node in schema[key].keys():
        frmt= f"- **{node}**\n"
        for prop in schema[key][node]:
            datatype_or_type = prop.get('datatype', prop.get('type'))
            format = f"  - `{prop['property']}`: {datatype_or_type}\n"
            frmt +=format
        final_str += frmt

    return final_str

def generate_full_schema_template(schema):
    schema_info = ""
    node_info = properties_string_generator(schema, "node_props")
    schema_info += node_info + "\n"
    rel_info = properties_string_generator(schema, "rel_props")
    schema_info += rel_info + "\n"
    relation = relationship_string_generator(schema)
    schema_info += relation

    return schema_info

if __name__ == "__main__":
    
    file_path = "./text2cypher/datasets/functional_cypher/schema_file.json"
    schema = read_json(file_path)
    schema_template = generate_full_schema_template(schema)
    print(schema_template)