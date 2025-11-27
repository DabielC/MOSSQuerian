from pocketflow import Flow
from nodes import GetSchema, GenerateSQL, ValidateSQL, ExecuteSQL, RewriteSQL, DebugSQL


def create_text_to_mongo_flow():
    """Creates the text-to-MongoDB workflow with a debug loop."""
    get_schema_node = GetSchema()
    generate_query_node = GenerateSQL()
    validate_query_node = ValidateSQL()
    execute_query_node = ExecuteSQL()
    rewrite_query_node = RewriteSQL()
    debug_query_node = DebugSQL()

    # Define the main flow sequence using the default transition operator
    get_schema_node >> generate_query_node >> validate_query_node >> execute_query_node

    # Route validator failures to the rewrite agent
    validate_query_node - "rewrite_retry" >> rewrite_query_node

    # Route empty-result retries through the rewrite agent
    execute_query_node - "rewrite_retry" >> rewrite_query_node
    rewrite_query_node >> validate_query_node

    # Route execution errors to the debug agent
    execute_query_node - "error_retry" >> debug_query_node

    # After debugging, re-run validation before execution
    debug_query_node >> validate_query_node

    return Flow(start=get_schema_node)
