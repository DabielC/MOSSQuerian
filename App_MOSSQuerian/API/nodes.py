import pymongo
import time
import yaml
import re
from collections import Counter
from numbers import Number
from statistics import mean

from pocketflow import Node
from utils.call_llm import call_llm_openai, call_llm_gemini


def _format_number(value: float) -> str:
    """Return a compact string version of a numeric value."""
    try:
        if value is None or (isinstance(value, float) and (value != value)):
            return "nan"
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.01:
            return f"{value:.3e}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _safe_preview(value, max_length: int = 48) -> str:
    """Generate a short text preview for a sample field value."""
    try:
        preview = repr(value)
    except Exception:
        preview = str(value)
    preview = preview.replace("\n", " ")
    if len(preview) > max_length:
        return preview[: max_length - 3] + "..."
    return preview


def _analyze_collection(collection, sample_limit: int = 50):
    """Collect lightweight statistics for a MongoDB collection."""
    stats = {
        "document_count": 0,
        "sampled_count": 0,
        "field_stats": {},
    }

    try:
        stats["document_count"] = collection.estimated_document_count()
    except Exception:
        # Fallback: avoid failing when the server forbids estimated counts
        stats["document_count"] = None

    sample_docs = list(collection.find().limit(sample_limit))
    stats["sampled_count"] = len(sample_docs)

    field_stats = {}
    for doc in sample_docs:
        for key, value in doc.items():
            if key == "_id":
                continue
            if key not in field_stats:
                field_stats[key] = {
                    "types": Counter(),
                    "numeric_values": [],
                    "string_values": Counter(),
                    "list_lengths": [],
                    "samples": [],
                }
            current = field_stats[key]

            value_type = type(value).__name__ if value is not None else "NoneType"
            current["types"][value_type] += 1

            if value is None:
                continue

            if isinstance(value, Number) and not isinstance(value, bool):
                try:
                    current["numeric_values"].append(float(value))
                except Exception:
                    pass

            if isinstance(value, str):
                view = value.strip()
                if view:
                    current["string_values"][view[:64]] += 1
                if len(current["samples"]) < 3:
                    current["samples"].append(_safe_preview(value))
            elif isinstance(value, list):
                try:
                    current["list_lengths"].append(len(value))
                except Exception:
                    pass
                if len(current["samples"]) < 3:
                    current["samples"].append(_safe_preview(value))
            else:
                if len(current["samples"]) < 3:
                    current["samples"].append(_safe_preview(value))

    stats["field_stats"] = field_stats
    return stats

class GetSchema(Node):
    def prep(self, shared):
        return shared["mongo_uri"], shared["db_name"]

    def exec(self, prep_res):
        mongo_uri, db_name = prep_res
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection_names = sorted(db.list_collection_names())
        schema_blocks = []
        summary_blocks = []
        collection_fields = {}
        collection_summaries = {}
        collection_stats = {}

        try:
            for coll_name in collection_names:
                collection = db[coll_name]
                stats = _analyze_collection(collection)
                collection_stats[coll_name] = stats
                field_stats = stats.get("field_stats", {}) or {}

                # Build schema view
                schema_lines = [f"Collection: {coll_name}"]
                if field_stats:
                    for field in sorted(field_stats.keys()):
                        type_counter = field_stats[field]["types"]
                        type_summary = ", ".join(
                            f"{t}:{c}" for t, c in type_counter.most_common(3)
                        ) or "unknown"
                        schema_lines.append(f"  - {field} (types: {type_summary})")
                    collection_fields[coll_name] = sorted(field_stats.keys())
                else:
                    schema_lines.append("  (No sampled fields)")
                    collection_fields[coll_name] = []
                schema_blocks.append("\n".join(schema_lines))

                # Build summary view
                summary_lines = [f"Collection: {coll_name}"]
                doc_count = stats.get("document_count")
                if doc_count is not None:
                    summary_lines.append(f"  documents: {doc_count}")
                else:
                    summary_lines.append("  documents: unknown")
                summary_lines.append(f"  sampled: {stats.get('sampled_count', 0)}")

                if field_stats:
                    for field in sorted(field_stats.keys()):
                        field_info = field_stats[field]
                        type_counter = field_info["types"]
                        type_summary = ", ".join(
                            f"{t}:{c}" for t, c in type_counter.most_common(3)
                        ) or "unknown"

                        parts = [f"types={type_summary}"]

                        numeric_values = field_info.get("numeric_values", [])
                        if numeric_values:
                            parts.append(
                                f"range={_format_number(min(numeric_values))}..{_format_number(max(numeric_values))}"
                            )
                            try:
                                numeric_avg = mean(numeric_values)
                                parts.append(f"avg={_format_number(numeric_avg)}")
                            except Exception:
                                pass

                        list_lengths = field_info.get("list_lengths", [])
                        if list_lengths:
                            try:
                                parts.append(
                                    f"avg_list_len={_format_number(mean(list_lengths))}"
                                )
                            except Exception:
                                pass

                        string_values = field_info.get("string_values", Counter())
                        if string_values:
                            top_entries = []
                            for value, count in string_values.most_common(3):
                                top_entries.append(f"{_safe_preview(value, 32)}({count})")
                            parts.append(f"top_values={', '.join(top_entries)}")
                        else:
                            samples = field_info.get("samples", [])
                            if samples:
                                parts.append("samples=" + ", ".join(samples))

                        summary_lines.append(f"  - {field}: {'; '.join(parts)}")
                else:
                    summary_lines.append("  (No sample documents found)")

                summary_block = "\n".join(summary_lines)
                summary_blocks.append(summary_block)
                collection_summaries[coll_name] = summary_block

            schema_text = "\n\n".join(schema_blocks).strip()
            summary_text = "\n\n".join(summary_blocks).strip()

            return {
                "schema_text": schema_text,
                "collection_summaries": collection_summaries,
                "collection_summary_text": summary_text,
                "collection_fields": collection_fields,
                "collection_stats": collection_stats,
            }
        finally:
            client.close()

    def post(self, shared, prep_res, exec_res):
        if isinstance(exec_res, dict):
            schema_text = exec_res.get("schema_text", "")
            summary_text = exec_res.get("collection_summary_text", "")
            shared["schema"] = schema_text
            shared["collection_summary_text"] = summary_text
            shared["collection_summaries"] = exec_res.get("collection_summaries", {})
            shared["collection_field_map"] = exec_res.get("collection_fields", {})
            shared["collection_stats"] = exec_res.get("collection_stats", {})
            shared["collection_names"] = list(shared["collection_field_map"].keys())
        else:
            # Backward compatibility if exec returned a string
            schema_text = str(exec_res)
            summary_text = ""
            shared["schema"] = schema_text
            shared["collection_summary_text"] = summary_text
            shared["collection_summaries"] = {}
            shared["collection_field_map"] = {}
            shared["collection_stats"] = {}
            shared["collection_names"] = []

        print("\n===== DB SCHEMA =====\n")
        print(schema_text or "(No schema information available)")
        if summary_text:
            print("\n===== COLLECTION SUMMARIES =====\n")
            print(summary_text)
        print("\n=====================\n")

class GenerateSQL(Node):
    def prep(self, shared):
        prepared = {
            "natural_query": shared["natural_query"],
            "schema": shared["schema"],
            "collection_summary": shared.get("collection_summary_text"),
            "api_key": shared.get("api_key"),
            "llm_provider": shared.get("llm_provider"),
            "llm_model_name": shared.get("llm_model_name"),
            "llm_base_url": shared.get("llm_base_url"),
            "thinking": shared.get("thinking"),
            "instruction": shared.get("instruction"),
        }
        return prepared

    def exec(self, prepared):
        natural_query = prepared["natural_query"]
        schema = prepared["schema"]
        collection_summary = prepared.get("collection_summary") or ""
        api_key = prepared["api_key"]
        provider = prepared["llm_provider"]
        model_name = prepared["llm_model_name"]
        base_url = prepared["llm_base_url"]
        thinking_raw = prepared["thinking"]
        instruction = prepared["instruction"]

        # Build instruction section
        instruction_section = ""
        if instruction:
            instruction_section = f"\nInstructions:\n{instruction}\n\nProcess this step by step:\n1. Analyze the question and schema\n2. Identify the appropriate collection and query type\n3. Build the MongoDB query\n4. Format as YAML\n"

        summary_section = ""
        if collection_summary:
            summary_section = f"Database Overview:\n{collection_summary}\n\n"

        prompt = (
            f"You are a MongoDB expert. Translate the following natural language question into a MongoDB query for data retrieval.\n\n"
            f"MongoDB Schema:\n{schema}\n\n"
            f"{summary_section}"
            f"Question: {natural_query}\n"
            f"{instruction_section}\n"
            f"This system is for QUERYING/SEARCHING data only. Do not generate insert, update, or delete operations.\n\n"
            f"Respond ONLY with the YAML block in this format:\n"
            f"```yaml\n"
            f"collection: collection_name\n"
            f"query_type: find|aggregate|count_documents|distinct\n"
            f"query:\n"
            f"  # For find: filter document\n"
            f"  # For aggregate: pipeline array\n"
            f"  # For count_documents: filter document\n"
            f"  # For distinct: {{field: 'field_name', filter: {{}}}}\n"
            f"```\n"
        )
        print(f"Prompt Insert : {prompt}")
        provider = (provider or "openai").lower()
        # Normalize thinking flag: Gemini expects 0/1, OpenAI expects True/False
        if provider == "gemini":
            try:
                gemini_thinking = int(thinking_raw)
            except Exception:
                gemini_thinking = 0
            print(f"Thinking : {gemini_thinking}")
            llm_response = call_llm_gemini(
                prompt,
                model_name=model_name,
                thinking_budget=gemini_thinking,
                api_key=api_key,
            )
        else:
            openai_thinking = str(thinking_raw)
            llm_response = call_llm_openai(
                prompt,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                thinking=openai_thinking,
            )
            print(f"Thinking : {openai_thinking}")

        # Extract content and usage
        text_response = llm_response["content"]
        token_usage = llm_response["usage"]

        # Store token usage and response info in prepared state
        prepared["token_usage"] = token_usage
        prepared["model_response"] = text_response
        prepared["response_length"] = len(text_response)

        # Robust YAML extraction: handle responses without fenced ```yaml blocks
        llm_text = text_response.strip()
        yaml_str = None
        if "```yaml" in llm_text:
            yaml_str = llm_text.split("```yaml", 1)[1].split("```", 1)[0].strip()
        elif "```" in llm_text:
            # Fallback: use first fenced block even if language tag missing
            parts = llm_text.split("```", 1)
            if len(parts) == 2:
                yaml_str = parts[1].split("```", 1)[0].strip()
        if yaml_str is None:
            # Final fallback: try to parse the entire response as YAML
            yaml_str = llm_text.strip("`").strip()

        try:
            structured_result = yaml.safe_load(yaml_str)
        except Exception as e:
            print("[ERROR] Failed to parse YAML:")
            print("YAML string:", yaml_str)
            print("Error:", e)
            # Fallback to a structured error
            return None, None, {"error": "Failed to parse LLM response as YAML", "details": str(e)}

        collection = structured_result.get("collection")
        query_type = structured_result.get("query_type", "aggregate")
        query = structured_result.get("query") or structured_result.get("mongo_query")

        if query is None:
            print("[ERROR] No 'query' or 'mongo_query' key in LLM response")
            print("Structured result:", structured_result)
            raise KeyError("Missing query in LLM response")

        if isinstance(query, str):
            try:
                query = yaml.safe_load(query)
            except Exception as e:
                print("[ERROR] Failed to parse query string as YAML")
                print("Query string:", query)
                print("Error:", e)
                raise e

        return (collection, query_type, query)

    def post(self, shared, prepared, exec_res):
        """Store the generated query tuple and model response in shared for downstream nodes."""
        collection, query_type, query = exec_res
        shared["generated_query"] = (collection, query_type, query)
        # Store token usage and response info in shared state
        shared["token_usage"] = prepared.get("token_usage", {})
        shared["model_response"] = prepared.get("model_response", "")
        shared["response_length"] = prepared.get("response_length", 0)
        return None

class ValidateSQL(Node):
    ALLOWED_QUERY_TYPES = {"find", "aggregate", "count_documents", "distinct"}

    def prep(self, shared):
        return {
            "natural_query": shared.get("natural_query"),
            "schema": shared.get("schema"),
            "collection_summary": shared.get("collection_summary_text"),
            "collection_fields": shared.get("collection_field_map"),
            "generated_query": shared.get("generated_query"),
            "api_key": shared.get("api_key"),
            "llm_provider": shared.get("llm_provider"),
            "llm_model_name": shared.get("llm_model_name"),
            "llm_base_url": shared.get("llm_base_url"),
            "thinking": shared.get("thinking"),
            "instruction": shared.get("instruction"),
        }

    def _run_llm_validation(self, prompt, provider, model_name, api_key, base_url, thinking_raw):
        provider = (provider or "openai").lower()
        if provider == "gemini":
            try:
                gemini_thinking = int(thinking_raw)
            except Exception:
                gemini_thinking = 0
            response = call_llm_gemini(
                prompt,
                model_name=model_name,
                thinking_budget=gemini_thinking,
                api_key=api_key,
            )
        else:
            openai_thinking = str(thinking_raw)
            response = call_llm_openai(
                prompt,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                thinking=openai_thinking,
            )
        return response

    def exec(self, prepared):
        generated_query = prepared.get("generated_query")
        if not generated_query:
            raise ValueError("No generated query available for validation")

        natural_query = prepared.get("natural_query")
        schema = prepared.get("schema") or ""
        collection_summary = prepared.get("collection_summary") or ""
        collection_fields = prepared.get("collection_fields") or {}
        api_key = prepared.get("api_key")
        provider = prepared.get("llm_provider")
        model_name = prepared.get("llm_model_name")
        base_url = prepared.get("llm_base_url")
        thinking_raw = prepared.get("thinking")
        instruction = prepared.get("instruction")

        collection, query_type, query = generated_query

        precheck_issues = []
        if not collection:
            precheck_issues.append("Missing collection name in generated query.")
        elif collection_fields and collection not in collection_fields:
            known = ", ".join(sorted(collection_fields.keys())) or "unknown"
            precheck_issues.append(
                f"Collection '{collection}' not found. Known collections: {known}."
            )

        if not query_type:
            precheck_issues.append("Missing query_type in generated query.")
        elif query_type not in self.ALLOWED_QUERY_TYPES:
            allowed = ", ".join(sorted(self.ALLOWED_QUERY_TYPES))
            precheck_issues.append(
                f"Query type '{query_type}' is not allowed. Use one of: {allowed}."
            )

        if query is None:
            precheck_issues.append("Generated query payload is empty.")
        else:
            if query_type == "find" and not isinstance(query, dict):
                precheck_issues.append("find queries must use a filter document (dict).")
            elif query_type == "aggregate" and not isinstance(query, list):
                precheck_issues.append("aggregate queries must be a pipeline list.")
            elif query_type == "count_documents" and not isinstance(query, dict):
                precheck_issues.append("count_documents queries must use a filter document (dict).")
            elif query_type == "distinct":
                if not isinstance(query, dict):
                    precheck_issues.append(
                        "distinct queries must use a dict with 'field' and optional 'filter'."
                    )
                else:
                    if "field" not in query:
                        precheck_issues.append("distinct queries must include a 'field'.")

        proposed_yaml = yaml.safe_dump(
            {
                "collection": collection,
                "query_type": query_type,
                "query": query,
            },
            sort_keys=False,
            allow_unicode=False,
        )

        instruction_section = ""
        if instruction:
            instruction_section = (
                "\nValidator instructions:\n"
                f"{instruction}\n"
                "Always double-check safety, relevance, and schema alignment.\n"
            )

        precheck_section = ""
        if precheck_issues:
            precheck_section = "Known static issues so far:\n" + "\n".join(
                f"- {issue}" for issue in precheck_issues
            ) + "\n\n"

        prompt = (
            "You are a MongoDB query validator. Review the proposed query to ensure it is safe, "
            "matches the user's request, and aligns with the database schema. Return a clear decision.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Database Overview:\n{collection_summary}\n\n"
            f"User Question: {natural_query}\n\n"
            f"Proposed Query (YAML):\n```yaml\n{proposed_yaml}```\n\n"
            "Checklist:\n"
            "- Confirm the collection exists and fits the request.\n"
            "- Confirm fields match the schema and data distribution.\n"
            "- Confirm the query type is appropriate.\n"
            "- Flag missing projections, filters, or aggregation stages.\n"
            "- Reject unsafe operations (writes, deletes, updates).\n"
            "- Provide concise guidance if changes are needed.\n\n"
            f"{precheck_section}"
            f"{instruction_section}"
            "Respond ONLY with YAML in this format. IMPORTANT: The values for 'reason', 'message', and 'suggestion' MUST be enclosed in double quotes.\n"
            "decision: pass|fail\n"
            'reason: "short sentence"\n'
            "confidence: number between 0 and 1\n"
            "issues:\n"
            '  - message: "description of a problem"\n'
            "    severity: high|medium|low\n"
            '    suggestion: "how to fix it"\n'
        )

        llm_response = None
        token_usage = {}
        text_response = ""
        if api_key and provider:
            llm_response = self._run_llm_validation(
                prompt,
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                thinking_raw=thinking_raw,
            )
            text_response = llm_response.get("content", "")
            token_usage = llm_response.get("usage", {})
        else:
            text_response = "(validation LLM not executed: missing provider or API key)"

        structured_result = {}
        if text_response:
            llm_text = text_response.strip()
            yaml_str = None
            if "```yaml" in llm_text:
                yaml_str = llm_text.split("```yaml", 1)[1].split("```", 1)[0].strip()
            elif "```" in llm_text:
                parts = llm_text.split("```", 1)
                if len(parts) == 2:
                    yaml_str = parts[1].split("```", 1)[0].strip()
            else:
                yaml_str = llm_text.strip("`").strip()

            if yaml_str:
                try:
                    structured = yaml.safe_load(yaml_str)
                    if isinstance(structured, dict):
                        structured_result = structured
                except Exception as exc:
                    print("[ERROR] Failed to parse validator YAML:")
                    print("YAML string:", yaml_str)
                    print("Error:", exc)
                    # Fallback to regex parsing
                    decision_match = re.search(r"decision:\s*(pass|fail)", yaml_str, re.IGNORECASE)
                    decision = decision_match.group(1).lower() if decision_match else 'fail'
                    reason_match = re.search(r"reason:\s*(.*)", yaml_str)
                    reason = reason_match.group(1).strip() if reason_match else "YAML parsing failed."
                    structured_result = {
                        "decision": decision,
                        "reason": reason,
                        "issues": [{"message": f"Failed to parse YAML from LLM: {exc}", "severity": "high", "suggestion": "Check the LLM response in the logs."}]
                    }

        decision_raw = (structured_result.get("decision") if isinstance(structured_result, dict) else None) or "fail"
        decision = str(decision_raw).strip().lower()
        if decision in {"pass", "approve", "approved", "valid"}:
            decision = "pass"
        else:
            decision = "fail"

        reason = structured_result.get("reason") if isinstance(structured_result, dict) else None
        if not reason:
            if decision == "pass" and not precheck_issues:
                reason = "Validator approved the query."
            else:
                reason = "Validator identified issues with the query."

        issues_data = structured_result.get("issues") if isinstance(structured_result, dict) else None
        parsed_issues = []
        if isinstance(issues_data, list):
            for item in issues_data:
                if isinstance(item, dict):
                    parsed_issues.append(
                        {
                            "message": item.get("message") or item.get("detail") or str(item),
                            "severity": (item.get("severity") or "medium").lower(),
                            "suggestion": item.get("suggestion") or item.get("fix"),
                        }
                    )
                else:
                    parsed_issues.append(
                        {
                            "message": str(item),
                            "severity": "medium",
                            "suggestion": None,
                        }
                    )

        for issue in precheck_issues:
            parsed_issues.insert(
                0,
                {
                    "message": issue,
                    "severity": "high",
                    "suggestion": "Adjust the query to satisfy this constraint before execution.",
                },
            )

        if precheck_issues:
            decision = "fail"

        confidence = structured_result.get("confidence") if isinstance(structured_result, dict) else None

        return {
            "decision": decision,
            "reason": reason,
            "issues": parsed_issues,
            "confidence": confidence,
            "token_usage": token_usage,
            "response_text": text_response,
            "prompt": prompt,
        }

    def post(self, shared, prepared, exec_res):
        decision = exec_res.get("decision")
        issues = exec_res.get("issues", [])
        reason = exec_res.get("reason", "")

        shared.setdefault("validation_responses", []).append(exec_res.get("response_text", ""))
        shared.setdefault("validation_token_usage", []).append(exec_res.get("token_usage", {}))
        shared.setdefault("validation_attempts", 0)
        shared["validation_attempts"] += 1

        blocking_levels = {"high", "medium", "blocker"}
        blocking_issues = [
            issue
            for issue in issues
            if (issue.get("severity") or "medium").lower() in blocking_levels
        ]

        if decision == "pass" and not blocking_issues:
            shared["validation_passed"] = True
            shared["retry_cause"] = None
            shared["retry_feedback"] = None
            shared["validation_feedback"] = None
            return None

        shared["validation_passed"] = False
        failure_feedback = {
            "message": reason,
            "cause": "validation_fail",
            "issues": issues,
        }

        shared["validation_feedback"] = failure_feedback
        shared["retry_cause"] = "validation_fail"
        shared["retry_feedback"] = failure_feedback
        shared.setdefault("retry_history", []).append(failure_feedback)
        shared["rewrite_attempts"] = shared.get("rewrite_attempts", 0) + 1

        print("\n===== QUERY VALIDATION FAILED =====\n")
        print(reason)
        if issues:
            for idx, issue in enumerate(issues, start=1):
                message = issue.get("message")
                severity = issue.get("severity")
                suggestion = issue.get("suggestion")
                print(f"{idx}. [{severity}] {message}")
                if suggestion:
                    print(f"   Suggestion: {suggestion}")
        else:
            print("No additional issues reported.")
        print("===================================\n")

        return "rewrite_retry"

class ExecuteSQL(Node):
    def prep(self, shared):
        return shared["mongo_uri"], shared["db_name"], shared["generated_query"]

    def exec(self, prep_res):
        mongo_uri, db_name, generated_query = prep_res
        collection, query_type, query = generated_query

        try:
            client = pymongo.MongoClient(mongo_uri)
            db = client[db_name]
            start_time = time.time()

            # Execute based on query type (read operations only)
            if query_type == "find":
                results = list(db[collection].find(query))
            elif query_type == "aggregate":
                results = list(db[collection].aggregate(query))
            elif query_type == "count_documents":
                count = db[collection].count_documents(query)
                results = [{"count": count}]
            elif query_type == "distinct":
                field = query.get("field")
                filter_doc = query.get("filter", {})
                distinct_values = db[collection].distinct(field, filter_doc)
                results = [{"distinct_values": distinct_values}]
            else:
                raise ValueError(f"Unsupported query_type for read operations: {query_type}")

            client.close()
            duration = time.time() - start_time

            empty_result = results is None or (isinstance(results, list) and len(results) == 0)
            if empty_result:
                feedback = {
                    "message": (
                        "Query executed successfully but returned no results; "
                        "rewrite the query to retrieve relevant documents."
                    ),
                    "cause": "empty_result",
                    "collection": collection,
                    "query_type": query_type,
                    "query": query,
                    "duration_s": duration,
                }
                print(f"MongoDB {query_type} executed in {duration:.3f} seconds but returned no results.")
                return (False, feedback, [])

            print(f"MongoDB {query_type} executed in {duration:.3f} seconds.")

            # Remove _id for display
            for doc in results:
                if isinstance(doc, dict):
                    doc.pop("_id", None)

            column_names = list(results[0].keys()) if results else []
            return (True, results, column_names)

        except Exception as e:
            print(f"MongoDB Error during execution: {e}")
            if 'client' in locals():
                try:
                    client.close()
                except:
                    pass
            error_feedback = {
                "message": str(e),
                "cause": "execution_error",
                "collection": collection,
                "query_type": query_type,
                "query": query,
                "error_type": type(e).__name__,
            }
            return (False, error_feedback, [])

    def post(self, shared, prep_res, exec_res):
        success, result_or_error, column_names = exec_res

        if success:
            shared["final_result"] = result_or_error
            shared["result_columns"] = column_names
            shared["execution_error"] = None
            shared["retry_cause"] = None
            shared["retry_feedback"] = None
            print("\n===== MONGODB QUERY SUCCESS =====\n")

            if isinstance(result_or_error, list):
                if column_names:
                    print(" | ".join(column_names))
                    print("-" * (sum(len(str(c)) for c in column_names) + 3 * (len(column_names) - 1)))

                if not result_or_error:
                    print("(No results found)")
                else:
                    for row in result_or_error:
                        if isinstance(row, dict):
                            print(" | ".join(str(row.get(col, '')) for col in column_names))
                        else:
                            print(str(row))
            else:
                print(result_or_error)
            print("\n=================================\n")
            return
        else:
            if isinstance(result_or_error, dict):
                error_message = result_or_error.get("message", "")
                retry_cause = result_or_error.get("cause", "unknown")
            else:
                error_message = str(result_or_error)
                retry_cause = "unknown"
                result_or_error = {"message": error_message, "cause": retry_cause}

            shared["execution_error"] = error_message
            shared["retry_cause"] = retry_cause
            shared["retry_feedback"] = result_or_error
            shared.setdefault("retry_history", []).append(result_or_error)
            shared["debug_attempts"] = shared.get("debug_attempts", 0) + 1
            max_attempts = shared.get("max_debug_attempts", 5)

            print(f"\n===== MONGODB QUERY FAILED (Attempt {shared['debug_attempts']}) =====\n")
            print(f"Details: {error_message}")
            if retry_cause == "empty_result":
                print("Cause: Query returned no results.")
            elif retry_cause != "execution_error":
                print(f"Cause: {retry_cause}")
            print("=========================================\n")

            if shared["debug_attempts"] >= max_attempts:
                print(f"Max debug attempts ({max_attempts}) reached. Stopping.")
                shared["final_error"] = (
                    f"Failed to execute MongoDB query after {max_attempts} attempts. Last issue: {error_message}"
                )
                return
            else:
                if retry_cause == "empty_result":
                    print("Attempting to rewrite the MongoDB query based on empty results...")
                    shared["rewrite_attempts"] = shared.get("rewrite_attempts", 0) + 1
                    return "rewrite_retry"
                else:
                    print("Attempting to debug the MongoDB query...")
                    return "error_retry"

class RewriteSQL(Node):
    def prep(self, shared):
        prepared = {
            "natural_query": shared.get("natural_query"),
            "schema": shared.get("schema"),
            "collection_summary": shared.get("collection_summary_text"),
            "generated_query": shared.get("generated_query"),
            "retry_feedback": shared.get("retry_feedback"),
            "api_key": shared.get("api_key"),
            "llm_provider": shared.get("llm_provider"),
            "llm_model_name": shared.get("llm_model_name"),
            "llm_base_url": shared.get("llm_base_url"),
            "thinking": shared.get("thinking"),
            "instruction": shared.get("instruction"),
        }
        return prepared

    def exec(self, prepared):
        natural_query = prepared["natural_query"]
        schema = prepared["schema"]
        collection_summary = prepared.get("collection_summary") or ""
        generated_query = prepared["generated_query"]
        feedback = prepared.get("retry_feedback") or {}
        api_key = prepared["api_key"]
        provider = prepared["llm_provider"]
        model_name = prepared["llm_model_name"]
        base_url = prepared["llm_base_url"]
        thinking_raw = prepared["thinking"]
        instruction = prepared["instruction"]

        if generated_query:
            collection, query_type, previous_query = generated_query
        else:
            collection, query_type, previous_query = None, None, None

        import json
        if previous_query:
            try:
                previous_query_str = json.dumps(previous_query, indent=2)
            except Exception:
                previous_query_str = str(previous_query)
        else:
            previous_query_str = "{}"

        cause = feedback.get("cause", "unknown")
        feedback_message = feedback.get("message", "")

        instruction_section = ""
        if instruction:
            instruction_section = (
                f"\nInstructions:\n{instruction}\n\nProcess this step by step:\n"
                "1. Diagnose why the previous query failed to produce useful results\n"
                "2. Adjust the collection, filters, and projection to answer the question\n"
                "3. Ensure the query aligns with the schema\n"
                "4. Produce the final MongoDB query in YAML format\n"
            )

        if cause == "empty_result":
            cause_message = feedback_message or (
                "The previous query executed but returned no documents. "
                "Adjust filters or broaden the search."
            )
        else:
            cause_message = feedback_message or (
                "The previous query failed. Provide an improved version that will succeed."
            )

        issues = feedback.get("issues")
        issues_section = ""
        if isinstance(issues, list) and issues:
            formatted = []
            for idx, issue in enumerate(issues, start=1):
                message = issue.get("message") if isinstance(issue, dict) else str(issue)
                severity = issue.get("severity") if isinstance(issue, dict) else None
                suggestion = issue.get("suggestion") if isinstance(issue, dict) else None
                parts = [f"{idx}. {message}"]
                if severity:
                    parts.append(f"severity={severity}")
                if suggestion:
                    parts.append(f"suggestion={suggestion}")
                formatted.append(" | ".join(parts))
            issues_section = "Specific issues reported:\n" + "\n".join(formatted) + "\n\n"

        prompt = (
            "You are a MongoDB expert. Rewrite the previous MongoDB query so it better answers the user's request.\n\n"
            f"MongoDB Schema:\n{schema}\n\n"
            f"Database Overview:\n{collection_summary}\n\n"
            f"Original Question: {natural_query}\n\n"
            f"Issue Summary: {cause_message}\n\n"
            f"{issues_section}"
            "Previous Query:\n"
            f"Collection: {collection}\n"
            f"Type: {query_type}\n"
            f"Query: {previous_query_str}\n\n"
            f"{instruction_section}"
            "Respond ONLY with the YAML block in this format:\n"
            "```yaml\n"
            "collection: collection_name\n"
            "query_type: find|aggregate|count_documents|distinct\n"
            "query:\n"
            "  # For find: filter document\n"
            "  # For aggregate: pipeline array\n"
            "  # For count_documents: filter document\n"
            "  # For distinct: {field: 'field_name', filter: {}}\n"
            "```\n"
        )

        provider = (provider or "openai").lower()
        if provider == "gemini":
            try:
                gemini_thinking = int(thinking_raw)
            except Exception:
                gemini_thinking = 0
            print(f"Thinking : {gemini_thinking}")
            llm_response = call_llm_gemini(
                prompt,
                model_name=model_name,
                thinking_budget=gemini_thinking,
                api_key=api_key,
            )
        else:
            openai_thinking = str(thinking_raw)
            llm_response = call_llm_openai(
                prompt,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                thinking=openai_thinking,
            )
            print(f"Thinking : {openai_thinking}")

        text_response = llm_response["content"]
        rewrite_token_usage = llm_response["usage"]

        prepared["rewrite_token_usage"] = rewrite_token_usage
        prepared["rewrite_response"] = text_response

        llm_text = text_response.strip()
        yaml_str = None
        if "```yaml" in llm_text:
            yaml_str = llm_text.split("```yaml", 1)[1].split("```", 1)[0].strip()
        elif "```" in llm_text:
            parts = llm_text.split("```", 1)
            if len(parts) == 2:
                yaml_str = parts[1].split("```", 1)[0].strip()
        if yaml_str is None:
            yaml_str = llm_text.strip("`").strip()

        try:
            structured_result = yaml.safe_load(yaml_str)
        except Exception as e:
            print("[ERROR] Failed to parse rewrite YAML:")
            print("YAML string:", yaml_str)
            print("LLM Response:", text_response)
            print("Error:", e)
            return None, None, {"error": "Failed to parse LLM rewrite response as YAML", "details": str(e)}

        collection = structured_result.get("collection")
        query_type = structured_result.get("query_type", "aggregate")
        query = structured_result.get("query") or structured_result.get("mongo_query")

        if query is None:
            print("[ERROR] No query in rewrite response")
            print("Structured result:", structured_result)
            raise KeyError("Missing query in rewrite response")

        if isinstance(query, str):
            try:
                query = yaml.safe_load(query)
            except Exception as e:
                print("[ERROR] Failed to parse rewrite query string")
                print("Query string:", query)
                raise e

        return (collection, query_type, query)

    def post(self, shared, prepared, exec_res):
        collection, query_type, query = exec_res
        shared["generated_query"] = (collection, query_type, query)

        if "rewrite_responses" not in shared:
            shared["rewrite_responses"] = []
        if "rewrite_response_lengths" not in shared:
            shared["rewrite_response_lengths"] = []
        if "rewrite_token_usage" not in shared:
            shared["rewrite_token_usage"] = []

        rewrite_response = prepared.get("rewrite_response", "")
        rewrite_token_usage = prepared.get("rewrite_token_usage", {})

        shared["rewrite_responses"].append(rewrite_response)
        shared["rewrite_response_lengths"].append(len(rewrite_response))
        shared["rewrite_token_usage"].append(rewrite_token_usage)

        return None

class DebugSQL(Node):
    def prep(self, shared):
        prepared = {
            "natural_query": shared.get("natural_query"),
            "schema": shared.get("schema"),
            "collection_summary": shared.get("collection_summary_text"),
            "generated_query": shared.get("generated_query"),
            "execution_error": shared.get("execution_error"),
            "retry_feedback": shared.get("retry_feedback"),
            "api_key": shared.get("api_key"),
            "llm_provider": shared.get("llm_provider"),
            "llm_model_name": shared.get("llm_model_name"),
            "llm_base_url": shared.get("llm_base_url"),
            "thinking": shared.get("thinking"),
            "instruction": shared.get("instruction"),
        }
        return prepared

    def exec(self, prepared):
        natural_query = prepared["natural_query"]
        schema = prepared["schema"]
        collection_summary = prepared.get("collection_summary") or ""
        generated_query = prepared["generated_query"]
        error_message = prepared["execution_error"]
        api_key = prepared["api_key"]
        provider = prepared["llm_provider"]
        model_name = prepared["llm_model_name"]
        base_url = prepared["llm_base_url"]
        thinking_raw = prepared["thinking"]
        instruction = prepared["instruction"]

        if generated_query:
            collection, query_type, failed_query = generated_query
        else:
            collection, query_type, failed_query = None, None, None

        import json
        if failed_query:
            try:
                failed_query_str = json.dumps(failed_query, indent=2)
            except Exception:
                failed_query_str = str(failed_query)
        else:
            failed_query_str = "{}"

        # Build instruction section for debug
        instruction_section = ""
        if instruction:
            instruction_section = f"\nInstructions:\n{instruction}\n\nProcess this step by step:\n1. Analyze the error message\n2. Identify the specific issue in the query\n3. Apply the correct MongoDB syntax\n4. Ensure the query matches the schema\n5. Format as YAML\n"

        prompt = (
            f"You are a MongoDB expert. The following MongoDB query failed to execute. Please correct it and return the fixed query in YAML format.\n\n"
            f"MongoDB Schema:\n{schema}\n\n"
            f"Database Overview:\n{collection_summary}\n\n"
            f"Original Question: {natural_query}\n\n"
            f"Failed Query:\n"
            f"Collection: {collection}\n"
            f"Type: {query_type}\n"
            f"Query: {failed_query_str}\n\n"
            f"Error Message: {error_message}\n"
            f"{instruction_section}\n"
            f"This system is for QUERYING/SEARCHING data only. Only use: find, aggregate, count_documents, distinct.\n\n"
            f"Respond ONLY with the corrected YAML block in this format:\n"
            f"```yaml\n"
            f"collection: collection_name\n"
            f"query_type: find|aggregate|count_documents|distinct\n"
            f"query:\n"
            f"  # For find: filter document\n"
            f"  # For aggregate: pipeline array\n"
            f"  # For count_documents: filter document\n"
            f"  # For distinct: {{field: 'field_name', filter: {{}}}}\n"
            f"```\n"
        )

        provider = (provider or "openai").lower()
        if provider == "gemini":
            try:
                gemini_thinking = int(thinking_raw)
            except Exception:
                gemini_thinking = 0
            print(f"Thinking : {gemini_thinking}")
            llm_response = call_llm_gemini(
                prompt,
                model_name=model_name,
                thinking_budget=gemini_thinking,
                api_key=api_key,
            )
        else:
            openai_thinking = str(thinking_raw)
            llm_response = call_llm_openai(
                prompt,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                thinking=openai_thinking,
            )
            print(f"Thinking : {openai_thinking}")

        # Extract content and usage
        text_response = llm_response["content"]
        debug_token_usage = llm_response["usage"]

        # Store debug token usage in prepared state
        prepared["debug_token_usage"] = debug_token_usage
        prepared["debug_response"] = text_response

        # Robust YAML extraction for debug as well
        llm_text = text_response.strip()
        yaml_str = None
        if "```yaml" in llm_text:
            yaml_str = llm_text.split("```yaml", 1)[1].split("```", 1)[0].strip()
        elif "```" in llm_text:
            parts = llm_text.split("```", 1)
            if len(parts) == 2:
                yaml_str = parts[1].split("```", 1)[0].strip()
        if yaml_str is None:
            yaml_str = llm_text.strip("`").strip()

        try:
            structured_result = yaml.safe_load(yaml_str)
        except Exception as e:
            print("[ERROR] Failed to parse debug YAML:")
            print("YAML string:", yaml_str)
            print("LLM Response:", text_response)
            print("Error:", e)
            return None, None, {"error": "Failed to parse LLM debug response as YAML", "details": str(e)}

        collection = structured_result.get("collection")
        query_type = structured_result.get("query_type", "aggregate")
        query = structured_result.get("query") or structured_result.get("mongo_query")

        if query is None:
            print("[ERROR] No query in debug response")
            print("Structured result:", structured_result)
            raise KeyError("Missing query in debug response")

        if isinstance(query, str):
            try:
                query = yaml.safe_load(query)
            except Exception as e:
                print("[ERROR] Failed to parse debug query string")
                print("Query string:", query)
                raise e

        return (collection, query_type, query)

    def post(self, shared, prepared, exec_res):
        """Update the generated query after debugging for re-execution."""
        collection, query_type, query = exec_res
        shared["generated_query"] = (collection, query_type, query)
        
        # Store debug responses as lists to track multiple attempts
        if "debug_responses" not in shared:
            shared["debug_responses"] = []
        if "debug_response_lengths" not in shared:
            shared["debug_response_lengths"] = []
        if "debug_token_usage" not in shared:
            shared["debug_token_usage"] = []
            
        debug_response = prepared.get("debug_response", "")
        debug_token_usage = prepared.get("debug_token_usage", {})
        
        shared["debug_responses"].append(debug_response)
        shared["debug_response_lengths"].append(len(debug_response))
        shared["debug_token_usage"].append(debug_token_usage)
        
        return None


class _NoOpAgent(Node):
    """Fallback agent used to keep the API wiring compatible with the legacy flow."""

    summary_key: str = ""
    response_key: str = ""
    token_key: str = ""

    def prep(self, shared):
        return shared

    def exec(self, prepared):
        return None

    def post(self, shared, prepared, _exec_res):
        if self.summary_key and self.summary_key not in shared:
            shared[self.summary_key] = None
        if self.response_key and self.response_key not in shared:
            shared[self.response_key] = None
        if self.token_key and self.token_key not in shared:
            shared[self.token_key] = []
        return None


class SchemaLinkAgent(_NoOpAgent):
    summary_key = "schema_link_summary"
    response_key = "schema_link_response"
    token_key = "schema_link_token_usage"

    def post(self, shared, prepared, exec_res):
        super().post(shared, prepared, exec_res)
        if "schema_linking" not in shared:
            shared["schema_linking"] = None
        return None


class SubproblemAgent(_NoOpAgent):
    summary_key = "subproblem_summary"
    response_key = "subproblem_response"
    token_key = "subproblem_token_usage"


class QueryPlanAgent(_NoOpAgent):
    summary_key = "query_plan_summary"
    response_key = "query_plan_response"
    token_key = "query_plan_token_usage"


class CorrectionPlanAgent(_NoOpAgent):
    summary_key = "correction_plan_summary"
    response_key = "correction_plan_response"
    token_key = "correction_plan_token_usage"


class CorrectionMongoAgent(_NoOpAgent):
    summary_key = "correction_sql_summary"
    response_key = "correction_sql_response"
    token_key = "correction_sql_token_usage"
