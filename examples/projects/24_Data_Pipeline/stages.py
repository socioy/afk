"""
---
name: Data Pipeline — Stages
description: Pipeline stage agents and their tools for data extraction, validation, transformation, and reporting.
tags: [agent, tools, pipeline, stages]
---
---
Each stage in the data pipeline is a standalone agent with a dedicated tool. The agents are
designed to be composed into a DelegationPlan DAG (see pipeline.py). Simulated data is kept
here so that stages are self-contained. In a real system, each stage would connect to a
database, API, or file system.
---
"""

from pydantic import BaseModel  # <- Pydantic for typed tool argument schemas.
from afk.agents import Agent  # <- Agent defines each pipeline stage.
from afk.tools import tool  # <- @tool decorator for creating tools.


# ===========================================================================
# Simulated data for the pipeline
# ===========================================================================

RAW_DATA: list[dict] = [  # <- Simulated raw data records. In a real pipeline, this comes from a database, API, or file.
    {"id": 1, "name": "Alice", "department": "Engineering", "salary": 95000, "tenure_years": 3},
    {"id": 2, "name": "Bob", "department": "Marketing", "salary": 72000, "tenure_years": 5},
    {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 110000, "tenure_years": 7},
    {"id": 4, "name": "Diana", "department": "Sales", "salary": 68000, "tenure_years": 2},
    {"id": 5, "name": "Eve", "department": "Engineering", "salary": 125000, "tenure_years": 10},
    {"id": 6, "name": "Frank", "department": "Marketing", "salary": 78000, "tenure_years": 4},
    {"id": 7, "name": "Grace", "department": "Sales", "salary": 82000, "tenure_years": 6},
    {"id": 8, "name": "Hank", "department": "Engineering", "salary": 105000, "tenure_years": 5},
]


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# Stage 1: Data extraction
# ===========================================================================

@tool(args_model=EmptyArgs, name="extract_data", description="Extract raw employee data from the source")
def extract_data(args: EmptyArgs) -> str:  # <- Extraction tool: reads simulated data and formats it for the agent.
    records = "\n".join(
        f"  {r['id']}. {r['name']} | {r['department']} | ${r['salary']:,} | {r['tenure_years']}yr"
        for r in RAW_DATA
    )
    return f"Extracted {len(RAW_DATA)} records:\n{records}"


extractor_agent = Agent(  # <- The first stage in the pipeline: extracts raw data.
    name="data-extractor",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data extraction agent. Use the extract_data tool to fetch raw employee records
    from the data source. Present the data clearly with all fields.
    """,
    tools=[extract_data],
)


# ===========================================================================
# Stage 2: Data validation
# ===========================================================================

@tool(args_model=EmptyArgs, name="validate_data", description="Validate data quality — check for missing fields and outliers")
def validate_data(args: EmptyArgs) -> str:  # <- Validation tool: checks each record for data quality issues.
    issues = []
    for r in RAW_DATA:
        if r["salary"] < 0:
            issues.append(f"  Record {r['id']}: negative salary")
        if r["tenure_years"] < 0:
            issues.append(f"  Record {r['id']}: negative tenure")
        if not r["name"]:
            issues.append(f"  Record {r['id']}: missing name")
    if not issues:
        return f"Validation passed: all {len(RAW_DATA)} records are valid."
    return f"Validation found {len(issues)} issues:\n" + "\n".join(issues)


validator_agent = Agent(
    name="data-validator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data validation agent. Use the validate_data tool to check data quality.
    Report any issues found, or confirm that all records pass validation.
    """,
    tools=[validate_data],
)


# ===========================================================================
# Stage 3: Data transformation (depends on extraction + validation)
# ===========================================================================

@tool(args_model=EmptyArgs, name="transform_data", description="Transform and aggregate data by department")
def transform_data(args: EmptyArgs) -> str:  # <- Transformation tool: aggregates records by department with averages.
    dept_stats: dict[str, dict] = {}
    for r in RAW_DATA:
        dept = r["department"]
        if dept not in dept_stats:
            dept_stats[dept] = {"count": 0, "total_salary": 0, "total_tenure": 0}
        dept_stats[dept]["count"] += 1
        dept_stats[dept]["total_salary"] += r["salary"]
        dept_stats[dept]["total_tenure"] += r["tenure_years"]

    lines = []
    for dept, stats in dept_stats.items():
        avg_salary = stats["total_salary"] / stats["count"]
        avg_tenure = stats["total_tenure"] / stats["count"]
        lines.append(f"  {dept}: {stats['count']} employees, avg salary ${avg_salary:,.0f}, avg tenure {avg_tenure:.1f}yr")
    return "Transformation complete — Department aggregates:\n" + "\n".join(lines)


transformer_agent = Agent(
    name="data-transformer",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data transformation agent. Use the transform_data tool to aggregate
    employee data by department. Present clear summaries with averages.
    """,
    tools=[transform_data],
)


# ===========================================================================
# Stage 4: Report generation (depends on transformation)
# ===========================================================================

@tool(args_model=EmptyArgs, name="generate_report", description="Generate a final executive summary report")
def generate_report(args: EmptyArgs) -> str:  # <- Report tool: computes overall statistics and highlights.
    total = len(RAW_DATA)
    total_salary = sum(r["salary"] for r in RAW_DATA)
    avg_salary = total_salary / total
    departments = len(set(r["department"] for r in RAW_DATA))
    top_earner = max(RAW_DATA, key=lambda r: r["salary"])
    most_tenured = max(RAW_DATA, key=lambda r: r["tenure_years"])
    return (
        f"Executive Summary Report\n"
        f"{'=' * 30}\n"
        f"Total employees: {total}\n"
        f"Departments: {departments}\n"
        f"Total payroll: ${total_salary:,}\n"
        f"Average salary: ${avg_salary:,.0f}\n"
        f"Top earner: {top_earner['name']} (${top_earner['salary']:,})\n"
        f"Most tenured: {most_tenured['name']} ({most_tenured['tenure_years']} years)"
    )


reporter_agent = Agent(
    name="report-generator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a report generation agent. Use the generate_report tool to create an
    executive summary. Present it in a professional, clear format.
    """,
    tools=[generate_report],
)
