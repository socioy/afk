"""Tooling for interactive policy-gated execution."""

from pydantic import BaseModel

from afk.tools import tool


class EchoArgs(BaseModel):
    text: str


@tool(args_model=EchoArgs, name="echo_change_plan")
def echo_change_plan(args: EchoArgs) -> dict[str, str]:
    return {"approved_plan": args.text}
