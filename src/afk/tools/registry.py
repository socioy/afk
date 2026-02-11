"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module provides a registry for tools that can be used by the AFK agent. 
"""
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Dict

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Union[str, bytes]]
    id: Optional[str] = None

class ToolRegistry:
    def __init__(self, agent_id: Optional[str] = None):
        self.registry: Dict[str, Tool] = {} 
    
    def re

