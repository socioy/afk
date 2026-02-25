"""Interactive provider that resolves deferred inputs in-memory."""

from afk.agents import ApprovalDecision, UserInputDecision
from afk.agents.types import DeferredDecision, UserInputRequest
from afk.core import InMemoryInteractiveProvider


class AutoResolveInteractiveProvider(InMemoryInteractiveProvider):
    """Auto-resolve deferred requests for deterministic demo runs."""

    async def request_approval(self, request):  # type: ignore[override]
        deferred = await super().request_approval(request)
        if isinstance(deferred, DeferredDecision):
            self.set_deferred_result(deferred.token, ApprovalDecision(kind="allow"))
        return deferred

    async def request_user_input(  # type: ignore[override]
        self,
        request: UserInputRequest,
    ):
        deferred = await super().request_user_input(request)
        if isinstance(deferred, DeferredDecision):
            self.set_deferred_result(
                deferred.token,
                UserInputDecision(kind="allow", value="change-window:02:00-03:00 UTC"),
            )
        return deferred
