import httpx


class GitHubClient:
    _BASE_URL = "https://api.github.com"

    def __init__(self, token: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._token = token
        self._http = http_client

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._BASE_URL,
                headers={
                    "Authorization": f"token {self._token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=30.0,
            )
        return self._http

    async def post_commit_status(
        self,
        repo_full_name: str,
        sha: str,
        state: str,
        description: str,
        context: str = "pr-checker",
    ) -> None:
        r = await self._client().post(
            f"/repos/{repo_full_name}/statuses/{sha}",
            json={"state": state, "description": description, "context": context},
        )
        r.raise_for_status()

    async def post_pr_comment(self, repo_full_name: str, pr_number: int, body: str) -> None:
        r = await self._client().post(
            f"/repos/{repo_full_name}/issues/{pr_number}/comments",
            json={"body": body},
        )
        r.raise_for_status()

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
