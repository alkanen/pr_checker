import uvicorn

from pr_checker.config import ServerConfig


def main() -> None:
    cfg = ServerConfig()
    uvicorn.run(
        "pr_checker.main:app",
        host="0.0.0.0",
        port=cfg.port,
        root_path=cfg.root_path,
        proxy_headers=True,
        forwarded_allow_ips=cfg.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
