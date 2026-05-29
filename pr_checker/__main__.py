import sys

import uvicorn
import yaml

from pr_checker.config import ServerConfig
from pr_checker.reviewer_config import ReviewerConfig


def main() -> None:
    if "--print-config" in sys.argv:
        print(yaml.dump(ReviewerConfig().model_dump(), sort_keys=False, allow_unicode=True), end="")
        return

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
