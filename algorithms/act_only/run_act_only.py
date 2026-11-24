"""Entry point for the Multi-Agent Act-only Overcooked baseline."""

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm_source", "-llm", default="deepseek-v4-flash")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--grid_horizon", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--env_file", type=Path, default=PROJECT_ROOT / ".env")
    args = parser.parse_args()
    if min(args.episodes, args.grid_horizon) < 1:
        parser.error("--episodes and --grid_horizon must be positive")

    sys.path.insert(0, str(PROJECT_ROOT))
    from algorithms.act_only.llm import configure_api, query_action
    from algorithms.act_only.environments.overcooked_ai.adapter import LAYOUT_NAME
    from algorithms.act_only.runner import ActOnlyRunner

    configure_api(args.llm_source, (PROJECT_ROOT / args.env_file).resolve())
    batch = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = (PROJECT_ROOT / args.data_dir / "overcooked_dialog" /
                LAYOUT_NAME / "act_only" / batch / "run_0" / "case_0")
    ActOnlyRunner(args, case_dir, query_action).run()


if __name__ == "__main__":
    main()
