from pathlib import Path
from typing import List, Optional
from src.runner import Runner

DEMO_MAX_STEPS = 500

def _discover_demo_models(demo_dir: Path) -> List[Path]:
    if not demo_dir.exists():
        return []
    return sorted([path for path in demo_dir.glob("*.pt") if path.is_file()])


def run_demo(*, map_path: str, tileset_path: str, render_mode: Optional[str], print_interval: int, demo_dir: str) -> None:
    demo_path = Path(demo_dir)
    model_paths = _discover_demo_models(demo_path)

    if not model_paths:
        resolved = demo_path.resolve()
        print(f"No demo checkpoints found in {resolved}.")
        return

    if render_mode is None:
        render_mode = "human"

    print(f"Found {len(model_paths)} demo checkpoints in {demo_path}.")

    runner = Runner(map_path=map_path, tileset_path=tileset_path, agent_type="DQN", render_mode=render_mode, max_steps=DEMO_MAX_STEPS, model_path=str(model_paths[0]), demo_mode=True)

    try:
        for index, model_path in enumerate(model_paths):
            if index > 0:
                if not runner.update_model_path(str(model_path)):
                    print(f"Skipping checkpoint '{model_path.name}' (failed to load).")
                    continue

            print(
                f"\n=== Demo model {index + 1} of {len(model_paths)}: {model_path.name} ==="
            )

            while runner.run(print_interval=print_interval):
                pass

            if runner.is_window_closed():
                print("Window closed. Exiting demo mode.")
                return

        print("\nDemo complete. Displayed all checkpoints once.")
    finally:
        runner.close()
