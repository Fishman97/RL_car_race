import argparse
from pathlib import Path
from src.runner import Runner
import src.train_dqn as train_dqn

DEFAULT_MAP_PATH = "assets/maps/track1.tmx"
DEFAULT_TILESET_PATH = "assets/tilesets/flat_race.tsx"

def main():
    parser = argparse.ArgumentParser(description="Car Racing Environment Runner")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["play", "train", "train-human"],
        help="Execution mode: 'play' to control/view an agent, 'train' to run headless training, 'train-human' to train while visualizing the latest policy.",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=None,
        choices=["DQN", "manual", "random"],
        help="Type of agent to use (default: manual)"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "none"],
        help="Render mode: 'human' for visualization, 'none' for no rendering (default: human)"
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=300,
        help="Number of steps between status prints (default: 300)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to a saved DQN checkpoint for evaluation"
    )
    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP_PATH,
        help="Path to the map file"
    )
    parser.add_argument(
        "--tileset",
        type=str,
        default=DEFAULT_TILESET_PATH,
        help="Path to the tileset file"
    )
    
    args = parser.parse_args()

    if args.mode is None:
        raise ValueError("Mode must be specified via --mode argument.")
    if args.mode == "play" and args.agent_type is None:
        args.agent_type = "manual"
    if args.agent_type is None:
        raise ValueError("Agent type must be specified via --agent-type argument.")

    if args.mode == "train":
        train_dqn.train(map_path=args.map, tileset_path=args.tileset)
        return

    if args.mode == "train-human":
        train_dqn.train_with_UI(map_path=args.map, tileset_path=args.tileset)
        return
    
    # Convert 'none' string to None for render_mode
    render_mode = None if args.render_mode == "none" else args.render_mode

    if args.agent_type == "DQN" and not args.model_path:
        save_dir = Path(train_dqn.SAVE_DIR)
        best_ckpt = train_dqn.find_latest_checkpoint(save_dir, "dqn_best")
        if best_ckpt:
            args.model_path = str(best_ckpt)
        else:
            print(f"Warning: no best checkpoint found in {save_dir}. Running without a model.")
    
    runner = Runner(map_path=args.map, tileset_path=args.tileset, agent_type=args.agent_type, render_mode=render_mode, model_path=args.model_path, max_steps=train_dqn.MAX_STEPS)

    try:
        while True:
            while runner.run(print_interval=args.print_interval):
                pass
            if runner.is_window_closed():
                break
    finally:
        runner.close()


if __name__ == "__main__":
    main()
