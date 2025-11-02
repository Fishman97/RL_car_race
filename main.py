import argparse
from src.runner import Runner


def main():
    parser = argparse.ArgumentParser(description="Car Racing Environment Runner")
    parser.add_argument(
        "--agent-type",
        type=str,
        default="manual",
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
        "--map",
        type=str,
        default="assets/maps/track1.tmx",
        help="Path to the map file (default: assets/maps/track1.tmx)"
    )
    parser.add_argument(
        "--tileset",
        type=str,
        default="assets/tilesets/flat_race.tsx",
        help="Path to the tileset file (default: assets/tilesets/flat_race.tsx)"
    )
    
    args = parser.parse_args()
    
    # Convert 'none' string to None for render_mode
    render_mode = None if args.render_mode == "none" else args.render_mode
    
    runner = Runner(
        map_path=args.map,
        tileset_path=args.tileset,
        agent_type=args.agent_type,
        render_mode=render_mode
    )
    
    runner.run(print_interval=args.print_interval)


if __name__ == "__main__":
    main()
