import numpy as np
from src.carRacingEnv import CarRacingEnv
from src.agents import ManualAgent

def main():
    map_path = "assets/maps/track1.tmx"
    tileset_path = "assets/tilesets/flat_race.tsx"

    env = CarRacingEnv(
        map_path=map_path,
        tileset_path=tileset_path,
        render_mode="human"
    )
    
    # Create the manual control agent
    agent = ManualAgent()
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\n=== MANUAL CONTROL ===")
    print("Use arrow keys to control the car:")
    print("  ← Left Arrow:  Steer left")
    print("  → Right Arrow: Steer right")
    print("  ↑ Up Arrow:    Accelerate")
    print("  ↓ Down Arrow:  Brake")
    print("Close the window to exit.")
    print("======================\n")
    
    # Reset environment
    observation, info = env.reset()
    
    running = True
    step = 0
    
    while running:
        # Agent decides action based on keyboard input
        action = agent.act(observation)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render and check if window should close
        running = env.render()
        
        # Print info every 60 steps (1 second at 60 FPS)
        if step % 60 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: {info['position']}")
            print(f"  Velocity: {info['velocity']:.2f}")
            print(f"  Reward: {reward:.2f}")
        
        # Auto-restart when episode ends
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Collision: {info['collision']}")
            print(f"  Finished: {info['finished']}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Restarting...")
            observation, info = env.reset()
            agent.reset()
            step = 0
        else:
            step += 1
    
    env.close()
    print("\nEnvironment closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
