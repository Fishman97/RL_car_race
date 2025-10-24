import numpy as np
from src.carRacingEnv import CarRacingEnv
from src.agents import SimpleAgent

def main():
    map_path = "assets/maps/track1.tmx"
    tileset_path = "assets/tilesets/flat_race.tsx"

    env = CarRacingEnv(
        map_path=map_path,
        tileset_path=tileset_path,
        render_mode="human"
    )
    
    # Create the agent
    agent = SimpleAgent(steering=0.0, acceleration=1.0, brake=0.0)
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Agent initialized with fixed action: steering={agent.steering}, acceleration={agent.acceleration}, brake={agent.brake}")
    
    # Reset environment
    observation, info = env.reset()
    print(f"\nInitial observation: {observation}")
    print(f"Initial info: {info}")
    
    running = True
    step = 0
    
    while running:
        # Agent decides action (ignores observation and returns same action)
        action = agent.act(observation)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render and check if window should close
        running = env.render()
        
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: {info['position']}")
            print(f"  Velocity: {info['velocity']:.2f}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Observation: {observation}")
        
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
