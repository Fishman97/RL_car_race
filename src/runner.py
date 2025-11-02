from src.carRacingEnv import CarRacingEnv
from src.agents import SimpleAgent, ManualAgent
import time


class Runner:
    def __init__(self, map_path, tileset_path, agent_type, render_mode):

        self.map_path = map_path
        self.tileset_path = tileset_path
        self.agent_type = agent_type
        self.render_mode = render_mode
        
        self.env = None
        self.agent = None

        self.env = CarRacingEnv(
            map_path=self.map_path,
            tileset_path=self.tileset_path,
            render_mode=self.render_mode
        )

        if self.agent_type == "simple":
            self.agent = SimpleAgent(0,0,0)
        elif self.agent_type == "manual":
            self.agent = ManualAgent()
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
        if self.agent_type == "manual":
            print("\n=== MANUAL CONTROL ===")
            print("Use arrow keys to control the car:")
            print("  ← Left Arrow:  Steer left")
            print("  → Right Arrow: Steer right")
            print("  ↑ Up Arrow:    Accelerate")
            print("  ↓ Down Arrow:  Brake")
            print("Close the window to exit.")
            print("======================\n")
    
    def run(self, print_interval=60):
        if self.env is None or self.agent is None:
            raise RuntimeError("Environment or agent not initialized.")
        
        observation, info = self.env.reset()
        
        if self.agent_type == "simple":
            print(f"\nInitial observation: {observation}")
            print(f"Initial info: {info}")
        
        running = True
        step = 0
        start_time = time.time()
        
        while running:
            action = self.agent.act(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            running = self.env.render()
            
            if step % print_interval == 0:
                elapsed_time = time.time() - start_time
                sps = step / elapsed_time if elapsed_time > 0 else 0
                print(f"\nStep {step}:")
                print(f"  Position: {info['position']}")
                print(f"  Velocity: {info['velocity']:.2f}")
                print(f"  Reward: {reward:.2f}")
                print(f"  Steps/sec: {sps:.2f}")
                print(f"  Observation: {observation}")
            
            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                print(f"  Collision: {info['collision']}")
                print(f"  Finished: {info['finished']}")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
                print(f"  Restarting...")
                observation, info = self.env.reset()
                self.agent.reset()
                step = 0
                start_time = time.time()
            else:
                step += 1
        
        self.env.close()
        print("\nEnvironment closed.")
