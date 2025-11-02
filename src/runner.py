from src.env.carRacingEnv import CarRacingEnv
from src.agents.manualAgent import ManualAgent
from src.agents.DQNAgent import DQNAgent
from src.agents.randomAgent import RandomAgent
import time


class Runner:
    def __init__(self, map_path, tileset_path, agent_type, render_mode):

        self.map_path = map_path
        self.tileset_path = tileset_path
        self.agent_type = agent_type
        self.render_mode = render_mode
        
        self.env = None
        self.agent = None
        self.runs = 0

        self.env = CarRacingEnv(
            map_path=self.map_path,
            tileset_path=self.tileset_path,
            render_mode=self.render_mode
        )

        if self.agent_type in {"DQN", "random"}:
            state_shape = getattr(self.env.observation_space, "shape", None)
            state_size = int(state_shape[0]) if state_shape else 0
            action_size = int(getattr(self.env.action_space, "n", 1))

            if self.agent_type == "DQN":
                self.agent = DQNAgent(
                    state_size=state_size,
                    action_size=action_size,
                    seed=0,
                )
            else:
                self.agent = RandomAgent(
                    state_size=state_size,
                    action_size=action_size,
                    seed=0,
                )
        elif self.agent_type == "manual":
            self.agent = ManualAgent()
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
        if self.agent_type == "manual":
            print("\n=== MANUAL CONTROL ===")
            print("Discrete actions triggered by arrow keys:")
            print("  ↑ Up Arrow:    Accelerate (action 0)")
            print("  ↓ Down Arrow:  Brake      (action 1)")
            print("  ← Left Arrow:  Steer left (action 2)")
            print("  → Right Arrow: Steer right(action 3)")
            print("Release all keys to coast (no-op).")
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
                self.runs += 1
                print(f"\n=== RUN {self.runs} ENDED at step {step} ===")
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
