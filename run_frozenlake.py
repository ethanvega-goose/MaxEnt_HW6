import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Import your algorithm classes
from MaximumEntropy import MaximumEntropy 
from functions.value_iteration import value_iteration

# 1. The Dynamic Expert Generator
def generate_sample_trajectories(env, expert_policy, num_trajectories, len_trajectory):
    """
    Generates trajectories using a mathematically calculated expert policy.
    The expert dynamically evaluates its state at every step.
    """
    trajs = []
    for _ in range(num_trajectories):
        traj = []
        s, _ = env.reset()
        for step in range(len_trajectory):
            
            # The expert dynamically chooses the best action for its CURRENT state
            a = np.argmax(expert_policy[s])
                
            s_p, r, terminated, truncated, _ = env.step(a)
            
            traj.append((s, a, s_p)) 
            s = s_p
            
            if terminated or truncated:
                while len(traj) < len_trajectory:
                    traj.append((s, 0, s)) 
                break
                
        trajs.append(traj)
        
    return np.array(trajs, dtype=int) 

if __name__ == "__main__" :
    # --- Parameters ---
    num_trajectories = 50 
    len_trajectory = 40   
    n_epochs = 100
    discount = 0.9
    lr = 0.1

    print("Initializing 8x8 Slippery Environment...")
    gym_env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode="human")
    
    gym_env.n_states = gym_env.observation_space.n
    gym_env.n_actions = gym_env.action_space.n
    gym_env.grid_size = 8 

    # --- Extract Dynamics ---
    dynamics = np.zeros((gym_env.n_states, gym_env.n_actions, gym_env.n_states))
    for s in gym_env.unwrapped.P:
        for a in gym_env.unwrapped.P[s]:
            for prob, next_state, reward, done in gym_env.unwrapped.P[s][a]:
                dynamics[s, a, next_state] += prob
    gym_env.dynamics = dynamics

    # --- Extract True Rewards to Train the Expert ---
    true_rewards = np.zeros(gym_env.n_states)
    for s in gym_env.unwrapped.P:
        for a in gym_env.unwrapped.P[s]:
            for prob, next_state, r, done in gym_env.unwrapped.P[s][a]:
                if r > 0:
                    true_rewards[next_state] = r
                    
    # The expert calculates the perfect way to play before generating data!
    print("Calculating True Expert Policy...")
    expert_policy = value_iteration(discount, gym_env, true_rewards)

    # --- Generate Trajectories and Features ---
    print("Generating Expert Data...")
    trajs = generate_sample_trajectories(gym_env, expert_policy, num_trajectories, len_trajectory)
    features = np.eye(gym_env.n_states) 

    # --- Initialize and Train MaxEnt IRL ---
    print("\nStarting Maximum Entropy IRL Training...")
    me = MaximumEntropy(gym_env, trajs, features, lr, discount)
    # Train and get the gradient history
    learned_rewards_flat, grad_history = me.train(n_epochs, plot=False)
    learned_rewards = learned_rewards_flat.reshape(gym_env.grid_size, gym_env.grid_size)

    # --- New Learning Curve Plot ---
    plt.figure(figsize=(6, 4))
    plt.plot(grad_history, color='red', linewidth=2)
    plt.title("Learning Curve: Feature Matching Error")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm (Expert vs. Agent Difference)")
    plt.grid(True)
    plt.show()

    # --- Plotting the Grids ---
    true_rewards = true_rewards.reshape(gym_env.grid_size, gym_env.grid_size)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pcolor(true_rewards, cmap='viridis')
    plt.colorbar()
    plt.title("True Reward (8x8 FrozenLake)")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.pcolor(learned_rewards, cmap='viridis')
    plt.colorbar()
    plt.title("Learned Reward (MaxEnt IRL)")
    plt.gca().invert_yaxis()
    
    print("Close the heatmap plot window to watch the AI test its learned reward map!")
    plt.show()

    # --- THE VISUAL PROOF ---
    print("\n=== VISUAL PROOF: Testing the Learned Reward ===")
    
    # Calculate the final policy based ONLY on our learned breadcrumb trail
    final_policy = value_iteration(discount, gym_env, learned_rewards.flatten())
    
    obs, _ = gym_env.reset()
    done = False
    
    print("Agent is now playing based on the learned reward function...")
    while not done:
        best_action = np.argmax(final_policy[obs]) 
        
        obs, r, terminated, truncated, _ = gym_env.step(best_action)
        gym_env.render() 
        
        done = terminated or truncated
        
    if r > 0:
        print("SUCCESS! The agent reached the goal using the inferred reward function!")
    else:
        print("FAILED. The agent did not reach the goal.")