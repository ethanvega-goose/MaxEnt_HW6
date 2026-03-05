import numpy as np

def value_iteration(discount, env, rewards, theta=1e-4):
    """
    Computes the optimal STOCHASTIC policy using Soft Value Iteration.
    Returns a policy matrix of shape (n_states, n_actions) containing 
    action probabilities instead of strict 1.0/0.0 rules.
    """
    V = np.zeros(env.n_states)
    
    # Calculate State Values (Soft Bellman Backup)
    while True:
        delta = 0
        
        # We use a temporary array to ensure all states update synchronously
        V_new = np.zeros(env.n_states) 
        
        for s in range(env.n_states):
            v = V[s]
            
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for s_p in range(env.n_states):
                    prob = env.dynamics[s, a, s_p] 
                    q_values[a] += prob * V[s_p]
                    
            # --- THE MAXENT DIFFERENCE ---
            # Instead of taking the hard np.max(), we use the Log-Sum-Exp 
            # trick to calculate a "Soft Maximum". 
            
            # (We subtract max_q before exponentiating for numerical stability 
            # so Python doesn't overflow to infinity)
            max_q = np.max(q_values)
            soft_max = max_q + np.log(np.sum(np.exp(q_values - max_q)))
            
            # Update the state value
            V_new[s] = rewards[s] + discount * soft_max
            delta = max(delta, abs(v - V_new[s]))
            
        V = V_new
        if delta < theta:
            break
            
    # --- Extract Stochastic Policy ---
    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        q_values = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            for s_p in range(env.n_states):
                prob = env.dynamics[s, a, s_p]
                q_values[a] += prob * V[s_p]
        
        # Convert the Q-values into a probability distribution using the Softmax function!
        # High Q-values get probabilities close to 1.0, low Q-values get probabilities close to 0.0
        max_q = np.max(q_values)
        exp_q = np.exp(q_values - max_q)
        policy[s] = exp_q / np.sum(exp_q) 
        
    return policy