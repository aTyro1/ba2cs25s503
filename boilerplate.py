import numpy as np
import plotext as plt

# Bandit Environment

class BernoulliBandit:
    def __init__(self, means):
        self.means = np.array(means)
        self.K = len(means)
        self.best_mean = np.max(means)

    def pull(self, arm):
        return int(np.random.rand() < self.means[arm])

# Greedy bandit

def greedy(bandit, T):
    """Pure greedy: always exploit the current best estimated arm.
    Pulls each arm once to initialise, then always picks argmax(mean)."""
    K = bandit.K
    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    empirical_means = [] #empirical means of arms
    no_of_pulls_per_arm = [] #no of pulls made by each arm
    total_arm_rewards=[] #total rewards of each arm
    cumulative_regret = 0

    # Initialise: pull each arm once
    for arm in range(K):
        rewards.append(bandit.pull(arm))
        empirical_means.append(rewards[arm])
        no_of_pulls_per_arm.append(1)
    total_arm_rewards =  rewards

    for t in range(K, T):
        opt_arm = np.argmax(empirical_means)
        no_of_pulls_per_arm[opt_arm]+=1
        rewards.append(bandit.pull(arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm]
        regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
        cumulative_regret += regrets[-1]

    return np.array(rewards), np.array(regrets)


#Epsilon-Greedy

def epsilon_greedy(bandit, T, epsilon=0.1):
    """Epsilon-greedy: explore uniformly at random with probability epsilon,
    exploit the current best arm otherwise."""
    K = bandit.K 
    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    empirical_means = [] #empirical means of arms
    no_of_pulls_per_arm = [] #no of pulls made by each arm
    total_arm_rewards=[] #total rewards of each arm
    cumulative_regret = 0

    # Initialise: pull each arm once
    for arm in range(K):
        rewards.append(bandit.pull(arm))
        empirical_means.append(rewards[arm])
        no_of_pulls_per_arm.append(1)
    total_arm_rewards=rewards
    
    for t in range(K, T):
        if(int(np.random.binomial(n=1, p=epsilon))):
            opt_arm = np.random.randint(0,K)
            no_of_pulls_per_arm[opt_arm]+=1;
            rewards.append(bandit.pull(opt_arm))
            total_arm_rewards[opt_arm] += rewards[-1]
            empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm]
            regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
            cumulative_regret += regrets[-1]
        else:
            opt_arm = np.argmax(empirical_means)
            no_of_pulls_per_arm[opt_arm]+=1;
            rewards.append(bandit.pull(opt_arm))
            total_arm_rewards[opt_arm] += rewards[-1]
            empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm]
            regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
            cumulative_regret += regrets[-1]


    return np.array(rewards), np.array(regrets)

# UCB1
def ucb1(bandit, T):
    K = bandit.K
    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    empirical_means = [] #empirical means of arms
    no_of_pulls_per_arm = [] #no of pulls made by each arm
    total_arm_rewards=[] #total rewards of each arm
    cumulative_regret = 0

    # Pull each arm once
    for arm in range(K):
        rewards.append(bandit.pull(arm))
        empirical_means.append(rewards[arm])
        no_of_pulls_per_arm.append(1)
    total_arm_rewards = rewards
    for t in range(K, T):
        opt_arm = np.argmax(empirical_means)
        no_of_pulls_per_arm[opt_arm]+=1;
        rewards.append(bandit.pull(opt_arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm] + (np.sqrt((2*np.log(t+K))/no_of_pulls_per_arm[opt_arm]))
        regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
        cumulative_regret += regrets[-1]

    return np.array(rewards), np.array(regrets)


# Thompson Sampling
def thompson_sampling(bandit, T):

    #Initialize algorithm dependent variable here
    K = bandit.K
    rewards, regrets = [], []
    beta_distributions = [] #beta distributions of each arms
    no_of_pulls_per_arm = [0] * K # total no of pulls made by each arm
    total_arm_rewards = [0] * K # total rewards obtained by each arm
    cumulative_regret = 0

    for arm in range(K):
        beta_distributions.append(np.random.beta(1,1,size=1))

    for t in range(T):
        opt_arm = np.argmax(beta_distributions)
        no_of_pulls_per_arm[opt_arm]+=1
        rewards.append(bandit.pull(opt_arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        regrets.append(abs(bandit.best_mean-beta_distributions[opt_arm])*(no_of_pulls_per_arm[opt_arm]/(t+1)))
        cumulative_regret += regrets[-1]
        a = total_arm_rewards[opt_arm]
        b = no_of_pulls_per_arm[opt_arm] - a
        if(a <= 0 or b<= 0):
            if(a<=0):
                a=1
            else:
                b=1
        beta_distributions[opt_arm] = np.random.beta(a,b,size=1)


    return np.array(rewards), np.array(regrets)

#function to calculate KL divergence
def kl_divergence(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    epsilon = 1e-10
    P = P + epsilon
    Q = Q + epsilon
    
    return abs(np.sum(P * np.log(P / Q)))

#function to calculate the value of q which gives maximum kl divergence of a given empirical mean
def max_q(empirical_mean,t,N_k):
    sequence=list(np.linspace(0,1,20))
    f = 1 + t*np.log(t)**2
    upper_bound = np.log(f)/N_k
    updated_sequence = [kl_divergence(empirical_mean,i) for i in sequence if kl_divergence(empirical_mean,i)<=upper_bound]
    if len(updated_sequence):
        return sequence[np.argmax(updated_sequence)]
    else:
        return empirical_mean

#KL-UCB
def kl_ucb(bandit,T):
    K = bandit.K
    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    empirical_means = [] #empirical means of arms
    no_of_pulls_per_arm = [] #no of pulls made by each arm
    total_arm_rewards=[] #total rewards of each arm
    cumulative_regret = 0

    # Pull each arm once
    for arm in range(K):
        rewards.append(bandit.pull(arm))
        empirical_means.append(rewards[arm])
        no_of_pulls_per_arm.append(1)
    total_arm_rewards = rewards

    for t in range(K,T):
        empirical_means = [max_q(empirical_means[i],t,no_of_pulls_per_arm[i]) for i in range(0,len(empirical_means))]
        opt_arm = np.argmax(empirical_means)
        no_of_pulls_per_arm[opt_arm]+=1;
        rewards.append(bandit.pull(opt_arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm]
        regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
        cumulative_regret += regrets[-1]
    
    return np.array(rewards), np.array(regrets)

# ──────────────────────────────────────────────
# Run & Plot
# ──────────────────────────────────────────────

def run_experiment(means, T, n_runs):
    algorithms = {
        # "Greedy":            greedy,
        # "Eps-Greedy(0.1)":   lambda b, T: epsilon_greedy(b, T, epsilon=0.1),
        # "UCB1":              ucb1,
        "KL-UCB":            kl_ucb,
        # "Thompson Sampling": thompson_sampling,
    }
    results = {name: [] for name in algorithms}

    for _ in range(n_runs):
        bandit = BernoulliBandit(means)
        X=np.arange(bandit.K,T)
        for name, algo in algorithms.items():
            _, regret = algo(bandit, T)
            results[name].append(regret)

    print(results['KL-UCB'])
    # Average over runs
    
    # ── Plot 1: Cumulative Regret Line Chart ──
    

    
    # ── Plot 2: Final Average Regret Bar Chart ──


if __name__ == "__main__":
    np.random.seed(42)
    MEANS = [0.1, 0.3, 0.5, 0.6, 0.9]   # 5-armed bandit, best arm = 0.9
    T     = 10000
    RUNS  = 50
    run_experiment(MEANS, T, RUNS)
