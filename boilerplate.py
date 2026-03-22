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
    cummulative_regret_per_time_stamp = [] #cumulative regret per time stamp

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
        cummulative_regret_per_time_stamp.append(cumulative_regret)

    return np.array(rewards), np.array(regrets), cummulative_regret_per_time_stamp


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
    cummulative_regret_per_time_stamp = [] #cumulative regret per time stamp

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
        cummulative_regret_per_time_stamp.append(cumulative_regret)


    return np.array(rewards), np.array(regrets),cummulative_regret_per_time_stamp

# UCB1
def ucb1(bandit, T):
    K = bandit.K
    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    empirical_means = [] #empirical means of arms
    no_of_pulls_per_arm = [] #no of pulls made by each arm
    total_arm_rewards=[] #total rewards of each arm
    cumulative_regret = 0
    cummulative_regret_per_time_stamp = [] #cumulative regret per time stamp
    calculated_em = [] #calculated emipircal means

    # Pull each arm once
    for arm in range(K):
        rewards.append(bandit.pull(arm))
        empirical_means.append(rewards[arm])
        no_of_pulls_per_arm.append(1)
    total_arm_rewards = rewards
    calculated_em = empirical_means

    for t in range(K, T):
        opt_arm = np.argmax(empirical_means)
        no_of_pulls_per_arm[opt_arm]+=1;
        rewards.append(bandit.pull(opt_arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm] + (np.sqrt((2*np.log(t+K))/no_of_pulls_per_arm[opt_arm]))
        calculated_em[opt_arm] = total_arm_rewards[opt_arm]/(no_of_pulls_per_arm[opt_arm])
        regrets.append(abs(bandit.best_mean-calculated_em[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
        cumulative_regret += regrets[-1]
        cummulative_regret_per_time_stamp.append(cumulative_regret)

    return np.array(rewards), np.array(regrets),cummulative_regret_per_time_stamp


# Thompson Sampling
def thompson_sampling(bandit, T):

    #Initialize algorithm dependent variable here
    K = bandit.K
    rewards, regrets = [], []
    beta_distributions = [] #beta distributions of each arms
    no_of_pulls_per_arm = [0] * K # total no of pulls made by each arm
    total_arm_rewards = [0] * K # total rewards obtained by each arm
    cumulative_regret = 0
    cummulative_regret_per_time_stamp = [] #cumulative regret per time stamp
    empirical_means=[0,0,0,0,0] #empirical means

    for arm in range(K):
        beta_distributions.append(np.random.beta(1,1,size=1)[0])

    for t in range(T):
        opt_arm = np.argmax(beta_distributions)
        no_of_pulls_per_arm[opt_arm]+=1
        rewards.append(bandit.pull(opt_arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        empirical_means[opt_arm] = total_arm_rewards[opt_arm]/(no_of_pulls_per_arm[opt_arm])
        regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/(t+1)))
        cumulative_regret += regrets[-1]
        for arm in range(K):
            a = total_arm_rewards[arm]
            b = no_of_pulls_per_arm[arm] - a
            if(a <= 0 or b<= 0):
                if(a<=0 and b<=0):
                    a=1
                    b=1
                if(b<=0):
                    b=1
                if(a<=0):
                    a=1
        beta_distributions[opt_arm] = np.random.beta(a,b,size=1)[0]
        cummulative_regret_per_time_stamp.append(cumulative_regret)


    return np.array(rewards), np.array(regrets),cummulative_regret_per_time_stamp

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
    sequence=list(np.linspace(0,1,10))
    f = 1 + t*np.log(t)**2
    upper_bound = np.log(f)/N_k
    updated_list = [i for i in range(0,10) if kl_divergence(empirical_mean,sequence[i])<=upper_bound]
    updated_sequence = [kl_divergence(empirical_mean,i) for i in sequence if kl_divergence(empirical_mean,i)<=upper_bound]
    if len(updated_sequence):
        return sequence[updated_list[np.argmin(updated_sequence)]]
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
    cumulative_regret = 0 #cumulative regret 
    cummulative_regret_per_time_stamp = [] #cumulative regret per time stamp


    # Pull each arm once
    for arm in range(K):
        rewards.append(bandit.pull(arm))
        empirical_means.append(rewards[arm])
        no_of_pulls_per_arm.append(1)
    total_arm_rewards = rewards

    for t in range(K,T):
        opt_arm = np.argmax([max_q(empirical_means[i],t,no_of_pulls_per_arm[i]) for i in range(0,len(empirical_means))])
        no_of_pulls_per_arm[opt_arm]+=1;
        rewards.append(bandit.pull(opt_arm))
        total_arm_rewards[opt_arm] += rewards[-1]
        empirical_means[opt_arm] = total_arm_rewards[opt_arm]/no_of_pulls_per_arm[opt_arm]
        regrets.append(abs(bandit.best_mean-empirical_means[opt_arm])*(no_of_pulls_per_arm[opt_arm]/t))
        cumulative_regret += regrets[-1]
        cummulative_regret_per_time_stamp.append(cumulative_regret)

        
    
    return np.array(rewards), np.array(regrets),cummulative_regret_per_time_stamp

# ──────────────────────────────────────────────
# Run & Plot
# ──────────────────────────────────────────────

def run_experiment(means, T, n_runs):
    algorithms = {
        "Greedy":            greedy,
        "Eps-Greedy(0.1)":   lambda b, T: epsilon_greedy(b, T, epsilon=0.1),
        "UCB1":              ucb1,
        "KL-UCB":            kl_ucb,
        "Thompson Sampling": thompson_sampling,
    }
    results = {name: [] for name in algorithms}
    cummulative_regrets = {name: [] for name in algorithms}

    
    for _ in range(n_runs):
        bandit = BernoulliBandit(means)
        X=np.arange(bandit.K,T)
        for name, algo in algorithms.items():
            _, regret,cr = algo(bandit, T)
            results[name].append(regret)
            cummulative_regrets[name].append(cr)

    # Average over runs
    avg_regrets = {name: np.mean(runs, axis=0) for name, runs in results.items()}
    avg_cummulative_regrets = {name: np.mean(runs, axis=0) for name, runs in cummulative_regrets.items()}
    
    # ── Plot 1: Cumulative Regret Line Chart ──
    
    X = np.linspace(0,T,T)
    # plt.plot(X,avg_cummulative_regrets['Greedy'],label="Greedy",marker='.')
    # plt.plot(X,avg_cummulative_regrets['Eps-Greedy(0.1)'],label="Eps-Greedy",marker=',')
    # plt.plot(X,avg_cummulative_regrets['UCB1'],label="UCB1",marker='--')
    # plt.plot(X,avg_cummulative_regrets['KL-UCB'],label="kl-ucb",marker='|')
    # plt.plot(X,avg_cummulative_regrets['Thompson Sampling'],label="TS")
    # plt.xlabel("Time step (t)")
    # plt.ylabel("Cumulative Regret")
    # plt.title("Cumulative regret vs time steps (t)")
    # plt.show() 
    # plt.savefig('./plot1.txt')
    # ── Plot 2: Final Average Regret Bar Chart ──
    plt.plot(X,avg_regrets['Greedy'],label='Greedy', marker='.')
    plt.plot(X,avg_regrets['Eps-Greedy(0.1)'],label='Eps-Greedy', marker=',')
    plt.plot(X,avg_regrets['UCB1'],label='UCB1',marker='--')
    plt.plot(X,avg_regrets['KL-UCB'],label='kl-ucb',marker='|')
    plt.plot(X,avg_regrets['Thompson Sampling'],label='tS')
    plt.xlabel("Time Step (t)")
    plt.ylabel("average regret")
    plt.title('Average regret vs time steps (t)')
    plt.show()
    plt.savefig('./plot2.txt')
  



if __name__ == "__main__":
    np.random.seed(42)
    MEANS = [0.1, 0.3, 0.5, 0.6, 0.9]   # 5-armed bandit, best arm = 0.9
    T     = 10000
    RUNS  = 50 #try to reduce the no of runs as kl-ucb is taking more running time because it needs to go over all the elements in set [0,1] for each K in each T of each run
    run_experiment(MEANS, T, RUNS)
