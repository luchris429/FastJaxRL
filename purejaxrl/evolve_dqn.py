"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import os
from importlib import import_module
import jax
import jax.numpy as jnp
import google.generativeai as genai
import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax
import flashbax as fbx
import time
import os
import inspect
import random
import re
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"



class QNetwork(nn.Module):
    action_dim: int
    activation: nn.Module  # a callable Flax module or any function you like
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = self.activation(x)  # use the injected activation instead of nn.relu
        x = nn.Dense(84)(x)
        x = self.activation(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


def make_train(config, activation_fn):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            activation=activation_fn,  # pass the activation you want to test
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(
                rng_a, q_vals, train_state.timesteps
            )  # explore with epsilon greedy_exploration
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, learn_batch.second.obs
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(params):
                    q_vals = network.apply(
                        params, learn_batch.first.obs
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0)),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def get_function(function_name, module_name):
    # Remove module from sys.modules if it exists
    import sys
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Remove pycache files
    import shutil
    pycache_dir = "__pycache__"
    if os.path.exists(pycache_dir):
        shutil.rmtree(pycache_dir)
    
    # Remove .pyc file if it exists
    pyc_file = f"{module_name}.pyc"
    if os.path.exists(pyc_file):
        os.remove(pyc_file)
        
    module = import_module(module_name)  # Now reimport the fresh module
    return getattr(module, function_name)

def run_experiment(config, activation_fn):
    # Add imports before writing the activation function
    imports = open("base_activations/base_imports.txt", "r").read()
    fn_name = "custom_activation"
    temp_activation = rename_function(activation_fn, fn_name)
    with open("temp_activation.py", "w") as f:
        f.write(imports + '\n\n' + temp_activation)
    try:
        custom_activation = get_function(fn_name, "temp_activation")
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, custom_activation)))
        outs = train_vjit(rngs)
        avg_score_over_seeds = outs["metrics"]['returns'].mean(axis=0)
        total_score = avg_score_over_seeds.sum()
    except Exception as e:
        with open("failed_activations.log", "a") as f:
            f.write(f"\nFailed Activation Function:\n{activation_fn}\n\nException:\n{str(e)}\n\n")
        total_score = - 100000000
        avg_score_over_seeds = -1

    return total_score, avg_score_over_seeds

def gen_crossover(act1, act2, score1, score2):
    act1, act2 = rename_function(act1, "activation1"), rename_function(act2, "activation2")
    model = genai.GenerativeModel("gemini-1.5-flash")
    with open("prompt_activation.txt", "r") as file:
        prompt = file.read()
    prompt = prompt.replace("[FUNCTION_CODE_1]", act1)
    prompt = prompt.replace("[FUNCTION_CODE_2]", act2)
    prompt = prompt.replace("[SCORE_1]", str(score1))
    prompt = prompt.replace("[SCORE_2]", str(score2))
    response = model.generate_content(prompt)
    parsed_response = response.text.split("```python")[1].split("```")[0]
    return parsed_response

def rename_function(function_str, function_name):
    # Extract original function name using regex
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', function_str)
    if not match:
        return function_str
    original_name = match.group(1)
    
    # Replace all occurrences of original name with new name, ensuring exact match with word boundaries
    pattern = r'\b' + re.escape(original_name) + r'\b'
    return re.sub(pattern, function_name, function_str)

def get_pair(population):
    activation1, activation2 = random.sample(list(population.keys()), 2)
    score1, score2 = population[activation1], population[activation2]
    return activation1, activation2, score1, score2

def plot_scores(best_list, filename):
    plt.figure(figsize=(12, 6))
    for idx, (_, score, scores) in enumerate(best_list):
        plt.plot(scores, label=f'Best {idx+1}')
    
    plt.xlabel('Steps')
    plt.ylabel('Average Return')
    plt.title('Performance of Best Activation Functions')
    plt.legend()
    os.makedirs("BestPlots", exist_ok=True)
    plt.savefig(f"BestPlots/{filename}.png")
    
    # Log plot to wandb
    wandb.log({
        "activation_performance": wandb.Image(plt)
    })
    
    plt.close()

# For processing
def main():


    best_score_so_far = -100000000
    best_activation_so_far = ""
    succesful_count = 0
    failed_count = 0

    genai.configure(api_key="AIzaSyAuTt22urZ-jow0KqRuMZxUpVI8SFsb9LU")
    config = {
        "NUM_ENVS": 10,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "Breakout-MinAtar",
        "SEED": 0,
        "NUM_SEEDS": 32,
        "WANDB_MODE": "disabled",  # set to online to activate wandb
        "ENTITY": "",
        "PROJECT": "",
    }
    wandb.init(
            project='Evolving_RL',
            entity='playing_around',
            name=config['ENV_NAME'] + '_actv_evo',
            mode='disabled'
        )
    best = []  # Changed to store (activation_fn, best_score, scores)
    #Set up base population
    population = {}
    base_dir = "base_activations"
    for filename in os.listdir(base_dir):
        if filename.startswith("activation_"):
            print(filename)
            activation_path = os.path.join(base_dir, filename)
            with open(activation_path, "r") as f:
                activation_fn = f.read()
            population[activation_fn], scores = run_experiment(config, activation_fn)
            print(population[activation_fn])
            if population[activation_fn] == -100000000:
                del population[activation_fn]
                failed_count += 1
            else:
                succesful_count += 1
            if activation_fn in population and population[activation_fn] > best_score_so_far:
                best_score_so_far = population[activation_fn]
                best_activation_so_far = activation_fn
                best.append((activation_fn, best_score_so_far, scores))
                plot_scores(best, "evolution_progress")

                print(f"New best activation: {best_activation_so_far} with score {best_score_so_far}")
                wandb.log({
                    "best_score": best_score_so_far,
                    "step": succesful_count
                })
    print(f"Base population size: {len(population)}")
    exit()
    #Execute evolution
    NUM_PHASES = 10
    NUM_PROMPTS = 20
    NUM_TO_KEEP = 10
    NUM_SAMPLES = 5
    for phase in range(NUM_PHASES):
        wandb.log({
            "phase": phase,
            "succesful_count": succesful_count,
            "failed_count": failed_count
        })
        # Sample 10 pairs with replacement
        pairs = []
        for _ in range(NUM_PROMPTS):
            activation1, activation2, score1, score2 = get_pair(population)
            pairs.append(gen_crossover(activation1, activation2, score1, score2))
        for pair in pairs:
            i = 0
            while i < NUM_SAMPLES:
                activation = gen_crossover(pair[0], pair[1], pair[2], pair[3])
                population[activation], scores = run_experiment(config, activation)
                if population[activation] == -100000000:
                    del population[activation]
                    failed_count += 1
                else:
                    if population[activation] > best_score_so_far:
                        best_score_so_far = population[activation]
                        best_activation_so_far = activation
                        best.append((activation, best_score_so_far, scores))
                        plot_scores(best, "evolution_progress")
                        print(f"New best activation: {best_activation_so_far} with score {best_score_so_far}")
                        wandb.log({
                            "best_score": best_score_so_far,
                            "step": succesful_count
                        })
                i += 1
                succesful_count += 1
        
        #Sort population
        population = dict(sorted(population.items(), key=lambda item: item[1], reverse=True))
        #Keep top 10
        population = dict(list(population.items())[:NUM_TO_KEEP])
        print(f"Phase {phase} complete")
        #Store population in text file
        os.makedirs("phase_results", exist_ok=True)
        with open(f"phase_results/phase_{phase}.txt", "w") as f:
            for activation, score in population.items():
                f.write("Activation:\n")
                f.write(activation)
                f.write(f"Score: {score}\n")
                f.write("\n\n")

    with open("best.txt", "w") as f:
        for activation, score in best:
            f.write(f"Activation: {activation}\n")
            f.write(f"Score: {score}\n")
            f.write("\n\n")

if __name__ == "__main__":
    main()