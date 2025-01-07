import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from wrappers import LogWrapper, FlattenObservationWrapper
import os
import wandb
import random
import matplotlib.pyplot as plt
import re
from importlib import import_module
import google.generativeai as genai

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: nn.Module  # a callable Flax module or any function you like

    @nn.compact
    def __call__(self, x):

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = self.activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = self.activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = self.activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = self.activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, activation_fn):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n, activation=activation_fn
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            # Debugging mode
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

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

best_scores_all_envs = {}

def evaluate_discrete(config, activation_fn):
    env_names = ["Asterix-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar", "SpaceInvaders-MinAtar"]
    for env_name in env_names:
        config_copy = config.copy()
        config_copy["ENV_NAME"] = env_name
        total, scores = run_experiment_discrete(config_copy, activation_fn)
        if env_name not in best_scores_all_envs:
            best_scores_all_envs[env_name] = []
        best_scores_all_envs[env_name].append(scores)
        plot_scores(best_scores_all_envs[env_name], f"test_{env_name}_train_{config['ENV_NAME']}")

def run_experiment_discrete(config, activation_fn):
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
        avg_score_over_seeds = outs["metrics"]['returned_episode_returns'].mean(axis=0)
        total_score = avg_score_over_seeds.sum()
        avg_score_over_seeds = avg_score_over_seeds.mean(-1).reshape(-1)
    except Exception as e:
        print(f"Failed Activation Function:\n{activation_fn}\n\nException:\n{str(e)}\n\n")
        total_score = -100000000
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


if __name__ == "__main__":
    genai.configure(api_key="AIzaSyAuTt22urZ-jow0KqRuMZxUpVI8SFsb9LU")
    config = {
        "LR": 5e-3,
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "Asterix-MinAtar",
        "ANNEAL_LR": True,
        "DEBUG": False,
        "WANDB_MODE": "online",
        "ENTITY": "playing_around",
        "PROJECT": "Evolving_RL",
        "NUM_SEEDS": 16,
        "SEED": 0,
    }

    wandb.init(
        project=config['PROJECT'],
        entity=config['ENTITY'],
        name='short' + config['ENV_NAME'] + '_actv_evo',
        mode=config['WANDB_MODE']
    )

    best_score_so_far = -100000000
    best_activation_so_far = ""
    succesful_count = 0
    failed_count = 0
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
            population[activation_fn], scores = run_experiment_discrete(config, activation_fn)
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
                evaluate_discrete(config, activation_fn)

                print(f"New best activation: {best_activation_so_far} with score {best_score_so_far}")
                wandb.log({
                    "best_score": best_score_so_far,
                    "step": succesful_count
                })
            if len(population) > 2:
                break
    print(f"Base population size: {len(population)}")


#Execute evolution
    NUM_PHASES = 10
    NUM_PROMPTS = 10
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
                population[activation], scores = run_experiment_discrete(config, activation)
                if population[activation] == -100000000:
                    del population[activation]
                    failed_count += 1
                else:
                    if population[activation] > best_score_so_far:
                        best_score_so_far = population[activation]
                        best_activation_so_far = activation
                        best.append((activation, best_score_so_far, scores))
                        plot_scores(best, "evolution_progress")
                        evaluate_discrete(config, activation)
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