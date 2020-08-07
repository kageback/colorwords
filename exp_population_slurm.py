import argparse
import numpy as np
from population_game import PopulationGame
from graph import PopulationGraph
from agents import SoftmaxAgent
from com_game import DiscreteGame
import com_enviroments
import json
import torch
import os
import evaluate
import Correlation_Clustering
from collections import Counter
parser = argparse.ArgumentParser(description='Run population game on Chalmers slurm clusters.')
parser.add_argument('--exp', help='path to .json file describing the exp')


def run(args):
    exp_file = args.exp
    print(exp_file)
    with open(exp_file, 'r') as f:
        exp = json.load(f)

    result_path = exp['exp_path'] + 'results/'

    # Set seed
    torch.manual_seed(exp['seed'])
    np.random.seed(exp['seed'])

    # Ini exp
    env = com_enviroments.make(exp['env'])
    agents = []
    for i in range(exp['n_agents']):
        agents.append(
            SoftmaxAgent(msg_dim=exp['msg_dim'], perception_dim=exp['perception_dim'], color_dim=exp['out_dim'], hidden_dim=exp['hidden_dim']))
    # Build graph
    if exp['csv'] == 0:
        csv = None
    else:
        csv = exp['csv']
    if exp['n_agents'] > 2:
        graph = PopulationGraph(agents, csv=csv)
        game = PopulationGame(max_epochs=exp['max_epochs'], batch_size=exp['batch_size'], print_interval=exp['print_interval'],
                              perception_noise=exp['perception_noise'])
        trained_population = game.play(graph, env)
        results = analyze(trained_population, env)
    else:
        game = DiscreteGame(com_noise=0,
                            msg_dim=exp['msg_dim'],
                            max_epochs=exp['max_epochs'],
                            perception_noise=exp['perception_noise'],
                            batch_size=exp['batch_size'],
                            print_interval=exp['print_interval'],
                            loss_type='REINFORCE')
        agent = game.play(env, agents[0], agents[1])
        results = analyze([agent[0]], env)

    with open(exp['exp_path'] + '/results/result_{}.json'.format(exp['run']), 'w') as f:
        json.dump(results, f, indent=4)

    # env.plot_with_colors(results['mode_map'])
def analyze(population, env):
    results = {}
    agent_maps = []
    agent_term_usage = []
    agent_wellformedness = []
    agent_kl_loss = []
    agent_suprisal = []

    for agent in population:
        V = evaluate.agent_language_map(env, a=agent)
        term_usage = evaluate.compute_term_usage(V)[0]
        wellformedness = evaluate.wellformedness(env, V)
        kl_loss = evaluate.regier2(env, V)
        suprisal = evaluate.compute_gibson_cost2(env, agent)

        agent_term_usage.append(term_usage)
        agent_maps.append([V])
        agent_wellformedness.append(wellformedness)
        agent_kl_loss.append(kl_loss)
        agent_suprisal.append(suprisal)

    occurence_count = Counter(agent_term_usage)
    most_common_terms = occurence_count.most_common(1)[0][0]

    results['term_usage_std'] = np.std(np.array(agent_term_usage)).item()
    results['term_usage_consensus'] = most_common_terms
    results['wellformedness_std'] = np.std(np.array(agent_wellformedness)).item()
    results['kl_loss_std'] = np.std(np.array(agent_kl_loss)).item()
    results['suprisal_std'] = np.std(np.array(agent_suprisal)).item()
    results['suprisal_mean'] = np.std(np.array(agent_suprisal)).item()

    if len(population) > 1:
        agent_maps = np.array(agent_maps).reshape((len(population), 330))
        agent_term_usage = np.array(agent_term_usage)
        maps_to_analyze = agent_maps[agent_term_usage==most_common_terms, :]
        agent_consensus_map = Correlation_Clustering.compute_consensus_map(agent_maps, k=most_common_terms,
                                                                           iter=10)
        agent_consensus_map = agent_consensus_map.tolist()
    else:
        agent_consensus_map = V

    population_wellformedness = evaluate.wellformedness(env, agent_consensus_map).detach().cpu().numpy()
    population_kl_loss = evaluate.regier2(env, agent_consensus_map)
    results['population_wellformedness'] = population_wellformedness[0].item()
    results['population_kl_loss'] = population_kl_loss
    results['mode_map'] = agent_consensus_map

    return results













if __name__ == '__main__':
    args = parser.parse_args()
    run(args)