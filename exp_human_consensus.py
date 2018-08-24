import gridengine as sge
import numpy as np
import com_enviroments
import evaluate
from gridengine.pipeline import Experiment
from sklearn.metrics.cluster import adjusted_rand_score

def main():
    consensus_iters = 10
    e = com_enviroments.make('wcs')
    sims= []


    # human maps
    human_maps = list(e.human_mode_maps.values())
    # human_lang_nums = range(1, 31) #31
    # human_maps = []
    # for lang_num in human_lang_nums:
    #     human_maps += [e.human_language_mode_map(lang_num)]
    #     print('mode map ' + str(lang_num) + ' of' + str(len(human_lang_nums)))

    # robo maps
    exp = Experiment.load('color_avg20.0')
    robo_maps = exp.get_flattened_results('agent_language_map')


    for k in range(3, 12):

        human_consensus_map = evaluate.compute_consensus_map(human_maps, k=k, iter=consensus_iters)
        e.plot_with_colors(human_consensus_map, save_to_path=exp.pipeline_path + 'human_consensus_language_map_' + str(k) + '.png')

        robo_consensus_map = evaluate.compute_consensus_map(robo_maps, k=k, iter=consensus_iters)

        e.plot_with_colors(robo_consensus_map, save_to_path=exp.pipeline_path + 'consensus_language_map_' + str(k) + '.png')

        # compare human and robo maps
        rand_i = adjusted_rand_score(list(human_consensus_map.values()), list(robo_consensus_map.values()))
        print(rand_i)
        sims += [rand_i]

    sims = np.array(sims)
    print(sims.mean())


if __name__ == "__main__":
    main()