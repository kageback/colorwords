import gridengine as sge
import numpy as np
import com_enviroments
import evaluate
from gridengine.pipeline import Experiment
from sklearn.metrics.cluster import adjusted_rand_score

def main():
    consensus_iters = 10
    e = com_enviroments.make('wcs')

    k = 3



    sims= []


    # human maps
    human_maps = list(e.human_mode_maps.values())

    # robo maps
    exp = Experiment.load('color_fix.1')
    robo_maps = exp.reshape('agent_language_map')



    human_rand = evaluate.mean_rand_index(human_maps)
    exp.log.info('mean rand for all human maps = {:.3f}'.format(human_rand))

    robo_rand = evaluate.mean_rand_index(robo_maps)
    exp.log.info('mean rand for all agent maps = {:.3f}'.format(robo_rand))

    cross_rand = evaluate.mean_rand_index(human_maps, robo_maps)
    exp.log.info('mean rand cross human and robot maps = {:.3f}'.format(cross_rand))



    for k in range(3, 12):
        cielab_map = evaluate.compute_cielab_map(e, k, iter=consensus_iters, bw_boost=1)
        e.plot_with_colors(cielab_map, save_to_path=exp.pipeline_path + 'cielab_map_' + str(k) + '.png')

        human_consensus_map = evaluate.compute_consensus_map(human_maps, k=k, iter=consensus_iters)
        e.plot_with_colors(human_consensus_map, save_to_path=exp.pipeline_path + 'human_consensus_language_map_' + str(k) + '.png')

        robo_consensus_map = evaluate.compute_consensus_map(robo_maps, k=k, iter=consensus_iters)

        e.plot_with_colors(robo_consensus_map, save_to_path=exp.pipeline_path + 'consensus_language_map_' + str(k) + '.png')

        # compare human and robo maps
        rand_i = adjusted_rand_score(human_consensus_map, robo_consensus_map)
        print('rand i between human consensus and agent consensus = {:.3f}'.format(rand_i))
        sims += [rand_i]

    sims = np.array(sims)
    print(sims.mean())


if __name__ == "__main__":
    main()