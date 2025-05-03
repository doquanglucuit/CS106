import os
random_seeds = [23520902 + _ for _ in range(5)]
list_layouts = ['smallClassic.lay', 'mediumClassic.lay', 'trickyClassic.lay', 'originalClassic.lay', 'powerClassic.lay']
list_algorithms = ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']
list_evalFn = ['scoreEvaluationFunction', 'betterEvaluationFunction', 'newBetterEvaluationFunction']

depth = 3 # depth = 2 with tricky, original and power layouts

with open ('result.csv', 'a') as f:
    f.write('layout,algorithm,evalFn,seed,point,WIN/LOSS,time\n')
    # f.write('\n')
    f.flush()
    for layout in list_layouts:
        for algorithm in list_algorithms:
            for evalFn in list_evalFn:
                for seed in random_seeds:
                    line = f'python pacman.py -l {layout} -p {algorithm} -a depth={depth},evalFn={evalFn} --frameTime 0 -s {seed}'
                    print(line)
                    tmp = layout + ',' + algorithm + ',' + evalFn + ',' + str(seed)
                    f.write(tmp)
                    f.flush()
                    os.system(line)

