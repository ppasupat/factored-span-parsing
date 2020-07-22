#!/bin/bash
exit 1  # Don't run this file directly

################################################
# Reproduce (but no glove)

./main.py train configs/base.yml configs/data/artificial-chain.yml configs/model/span-node.yml
X=7; STEP=30; ./main.py test out/${X}.exec/config.json -l out/${X}.exec/${STEP}
./main.py train configs/base.yml configs/data/artificial-chain.yml configs/model/span-node.yml -s 1

./main.py train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml

./main.py train configs/base.yml configs/data/artificial-chain.yml configs/model/span-edge.yml
X=6; STEP=30; ./main.py test out/${X}.exec/config.json -l out/${X}.exec/${STEP}

./main.py train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml

################################################
# nlpsub on those

./submit.py arti-node train configs/base.yml configs/data/artificial-chain.yml configs/model/span-node.yml
X=1.arti-node; STEP=30; ./submit.py arti-node-test test out/${X}/config.json -l out/${X}/${STEP}

./submit.py node train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml
./submit.py node-batch-10 train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/batch-size-10.yml

./submit.py arti-edge train configs/base.yml configs/data/artificial-chain.yml configs/model/span-edge.yml
X=4.arti-edge; STEP=8; ./submit.py arti-edge-test test out/${X}/config.json -l out/${X}/${STEP}

./submit.py edge train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml

################################################
# Alternative representation

./main.py train configs/base.yml configs/data/artificial-chain.yml configs/model/span-node.yml configs/addons/alt-rep.yml
X=32.exec; STEP=10; ./main.py test out/${X}/config.json -l out/${X}/${STEP}

./submit.py node-alt train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/alt-rep.yml
./submit.py edge-alt train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/alt-rep.yml

################################################
# Add glove and unkification

./submit.py node-glove train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml
./submit.py node-glove-unk train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml configs/addons/unkify.yml

./submit.py node-altmore train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/alt-rep-more.yml
./submit.py edge-altmore train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/alt-rep-more.yml
./submit.py edge-glove-unk train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml
./submit.py node-glove-unk-alt train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep.yml
./submit.py edge-glove-unk-alt train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep.yml
./submit.py node-glove-unk-altmore train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep-more.yml
./submit.py edge-glove-unk-altmore train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep-more.yml

X=21.node-glove-unk-altmore; STEP=16; ./submit.py node-glove-unk-altmore-test test out/${X}/config.json -l out/${X}/${STEP}
X=22.edge-glove-unk-altmore; STEP=24; ./submit.py edge-glove-unk-altmore-test test out/${X}/config.json -l out/${X}/${STEP}

################################################
# More experiments

./submit.py node-glove-unk-altmore train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep-more.yml -s 1
./submit.py edge-glove-unk-altmore train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep-more.yml -s 1
./submit.py node-glove-unk-altmost train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep-most.yml -s 1
./submit.py edge-glove-unk-altmost train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/addons/alt-rep-most.yml -s 1

X=25.node-glove-unk-altmore; STEP=15; ./submit.py node-glove-unk-altmore-test test out/${X}/config.json -l out/${X}/${STEP}
X=26.edge-glove-unk-altmore; STEP=22; ./submit.py edge-glove-unk-altmore-test test out/${X}/config.json -l out/${X}/${STEP}

################################################
# Ablation

./submit.py ablate-no-inside train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/no-inside.yml -s 1
./submit.py ablate-no-stern train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/no-stern.yml -s 1
./submit.py ablate-no-average train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/no-average.yml -s 1
./submit.py ablate-no-attention train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/no-attention.yml -s 1
./submit.py ablate-only-inside train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/only-inside.yml -s 1
./submit.py ablate-only-stern train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/only-stern.yml -s 1
./submit.py ablate-only-average train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/only-average.yml -s 1
./submit.py ablate-only-attention train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/only-attention.yml -s 1

X=32.ablate-no-stern; STEP=12; ./submit.py ablate-no-stern-test test out/${X}/config.json -l out/${X}/${STEP}

################################################
# Ablation again

./submit.py ablate-iva train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/iva.yml -s 1
./submit.py ablate-ival train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/ival.yml -s 2
./submit.py ablate-val train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/val.yml -s 2
./submit.py ablate-ivl train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/ivl.yml -s 2
./submit.py ablate-ial train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/ial.yml -s 2

./submit.py ablate-i train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/i.yml -s 2
./submit.py ablate-v train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/v.yml -s 2
./submit.py ablate-a train configs/base.yml configs/data/top_dataset.yml configs/model/span-edge.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/a.yml -s 2
./submit.py ablate-ival-node train configs/base.yml configs/data/top_dataset.yml configs/model/span-node.yml configs/addons/glove-200.yml configs/addons/unkify.yml configs/ablation/ival.yml -s 2

X=47.ablate-ival; STEP=16; ./submit.py ablate-ival-test test out/${X}/config.json -l out/${X}/${STEP}
