export PREFIX=$NUT_VOLUME_MAIN_PATH/logs/`date '+started%y-%m-%d_%Hh%Mm%Ss'`
echo "Will pretrain and train the network, end to end. Will generate two images: reconstruction of images coming from the testing set and from the training set."
echo "This should take about one day to complete on a modern GPU. Faster training method would leverage pretrained models of VGG or Alex Net."
echo "See log files: $PREFIX*** (one for each training step)"
pwd
mkdir -p $NUT_VOLUME_MAIN_PATH/snapshots
mkdir -p $NUT_VOLUME_MAIN_PATH/logs
mkdir -p $PREFIX.data
cp -R proto $PREFIX.data/proto
cp -R nut.yml $PREFIX.data/

caffe train --solver="proto/solverPretrainConvCifar10.prototxt" > $PREFIX.step000_pretrainConvCifar10.txt 2>&1
caffe train --solver="proto/solverPretrainConv.prototxt" --weights="snapshots/pretraining_conv_cifar10_iter_7000.caffemodel" > $PREFIX.step010_pretrainConv.txt 2>&1
caffe train --solver="proto/solverReconstructIncremental1v2.prototxt" --weights="snapshots/pretraining_conv_iter_40000.caffemodel" > $PREFIX.step020_reconstructInc1.txt 2>&1  # --iterations=2000
caffe train --solver="proto/solverReconstructIncremental2v2.prototxt" --weights="snapshots/reconstructing_incremental_1_iter_2000.caffemodel"  > $PREFIX.step030_reconstructInc2.txt 2>&1  #  --iterations=2000
caffe train --solver="proto/solverReconstructFull.prototxt" --weights="snapshots/reconstructing_incremental_2_iter_2000.caffemodel"  > $PREFIX.step040_recontructFull.txt 2>&1 # --iterations=15000
caffe train --solver="proto/solverReconstructFull_FC0.prototxt" --weights="snapshots/reconstructing_full_extra_iter_15000.caffemodel"  > $PREFIX.step050_reconstructFullFC0.txt 2>&1 # --iterations=30000
caffe test  --model=proto/reconstructFull_FC0.prototxt --weights=snapshots/reconstructing_full_extra_FC0_iter_30000.caffemodel  > $PREFIX.step060_test.txt 2>&1 # --iterations=50

python $NUT_VOLUME_MAIN_PATH/VisualizeReconstructionOfImage.py snapshots/reconstructing_full_extra_FC0_iter_30000.caffemodel proto/reconstructFull_FC0.prototxt > $PREFIX.step070_visualizeTrain.txt 2>&1 # --iterations=50
python $NUT_VOLUME_MAIN_PATH/VisualizeReconstructionOfImage.py snapshots/reconstructing_full_extra_FC0_iter_30000.caffemodel proto/reconstructFull_FC0_test.prototxt > $PREFIX.step070_visualizeTest.txt 2>&1 # --iterations=50

# echo "Will pretrain and train the network, end to end. Will generate two images: reconstruction of images coming from the testing set and from the training set."
# echo "This should take about one day to complete on a modern GPU. This is a very naive pretraining/training implementation, which is kept here since it proved to yield consistent results."
# echo "Smarter and faster training involves pretraining on CIFAR-10 before CIFAR-100. Fastest method would leverage pretrained models of VGG or Alex Net."
# echo "See log files: $PREFIX*** (one for each training step)"
# mkdir -p $NUT_VOLUME_MAIN_PATH/logs
# # USELESS STEP (TESTED with old caffe) - caffe train --solver="proto/solverPretrain.prototxt" > $PREFIX.step000_pretrain.txt 2>&1
#   # - caffe train --solver="proto/solverPretrain.prototxt" --snapshot="snapshots/pretraining__iter_2000.solverstate"  # just to go on in screen during the night
#   # accu from 1.1% to 1.5% after 2000 iterations
#   # 5000: 1.3
#   # 10000: 11
#   # 16000: 22
#   # 20000: 25
#   # 30000: 31
#   # 40000: 33
#   # 89000: 35 (18h)
# # - caffe train --solver="proto/solverPretrainConv.prototxt" --weights="snapshots/pretraining__iter_40000.caffemodel" > $PREFIX.step010_pretrainConv.txt 2>&1
# caffe train --solver="proto/solverPretrainConv.prototxt"  > $PREFIX.step010_pretrainConv.txt 2>&1
# # cat $PREFIX.step010_pretrainConv.txt
#   # - caffe train --solver="proto/solverPretrainConv.prototxt" --snapshot="snapshots/pretraining_conv_iter_7000.solverstate" > $PREFIX.step011_solverPretrainConvCont.txt 2>&1 # just to go on for the week end
#     # 8h from 7000 to 40000:
#     # Iteration 39990, loss = 2.52498
#     #     Train net output #0: prob = 2.52498 (* 1 = 2.52498 loss)
#     # Iteration 39990, lr = 0.0001
#     # Snapshotting to snapshots/pretraining_conv_iter_40000.caffemodel
#     # Snapshotting solver state to snapshots/pretraining_conv_iter_40000.solverstate
#     # Iteration 40000, loss = 2.73032
#     # Iteration 40000, Testing net (#0)
#     #     Test net output #0: accurLayerWinnerTop1 = 0.323333
#     #     Test net output #1: accurLayerWinnerTop5 = 0.615893
#     #     Test net output #2: prob = 2.76338 (* 1 = 2.76338 loss)
# # - BAD learning rate : caffe train --solver="proto/solverReconstructIncremental1.prototxt" --weights="snapshots/pretraining_conv_iter_40000.caffemodel" > $PREFIX.step020_reconstructInc1.txt 2>&1  # --iterations=2000
#     # Nan :
#     #   Iteration 0, Testing net (#0)
#     #       Test net output #0: L2_loss = 4.07108e+08 (* 1 = 4.07108e+08 loss)
#     #   Iteration 0, loss = 4.0589e+08
#     #       Train net output #0: L2_loss = 4.0589e+08 (* 1 = 4.0589e+08 loss)
#     #   Iteration 0, lr = 1e-07
#     #   Iteration 5, loss = nan
#   #       Train net output #0: L2_loss = nan (* 1 = nan loss)
# caffe train --solver="proto/solverReconstructIncremental1v2.prototxt" --weights="snapshots/pretraining_conv_iter_40000.caffemodel" > $PREFIX.step020_reconstructInc1.txt 2>&1  # --iterations=2000
#     # Iteration 0, Testing net (#0)
#     #     Test net output #0: L2_loss = 4.07729e+08 (* 1 = 4.07729e+08 loss)
#     # Iteration 0, loss = 4.06565e+08
#     #     Train net output #0: L2_loss = 4.06565e+08 (* 1 = 4.06565e+08 loss)
#     # Iteration 0, lr = 5e-10
#     # ...
#     # Iteration 1995, loss = 1.25529e+07
#     #     Train net output #0: L2_loss = 1.25529e+07 (* 1 = 1.25529e+07 loss)
#     # Iteration 1995, lr = 5e-10
#     # Snapshotting to snapshots/reconstructing_incremental_1_iter_2000.caffemodel
#     # Snapshotting solver state to snapshots/reconstructing_incremental_1_iter_2000.solverstate
#     # Iteration 2000, loss = 1.2672e+07
#     # Iteration 2000, Testing net (#0)
#     #     Test net output #0: L2_loss = 1.29921e+07 (* 1 = 1.29921e+07 loss)
# caffe train --solver="proto/solverReconstructIncremental2v2.prototxt" --weights="snapshots/reconstructing_incremental_1_iter_2000.caffemodel"  > $PREFIX.step030_reconstructInc2.txt 2>&1  #  --iterations=2000
#     # Iteration 0, Testing net (#0)
#     #     Test net output #0: L2_loss = 2.1654e+08 (* 1 = 2.1654e+08 loss)
#     # Iteration 0, loss = 2.15841e+08
#     #     Train net output #0: L2_loss = 2.15841e+08 (* 1 = 2.15841e+08 loss)
#     # Iteration 0, lr = 5e-10
#     # ...
#     # Iteration 1995, loss = 5.96489e+06
#     #     Train net output #0: L2_loss = 5.96489e+06 (* 1 = 5.96489e+06 loss)
#     # Iteration 1995, lr = 5e-10
#     # Snapshotting to snapshots/reconstructing_incremental_2_iter_2000.caffemodel
#     # Snapshotting solver state to snapshots/reconstructing_incremental_2_iter_2000.solverstate
#     # Iteration 2000, loss = 6.0563e+06
#     # Iteration 2000, Testing net (#0)
#     #     Test net output #0: L2_loss = 6.17771e+06 (* 1 = 6.17771e+06 loss)
# caffe train --solver="proto/solverReconstructFull.prototxt" --weights="snapshots/reconstructing_incremental_2_iter_2000.caffemodel"  > $PREFIX.step040_recontructFull.txt 2>&1 # --iterations=15000
# caffe train --solver="proto/solverReconstructFull_FC0.prototxt" --weights="snapshots/reconstructing_full_extra_iter_15000.caffemodel"  > $PREFIX.step050_reconstructFullFC0.txt 2>&1 # --iterations=30000
# caffe test  --model=proto/reconstructFull_FC0.prototxt --weights=snapshots/reconstructing_full_extra_FC0_iter_30000.caffemodel  > $PREFIX.step060_test.txt 2>&1 # --iterations=50
