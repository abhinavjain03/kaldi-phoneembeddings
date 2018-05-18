
base="exp/tri4_"
# dataset="test_cslu_hi"
dataset="train_swbd_259890"
base2="_phone_ali/ali."
job="2"
lastIn=".gz"
lastOut=".txt"
in=$base$dataset$base2$job$lastIn
out=$base$dataset$base2$job$lastOut

# echo $in
# echo $out

. ./path.sh
copy-int-vector "ark:gunzip -c $in |" ark,t:$out