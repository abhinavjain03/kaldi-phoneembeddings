base="exp/make_bnf/raw_bnfeat_"
dataset="train_swbd_259890_sp_hires."
job="3"
lastIn=".ark"
lastOut=".txt"
in=$base$dataset$job$lastIn
out=$base$dataset$job$lastOut

# echo $in
# echo $out

. ./path.sh
copy-feats --compress=true ark:$in ark,t:$out

