. ./path.sh
. ./cmd.sh

for i in `seq 107`; do
	if [ $i -eq 1 ]; then
		continue
	fi
	input=`echo "ark:cat exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.1.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.2.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.3.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.4.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.5.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.6.$i.ark|"`
	echo "###################Shuffling Set $i##########################"
	nnet3-shuffle-egs --srand=$i "$input" ark:exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs.$i.ark 
 	echo "###################Removing Set $i##########################"
	for x in $(seq 6); do
 		rm exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.$x.$i.ark
	done
done



