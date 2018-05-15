. ./path.sh
. ./cmd.sh

#DIRECTORIES
exp=exp
swbd_exp=swbd_exp
swbd_models=${swbd_exp}/trainedModels_onespkmanyseg
phoneFilePath=${swbd_exp}/data/data/lang/phones.txt
mfccdir=$exp/mfcc_hires_text

#VARIABLES
nj=6
bnf_dim=300

affix=
train_stage=-10
num_targets=-1
train_sets="test_cslu_hi train_swbd_259890"


. utils/parse_options.sh


dir=$exp/nnet3/tdnn_${affix}

# train_sets="train_swbd_24000"

mfcc=1
mfccHires=1
align=1
getPhoneAlign=1
config=1
egsTrain=1
train=1

mfccdir=exp/mfcc
if [ $mfcc -eq 1 ]; then
	for x in $train_sets; do
		steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
			data/${x}_hires $exp/make_mfcc/${x} $mfccdir
		steps/compute_cmvn_stats.sh data/${x} $exp/make_mfcc/${x} $mfccdir
		utils/fix_data_dir.sh data/${x}
	done
fi

mfccdir=exp/mfcc_hires
if [ $mfccHires -eq 1 ]; then
	for x in $train_sets; do
		utils/copy_data_dir.sh data/$x data/${x}_hires
		steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
			--mfcc-config conf/mfcc_hires.conf \
			data/${x}_hires $exp/make_mfcc/${x}_hires $mfccdir
		steps/compute_cmvn_stats.sh data/${x}_hires $exp/make_mfcc/${x}_hires $mfccdir
		utils/fix_data_dir.sh data/${x}_hires
	done
fi


if [ $align -eq 1 ]; then
	for x in $train_sets; do
		dataalidir=$exp/tri4_${x}_ali
		steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
			data/$x data/lang ${swbd_models}/tri4 $dataalidir
	done
fi

if [ $getPhoneAlign -eq 1 ]; then

	for x in $train_sets; do
		dataalidir=$exp/tri4_${x}_ali
		dataphoneali=$exp/tri4_${x}_phone_ali
		$cmd JOB=1:$nj $exp/log/temp.JOB.log \
			ali-to-phones $swbd_models/tri4/final.mdl \
    		"ark:gunzip -c $dataalidir/ali.JOB.gz|" "ark:|gzip -c >$dataphoneali/ali.JOB.gz";
    done
fi


if [ $config -eq 1 ]; then

	mkdir -p $dir/configs
	
	if [ $num_targets -eq -1 ]; then
		num_targets=`wc -l $phoneFilePath | cut -d' ' -f1`
	fi

	cat <<EOF > $dir/configs/network.xconfig
	input dim=40 name=input

	# please note that it is important to have input layer with the name=input
	# as the layer immediately preceding the fixed-affine-layer to enable
	# the use of short notation for the descriptor
	# the first splicing is moved before the lda layer, so no splicing here
	relu-renorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=1024
	relu-renorm-layer name=tdnn2 dim=1024
	relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
	relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
	relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
	relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
	relu-renorm-layer name=tdnn_bn dim=$bnf_dim

	relu-renorm-layer name=prefinal-affine-1 input=tdnn_bn dim=1024
	output-layer name=output dim=${num_targets} max-change=1.5
EOF
	steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi


trainSet=train_swbd_259890
alidir=$exp/tri4_${trainSet}_phone_ali
if [ $egsTrain -eq 1 ]; then
	
	cmd=run.pl
	left_context=16
	right_context=12

	context_opts="--left-context=$left_context --right-context=$right_context"

	transform_dir=${accentalidir}
	cmvn_opts="--norm-means=false --norm-vars=false"
	extra_opts=()
	extra_opts+=(--cmvn-opts "$cmvn_opts")
	extra_opts+=(--left-context $left_context)
	extra_opts+=(--right-context $right_context)
	echo "$0: calling get_egs.sh for generating examples with alignments as output"


	steps/nnet3/get_egs_mod.sh $egs_opts "${extra_opts[@]}" \
		--num-utts-subset 300 \
		--nj $nj \
		--num-pdfs ${num_targets} \
		--samples-per-iter 200000 \
		--cmd "$cmd" \
		--frames-per-eg 8 \
		data/${trainSet}_hires ${alidir} $dir/egs || exit 1;

fi


if [ $train -eq 1 ]; then

	steps/nnet3/train_raw_dnn.py --stage=$train_stage \
		--cmd="$decode_cmd" \
		--feat.cmvn-opts="--norm-means=false --norm-vars=false" \
		--trainer.num-epochs 2 \
		--trainer.optimization.num-jobs-initial 9 \
		--trainer.optimization.num-jobs-final 12 \
		--trainer.optimization.initial-effective-lrate 0.0017 \
		--trainer.optimization.final-effective-lrate 0.00017 \
		--egs.dir $dir/egs \
		--cleanup.preserve-model-interval 20 \
		--use-gpu true \
		--use-dense-targets false \
		--feat-dir=data/${trainSet}_hires \
		--reporting.email="$reporting_email" \
		--dir $dir  || exit 1;

fi