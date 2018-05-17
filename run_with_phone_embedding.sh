. ./path.sh
. ./cmd.sh

#DIRECTORIES
exp=exp
swbd_exp=swbd_exp
swbd_models=${swbd_exp}/trainedModels_onespkmanyseg
phoneFilePath=${swbd_exp}/data/data/lang/phones.txt
mfccdir=$exp/mfcc_hires_text

bnfNnetModelDir=$exp/nnet3/tdnn_swbd259890_bnf300_6layers

#VARIABLES
nj=6
bnf_dim=300
cmd=run.pl


affix=
train_stage=-10
train_sets="test_cslu_hi train_swbd_259890"


. utils/parse_options.sh


dir=$exp/nnet3/tdnn_${affix}

# train_sets="train_swbd_24000"

mfccBase=0
mfccHiresBase=0
mfccSp=0
mfccHiresSp=0
get_bnf_features=0
append_bnf_mfcc=0
align=0
ivector=0
config=0
egsTrain=1
train=1
decode=0
wer=0


mfccdir=exp/mfcc
if [ $mfccBase -eq 1 ]; then
	for x in $train_sets; do
		steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
			data/${x} $exp/make_mfcc/${x} $mfccdir
		steps/compute_cmvn_stats.sh data/${x} $exp/make_mfcc/${x} $mfccdir
		utils/fix_data_dir.sh data/${x}
	done
fi

mfccdir=exp/mfcc_hires
if [ $mfccHiresBase -eq 1 ]; then
	for x in $train_sets; do
		utils/copy_data_dir.sh data/$x data/${x}_hires
		steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
			--mfcc-config conf/mfcc_hires.conf \
			data/${x}_hires $exp/make_mfcc/${x}_hires $mfccdir
		steps/compute_cmvn_stats.sh data/${x}_hires $exp/make_mfcc/${x}_hires $mfccdir
		utils/fix_data_dir.sh data/${x}_hires
	done
fi

mfccdir=exp/mfcc_perturbed
if [ $mfccSp -eq 1 ]; then
	for x in $train_sets; do
		utils/perturb_data_dir_speed.sh 0.9 data/${x} data/temp1
		utils/perturb_data_dir_speed.sh 1.1 data/${x} data/temp2
		utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${x} data/temp0
		utils/combine_data.sh data/${x}_sp data/temp0 data/temp1 data/temp2
		rm -r data/temp0 data/temp1 data/temp2
		utils/validate_data_dir.sh --no-feats --no-text data/${x}_sp


		steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc.conf --cmd "$train_cmd" \
				data/${x}_sp exp/make_mfcc/${x} $mfccdir
		steps/compute_cmvn_stats.sh data/${x}_sp exp/make_mfcc/$x $mfccdir
		utils/fix_data_dir.sh data/${x}_sp
	done
fi

mfccdir=exp/mfcc_perturbed_hires
if [ $mfccHiresSp -eq 1 ]; then
	for x in $train_sets; do
		utils/copy_data_dir.sh data/${x}_sp data/${x}_sp_hires
		steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
			--mfcc-config conf/mfcc_hires.conf \
			data/${x}_sp_hires $exp/make_mfcc/${x}_hires $mfccdir
		steps/compute_cmvn_stats.sh data/${x}_sp_hires $exp/make_mfcc/${x}_hires $mfccdir
		utils/fix_data_dir.sh data/${x}_sp_hires
	done
fi


if [ $get_bnf_features -eq 1 ]; then
	for x in $train_sets; do
		bnf_feat_dir=data/${x}_sp_bnf
		steps/nnet3/make_bottleneck_features_from_singletask.sh \
			--nj $nj \
			--use-gpu true \
			--cmd "$train_cmd" \
			tdnn_bn.renorm data/${x}_sp_hires data/${x}_sp_bnf \
			$bnfNnetModelDir $exp/make_bnf/${x}_sp_hires $exp/make_bnf
	done
fi

trainSet=train_swbd_259890
bnf_feat_dir=data/${trainSet}_sp_bnf
appended_dir=data/${trainSet}_mfcc_bnf_appended_sp
#appended_dir=$bnf_feat_dir
dump_bnf_dir=exp/append_mfcc_bnf
if [ $append_bnf_mfcc -eq 1 ]; then

    steps/append_feats.sh \
        --cmd "$train_cmd" \
        --nj $nj \
      $bnf_feat_dir data/${trainSet}_sp_hires $appended_dir \
      exp/append_hires_mfcc_bnf/${trainSet}_sp $dump_bnf_dir || exit 1;
    steps/compute_cmvn_stats.sh $appended_dir \
        exp/make_cmvn_mfcc_bnf $dump_bnf_dir || exit 1;

fi

dataalidir=$exp/tri4_${trainSet}_sp_ali
if [ $align -eq 1 ]; then
	#for x in $train_sets; do
	# dataalidir=$exp/tri4_${trainSet}_sp_ali
	steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
		data/${trainSet}_sp data/lang ${swbd_models}/tri4 $dataalidir
	#done
fi

online_ivector_dir=$exp/nnet3/ivectors_${trainSet}_sp_hires
if [ $ivector -eq 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${trainSet}_sp_hires ${swbd_models}/nnet_online/extractor $online_ivector_dir || exit 1;
fi


if [ $config -eq 1 ]; then

  mkdir -p $dir/configs

  num_pdfs=`tree-info ${dataalidir}/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  feat_dim=`feat-to-dim scp:${appended_dir}/feats.scp -`


  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$feat_dim name=input
  

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(-2,-1,0,1,2, ReplaceIndex(ivector, t, 0)) dim=1024
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim

  relu-renorm-layer name=prefinal-affine-1 input=tdnn_bn dim=1024
  output-layer name=output dim=${num_pdfs} max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

if [ $egsTrain -eq 1 ]; then

  cmd=run.pl
  left_context=16
  right_context=12

  context_opts="--left-context=$left_context --right-context=$right_context"

    transform_dir=${dataalidir}
    cmvn_opts="--norm-means=false --norm-vars=false"
    extra_opts=()
    extra_opts+=(--cmvn-opts "$cmvn_opts")
    extra_opts+=(--online-ivector-dir ${online_ivector_dir})
    extra_opts+=(--left-context $left_context)
    extra_opts+=(--right-context $right_context)
    echo "$0: calling get_egs.sh for generating examples with alignments as output"


  steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
    --num-utts-subset 300 \
    --nj $nj \
    --nj-shuffle 1 \
      --samples-per-iter 400000 \
      --shuffle 0 \
      --cmd "$cmd" \
      --frames-per-eg 8 \
      $appended_dir ${dataalidir} $dir/egs || exit 1;

fi


# if [ $shuffleonebyone -eq 1 ]; then
# 	for i in `seq 107`; do
# 		input=`echo "ark:cat exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.1.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.2.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.3.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.4.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.5.$i.ark exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.6.$i.ark|"`
# 		echo "###################Shuffling Set $i##########################"
# 		nnet3-shuffle-egs --srand=$i "$input" ark:exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs.$i.ark 
# 	 	echo "###################Removing Set $i##########################"
# 		for x in $(seq 6); do
# 	 		rm exp/nnet3/tdnn_phone_embed300_ivecs_6layers/egs/egs_orig.$x.$i.ark
# 		done
# 	done
# fi



if [ $train -eq 1 ]; then

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir $dir/egs \
    --feat.online-ivector-dir ${online_ivector_dir} \
    --cleanup.preserve-model-interval 20 \
    --use-gpu true \
    --ali-dir $dataalidir \
    --lang data/lang \
    --feat-dir=$appended_dir \
    --reporting.email="$reporting_email" \
    --dir $dir  || exit 1;

fi

graph_dir=exp/tri4/graph_sw1_tg
if [ $decode -eq 1 ]; then

  mfccdev=0
  calculatehiresdev=0
  ivectordev=0
  get_bnf_features_dev=0
  append_bnf_mfcc=0
  decodedev=1
  # devsets="cv_test_onlyindian cv_dev_nz cv_test_onlynz"
  devsets="cv_test_onlyindian"

  mfccdir=exp/mfcc
  if [ $mfccdev -eq 1 ]; then
    for x in $devsets; do
        steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf --cmd "$train_cmd" \
                           data/$x exp/make_mfcc/$x $mfccdir
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh data/$x
    done

    echo "============================================="
    echo "MFCCs DONE!!"
    echo "============================================="
  fi

  #calculate hires mfccs
  mfcchiresdir=exp/mfcc_hires
  if [ $calculatehiresdev -eq 1 ]; then
    for x in $devsets; do
      utils/copy_data_dir.sh data/$x data/${x}_hires
      steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_16k.conf \
            --cmd "$train_cmd" data/${x}_hires exp/make_mfcc/${x}_hires $mfcchiresdir;
        steps/compute_cmvn_stats.sh data/${x}_hires exp/make_mfcc/${x}_hires $mfcchiresdir;

      utils/fix_data_dir.sh data/${x}_hires
    done

    echo "============================================="
    echo "HIRES MFCCs DONE!!"
    echo "============================================="
  fi

  
  if [ $ivectordev -eq 1 ]; then
    for x in $devsets; do
      online_ivector_dir=exp/nnet3/ivectors_${x}
            steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        data/$x  exp/nnet3/extractor ${online_ivector_dir} || exit 1;

        #       online_ivector_dir=exp/nnet3/ivectors_${x}_hires
        #     steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        # data/${x}_hires  exp/alreadyTrainedModelsOnSwbd/nnet_online/extractor ${online_ivector_dir} || exit 1;



        echo "============================================="
        echo "CALCULATION OF IVECTORS DONE!!!!"
        echo "============================================="
    done
  fi

  
  
  if [ $get_bnf_features_dev -eq 1 ]; then

    for x in $devsets; do
      bnf_feat_dir=data/${x}_bnf
      if [ $singletask -eq 1 ]; then
        steps/nnet3/make_bottleneck_features_from_singletask.sh \
        --nj $nj \
        --use-gpu true \
        --cmd "$train_cmd" \
            tdnn_bn.renorm data/${x}_hires $bnf_feat_dir \
            $bnfNnetModelDir exp/make_bnf/${x} exp/make_bnf
      else
              steps/nnet3/make_bottleneck_features.sh \
      --nj $nj \
      --use-gpu true \
      --cmd "$train_cmd" \
          acc_btn.renorm data/${x}_hires $bnf_feat_dir \
          $bnfNnetModelDir exp/make_bnf/${x} exp/make_bnf
        fi

    done

  fi

  
  dump_bnf_dir=exp/append_mfcc_bnf
  if [ $append_bnf_mfcc -eq 1 ]; then

    for x in $devsets; do 
    	bnf_feat_dir=data/${x}_bnf
      	appended_dir=data/${x}_mfcc_bnf_appended
      # utils/fix_data_dir.sh $bnf_feat_dir
      steps/append_feats.sh \
          --cmd "$train_cmd" \
          --nj $nj \
        $bnf_feat_dir data/${x}_hires $appended_dir \
        exp/append_hires_mfcc_bnf/${x} $dump_bnf_dir || exit 1;
      steps/compute_cmvn_stats.sh $appended_dir \
          exp/make_cmvn_mfcc_bnf $dump_bnf_dir || exit 1;
          utils/fix_data_dir.sh $appended_dir
    done

  fi



  if [ $decodedev -eq 1 ]; then
    for decode_set in $devsets; do
    	bnf_feat_dir=data/${decode_set}_bnf
      # appended_dir=data/${decode_set}_mfcc_bnf_appended
      appended_dir=data/${decode_set}_hires
      num_jobs=`cat $appended_dir/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
        $graph_dir $appended_dir $dir/decode_${decode_set}_new || exit 1;
      
    done
  fi
fi


if [ $wer -eq 1 ]; then
  #for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 
  for x in ${dir}/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
  #for x in exp/*/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
fi
