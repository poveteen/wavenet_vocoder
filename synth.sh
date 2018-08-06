
python synthesis.py \
       checkpoints/checkpoint_step000410000_ema.pth \
       generated/eval \
       --preset=presets/seoul_corpus.json \
       --conditional="../tacotron2/tacotron_output/eval/speech-mel-00000(mel-tar).npy" \
       --speaker-id=113 \
       --max-abs-value=4.0