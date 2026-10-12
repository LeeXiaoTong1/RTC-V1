#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for ARG in "$@"; do
  case "$ARG" in
    --resume|--resume=*|--help|-h)
      exec bash run_w2v_v3184.sh "$@"
      ;;
  esac
done
printf '%s\n' 'V3.18.4 published-noise sources: RNNoise office/coffee + ESC-50; 100% target noisy views; local three-profile RTC.'
printf '%s\n' 'Noise/SNR/RTC settings are a documented local simulation, not exact organizer reproduction.'
exec bash run_w2v_v3184.sh \
  --noise-manifest "$PWD/assets/noise_v3184_rnnoise/manifest.json" \
  --scene-probability 1.0 \
  --rtc-policy three-profile \
  "$@"
