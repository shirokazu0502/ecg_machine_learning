#!/bin/bash

# パラメータ設定
MODEL_TYPES=("unet")
BETAS=(1.0)
DEPTHS=(2 3 4 5 6)

# ループ処理
for model in "${MODEL_TYPES[@]}"; do
    for beta in "${BETAS[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            
            echo "================================================================"
            echo "STARTING: model_type=$model, beta=$beta, unet_depth=$depth"
            echo "================================================================"
            
            # Python実行
            python3 grid_search_and_best_param_train.py \
                --model_type "$model" \
                --beta "$beta" \
                --unet_depth "$depth"
            
            echo "FINISHED: model_type=$model, beta=$beta, unet_depth=$depth"
            echo ""
            
        done
    done
done

echo "全てのグリッドサーチが完了しました。"