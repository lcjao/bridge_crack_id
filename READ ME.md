使用方法：
cd bridge_crack_id
# 1. 生成数据
python scripts/01_generate_data.py --n_samples 10000 --output outputs/data/training_data.npz
python scripts/01_generate_data.py --n_samples 1000 --output outputs/data/training_data.npz
python scripts/01_generate_data.py --n_samples 100 --output outputs/data/training_data.npz 
# 2. 训练模型
python scripts/02_train_model.py --data outputs/data/training_data.npz --epochs 200
python scripts/02_train_model.py --data outputs/data/training_data.npz --epochs 200 --plot
python scripts/02_train_lstm_model.py --data outputs/data/test_data.npz --model outputs/models/lstm_model.pth --epochs 50 --batch_size 3 --seq_len 20

# 3. 运行推理
python scripts/03_run_inference.py --model outputs/models/cracknet.json --data outputs/data/training_data.npz
python scripts/03_run_inference.py --model outputs/models/cracknet.json --data outputs/data/training_data.npz --plot
python scripts/03_run_lstm_inference.py --model outputs/models/lstm_model.pth --input outputs/data/test_data.npz --output outputs/predictions.npy
